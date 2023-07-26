import time

from langchain.callbacks import get_openai_callback
from langchain.chains.openai_functions import create_openai_fn_chain, create_structured_output_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from chatbot_step import MessageGenerationType
from constants import ContextKeys
from helpers import generate_answers_from_documents_for_question, get_most_relevant_docs_in_vector_store_for_answering_question, get_documents_chunks_for_files
from llm_utils import stream_from_llm_generation
from settings import GPT_MODEL
from openai_functions_utils import function_for_comprehensiveness_check, get_json_schema_for_comprehensiveness_check
from prompts import (
    get_prompt_template_for_generating_original_answer,
    get_prompt_template_for_comprehensiveness_check_openai_functions,
    get_prompt_template_for_generating_answer_to_implicit_question,
    get_prompt_template_for_generating_final_answer
)


def generate_answer_to_question(vars: dict) -> str:
    '''Generate an answer to a grant application question.'''

    documents_chunks = get_documents_chunks_for_files(vars[ContextKeys.PRIOR_GRANT_APPLICATIONS])
    vector_store = Chroma.from_documents(documents=documents_chunks, embedding=OpenAIEmbeddings(client=None))

    if vars[ContextKeys.APPLICATION_QUESTION] == '':
        vars[ContextKeys.APPLICATION_ANSWER] = ''
        print(f'No application question provided, skipping generation of answer.\n')
        return 'No answer generated due to missing application question.'

    question_for_prompt = vars[ContextKeys.APPLICATION_QUESTION]
    most_relevant_documents = get_most_relevant_docs_in_vector_store_for_answering_question(
        vector_store=vector_store, question=question_for_prompt, n_results=1)

    vars[ContextKeys.MOST_RELEVANT_DOCUMENTS] = most_relevant_documents

    if vars[ContextKeys.WORD_LIMIT] and vars[ContextKeys.WORD_LIMIT].isdigit():
        question_for_prompt += f' ({vars[ContextKeys.WORD_LIMIT]} words)'

    chat_openai = ChatOpenAI(client=None, model=GPT_MODEL, temperature=0)

    vars[ContextKeys.APPLICATION_ANSWER] = generate_answers_from_documents_for_question(
        most_relevant_documents, chat_openai, question_for_prompt)[0]

    print(f'Final answer for application question "{vars[ContextKeys.APPLICATION_QUESTION]}":\n\n{vars[ContextKeys.APPLICATION_ANSWER]}\n')

    return f'Generated answer ({len(vars[ContextKeys.APPLICATION_ANSWER].split())} words) to "{vars[ContextKeys.APPLICATION_QUESTION]}":\n\n{vars[ContextKeys.APPLICATION_ANSWER]}'


def generate_answer_to_question_stream(vars: dict) -> MessageGenerationType:
    '''Generate and stream an answer to a grant application question by streaming tokens from the LLM.'''

    documents_chunks = get_documents_chunks_for_files(vars[ContextKeys.PRIOR_GRANT_APPLICATIONS])
    vector_store = Chroma.from_documents(documents=documents_chunks, embedding=OpenAIEmbeddings(client=None))

    if vars[ContextKeys.APPLICATION_QUESTION] == '':
        vars[ContextKeys.APPLICATION_ANSWER] = ''
        print(f'No application question provided, skipping generation of answer.\n')
        return 'No answer generated due to missing application question.'

    question_for_prompt = vars[ContextKeys.APPLICATION_QUESTION]
    most_relevant_documents = get_most_relevant_docs_in_vector_store_for_answering_question(
        vector_store=vector_store, question=question_for_prompt, n_results=1)

    vars[ContextKeys.MOST_RELEVANT_DOCUMENTS] = most_relevant_documents

    if vars[ContextKeys.WORD_LIMIT] and vars[ContextKeys.WORD_LIMIT].isdigit():
        question_for_prompt += f' ({vars[ContextKeys.WORD_LIMIT]} words)'

    final_answer = ''
    print(f'-------- Streaming tokens in generate_answer_to_question_stream --------\n')
    for next_token, response in stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_original_answer(),
        chain_type='qa_chain',
        verbose=True,
        docs=most_relevant_documents,
        question=question_for_prompt
    ):
        print(next_token)
        final_answer = response
        yield response

    vars[ContextKeys.APPLICATION_ANSWER] = final_answer
    word_count = len(vars[ContextKeys.APPLICATION_ANSWER].split())

    yield [final_answer, f'Generated answer contains {word_count} words.']

    print(f'Generated answer ({word_count} words) '
          f'to "{vars[ContextKeys.APPLICATION_QUESTION]}":\n\n{vars[ContextKeys.APPLICATION_ANSWER]}\n')


def check_for_comprehensiveness(vars: dict, use_json_schema: bool = False) -> str:
    '''Check for comprehensiveness of an answer to a grant application question using OpenAI functions.'''

    if vars[ContextKeys.CHECK_COMPREHENSIVENESS] != 'Yes':
        return 'Skipping check for comprehensiveness.'

    chat_openai = ChatOpenAI(client=None, model=GPT_MODEL, temperature=0)
    prompt = get_prompt_template_for_comprehensiveness_check_openai_functions()

    with get_openai_callback() as cb:
        chain = (
            create_structured_output_chain(get_json_schema_for_comprehensiveness_check(), chat_openai, prompt, verbose=True)
                if use_json_schema else
            create_openai_fn_chain([function_for_comprehensiveness_check], chat_openai, prompt, verbose=True)
        )
        response = chain.run(question=vars[ContextKeys.APPLICATION_QUESTION], answer=vars[ContextKeys.APPLICATION_ANSWER])
        print(f'Summary info OpenAI callback:\n{cb}\n')

    print(f'Response for check of comprehensiveness:\n\n{response}\n')

    assert type(response) == dict
    vars[ContextKeys.MISSING_INFORMATION] = response['missing_information']

    vars[ContextKeys.IMPLICIT_QUESTIONS] = []
    questions = response['implicit_questions']
    if type(questions) is dict:
        # if response['implicit_questions'] is a dict, then just set vars[ContextKeys.IMPLICIT_QUESTIONS] to the values of that dict
        vars[ContextKeys.IMPLICIT_QUESTIONS] = [q for q in questions.values()] # type: ignore
    elif type(questions) is list:
        vars[ContextKeys.IMPLICIT_QUESTIONS] = [q if type(q) is str else q['question'] for q in questions] # type: ignore
    else:
        print(f'Unexpected type for implicit questions: {type(questions)}\n')
        vars[ContextKeys.IMPLICIT_QUESTIONS] = []

    if len(vars[ContextKeys.IMPLICIT_QUESTIONS]) > 0:
        print(f'Implicit questions:')
        for i, q in enumerate(vars[ContextKeys.IMPLICIT_QUESTIONS]):
            print(f'Question {i + 1}: {q}')
        print()

    return vars[ContextKeys.MISSING_INFORMATION]


def generate_answers_for_implicit_questions(vars: dict) -> list[str] | None:
    '''Generate answers for implicit questions to be answered to make the answer comprehensive.'''

    if vars[ContextKeys.CHECK_COMPREHENSIVENESS] != 'Yes' or ContextKeys.IMPLICIT_QUESTIONS not in vars:
        return None

    vars[ContextKeys.ANSWERS_TO_IMPLICIT_QUESTIONS] = []
    answers = []
    for i, question in enumerate(vars[ContextKeys.IMPLICIT_QUESTIONS]):
        with get_openai_callback() as cb:
            chain = load_qa_chain(
                llm=ChatOpenAI(client=None, model=GPT_MODEL, temperature=0),
                chain_type='stuff',
                verbose=False,
                prompt=get_prompt_template_for_generating_answer_to_implicit_question(),
                streaming=True
            )
            answer = chain.run(input_documents=vars[ContextKeys.MOST_RELEVANT_DOCUMENTS], question=question)

            if 'Not enough information' not in answer:
                vars[ContextKeys.ANSWERS_TO_IMPLICIT_QUESTIONS].append(answer)
                print(f'Generated answer for question #{i+1} "{question}":\n\n{answer}\n')
            else:
                print(f'Excluding answer for question #{i+1} as not enough information provided to answer question "{question}"\n')

            print(f'Summary info OpenAI callback:\n{cb}\n\n')
            answers.append(f'({i+1}) {question}\n\n{answer}')

    return answers

def generate_answers_for_implicit_questions_stream(vars: dict) -> MessageGenerationType:
    '''Generate and stream answers for implicit questions to be answered to make the answer comprehensive.'''

    if vars[ContextKeys.CHECK_COMPREHENSIVENESS] != 'Yes' or ContextKeys.IMPLICIT_QUESTIONS not in vars:
        return

    vars[ContextKeys.ANSWERS_TO_IMPLICIT_QUESTIONS] = []
    answers: list[str] = []
    for i, question in enumerate(vars[ContextKeys.IMPLICIT_QUESTIONS]):
        start_of_sentence_for_answer = f'({i+1}) {question}'
        answers.append(start_of_sentence_for_answer)
        time.sleep(0.5)
        yield answers
        print(f'-------- Streaming tokens in generate_answers_for_implicit_questions_stream --------\n')
        for next_token, response in stream_from_llm_generation(
            prompt=get_prompt_template_for_generating_answer_to_implicit_question(),
            chain_type='qa_chain',
            model='gpt-3.5-turbo',
            verbose=False,
            docs=vars[ContextKeys.MOST_RELEVANT_DOCUMENTS],
            question=question
        ):
            print(next_token)
            answers[-1] = f'{start_of_sentence_for_answer}\n\n{response}'
            yield answers

        if 'Not enough information' not in answers[-1]:
            vars[ContextKeys.ANSWERS_TO_IMPLICIT_QUESTIONS].append(answers[-1])
            print(f'Generated answer for question #{i+1} "{question}":\n\n{answers[-1]}\n')
        else:
            print(f'Excluding answer for question #{i+1} as not enough information provided to answer question "{question}"\n')


def generate_final_answer(vars: dict) -> list[str] | None:
    '''Generate a final answer to a grant application question.'''

    if vars[ContextKeys.CHECK_COMPREHENSIVENESS] != 'Yes' or ContextKeys.IMPLICIT_QUESTIONS not in vars:
        return None

    with get_openai_callback() as cb:
        chain = LLMChain(
            llm=ChatOpenAI(client=None, model=GPT_MODEL, temperature=0),
            prompt=get_prompt_template_for_generating_final_answer(),
            verbose=True
        )
        response = chain.run(
            question=vars[ContextKeys.APPLICATION_QUESTION],
            answers_to_implicit_questions='\n\n'.join(vars[ContextKeys.ANSWERS_TO_IMPLICIT_QUESTIONS]),
            word_limit=vars[ContextKeys.WORD_LIMIT],
            original_answer=vars[ContextKeys.APPLICATION_ANSWER]
        )
        print(f'Summary info OpenAI callback:\n{cb}\n')

    print(f'Final answer for application question "{vars[ContextKeys.APPLICATION_QUESTION]}":\n\n{response}\n')

    number_of_implicit_questions = len(vars[ContextKeys.ANSWERS_TO_IMPLICIT_QUESTIONS])
    implicit_questions_integration_summary = (
        f'answers to {number_of_implicit_questions} implicit questions'
            if number_of_implicit_questions > 1 else
        ('answer to 1 implicit question'
            if number_of_implicit_questions == 1 else
        'answers to none of the implicit questions'))

    return [(f'Here is the final answer ({len(response.split())} words) to "{vars[ContextKeys.APPLICATION_QUESTION]}" '
             f'after integrating {implicit_questions_integration_summary}:\n\n{response}'),
            (f'For reference, here is the original answer generated ({len(vars[ContextKeys.APPLICATION_ANSWER].split())} words) '
             f'prior to the comprehensiveness checker:\n\n{vars[ContextKeys.APPLICATION_ANSWER]}')]

def generate_final_answer_stream(vars: dict) -> MessageGenerationType:
    '''Generate and stream a final answer to a grant application question.'''

    if vars[ContextKeys.CHECK_COMPREHENSIVENESS] != 'Yes' or ContextKeys.IMPLICIT_QUESTIONS not in vars:
        return
    elif len(vars[ContextKeys.IMPLICIT_QUESTIONS]) == 0:
        return f'No final answer generated due to having no implicit questions.'
    elif len(vars[ContextKeys.ANSWERS_TO_IMPLICIT_QUESTIONS]) == 0:
        return f'No final answer generated due having no answers to any of the implicit questions.'

    num_implicit_questions = len(vars[ContextKeys.ANSWERS_TO_IMPLICIT_QUESTIONS])
    implicit_questions_integration_summary = (
        f'answers to {num_implicit_questions} implicit questions' if num_implicit_questions > 1 else (
            'answer to 1 implicit question' if num_implicit_questions == 1 else
            'answers to none of the implicit questions'))

    start_of_sentence_for_final_answer = (
        f'Here is the final answer to "{vars[ContextKeys.APPLICATION_QUESTION]}" '
        f'after integrating {implicit_questions_integration_summary}')
    final_answer = start_of_sentence_for_final_answer
    final_llm_response = ''

    print(f'-------- Streaming tokens in generate_final_answer_stream --------\n')
    for next_token, llm_response in stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_final_answer(),
        chain_type='llm_chain',
        verbose=True,
        question=vars[ContextKeys.APPLICATION_QUESTION],
        answers_to_implicit_questions='\n\n'.join(vars[ContextKeys.ANSWERS_TO_IMPLICIT_QUESTIONS]),
        word_limit=vars[ContextKeys.WORD_LIMIT],
        original_answer=vars[ContextKeys.APPLICATION_ANSWER]
    ):
        print(next_token)
        final_llm_response = llm_response
        final_answer = f'{start_of_sentence_for_final_answer}:\n\n{llm_response}'
        yield final_answer

    print(f'Final answer for application question "{vars[ContextKeys.APPLICATION_QUESTION]}":\n\n{final_llm_response}\n')

    comparison_answer = (
        f'The word limit requested was {vars[ContextKeys.WORD_LIMIT]} words.\n'
        f'The new answer contains {len(final_llm_response.split())} words.\n'
        f'For reference, the original answer, which contains {len(vars[ContextKeys.APPLICATION_ANSWER].split())} words, '
        f'is the following:\n\n{vars[ContextKeys.APPLICATION_ANSWER]}')

    time.sleep(0.5)
    yield [final_answer, comparison_answer]
    time.sleep(0.5)
