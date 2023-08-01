import copy
import time

from devtools import debug

from langchain.callbacks import get_openai_callback
from langchain.chains.openai_functions import create_openai_fn_chain, create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from chatbot_step import MessageOutputType
from context import ImplicitQuestion, UserContext
from helpers import get_most_relevant_docs_in_vector_store_for_answering_question, get_documents_chunks_for_files
from llm_streaming_utils import stream_from_llm_generation
from settings import GPT_MODEL
from openai_functions_utils import function_for_comprehensiveness_check, get_json_schema_for_comprehensiveness_check
from prompts import (
    get_prompt_template_for_generating_original_answer,
    get_prompt_template_for_comprehensiveness_check_openai_functions,
    get_prompt_template_for_generating_answer_to_implicit_question,
    get_prompt_template_for_generating_final_answer
)



def generate_answer_to_question_stream(context: UserContext) -> MessageOutputType:
    '''Generate and stream an answer to a grant application question by streaming tokens from the LLM.'''

    documents_chunks = get_documents_chunks_for_files(context.prior_grant_applications)
    vector_store = Chroma.from_documents(documents=documents_chunks, embedding=OpenAIEmbeddings(client=None))

    question_context = context.questions[-1]
    if question_context.question is None:
        message = 'No answer generated due to missing application question.'
        debug(message)
        yield message
        return

    question_context.most_relevant_documents = get_most_relevant_docs_in_vector_store_for_answering_question(
        vector_store=vector_store, question=question_context.question, n_results=1)

    final_answer = ''
    for _, response in stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_original_answer(),
        chain_type='qa_chain',
        verbose=False,
        docs=question_context.most_relevant_documents,
        question=question_context.question,
        word_limit=question_context.word_limit
    ):
        final_answer = f'*{response}*'
        yield final_answer

    answer_word_count = len(final_answer.split())
    question_context.answer = final_answer

    time.sleep(0.5)
    yield [final_answer, f'Generated answer contains **{answer_word_count}** words.']
    time.sleep(0.5)


def check_for_comprehensiveness(context: UserContext, use_json_schema: bool = False) -> MessageOutputType:
    '''Check for comprehensiveness of an answer to a grant application question using OpenAI functions.'''

    question_context = context.questions[-1]
    comprehensiveness_context = question_context.comprehensiveness

    if not comprehensiveness_context.do_check:
        yield 'Skipping check for comprehensiveness.'
        return

    chat_openai = ChatOpenAI(client=None, model=GPT_MODEL, temperature=0)
    prompt = get_prompt_template_for_comprehensiveness_check_openai_functions()

    with get_openai_callback() as cb:
        chain = (
            create_structured_output_chain(get_json_schema_for_comprehensiveness_check(), chat_openai, prompt, verbose=True)
                if use_json_schema
                else
            create_openai_fn_chain([function_for_comprehensiveness_check], chat_openai, prompt, verbose=True)
        )
        
        response = chain(inputs=dict(question=question_context.question, answer=question_context.answer))['function']
        debug(**{'Summary info OpenAI callback': cb})


    comprehensiveness_context.missing_information = response['missing_information']

    questions = response['implicit_questions']
    if type(questions) is dict:
        comprehensiveness_context.implicit_questions = {
            i+1: ImplicitQuestion(q) for i, q in enumerate(questions.values())}
    elif type(questions) is list:
        comprehensiveness_context.implicit_questions = {
            i+1: ImplicitQuestion(q if type(q) is str else q['question']) for i, q in enumerate(questions)}
    else:
        raise ValueError(f'Unexpected type for implicit questions: {type(questions)}\n')

    debug(**{f'Implicit question #{i}': q.question for i, q in comprehensiveness_context.implicit_questions.items()})

    answers = [comprehensiveness_context.missing_information]
    yield answers
    time.sleep(1)

    answers.append('')
    for i, q in comprehensiveness_context.implicit_questions.items():
        answers[-1] += (f'(**{i}**) **{q.question}**\n')
        time.sleep(0.5)
        yield answers



def generate_answer_for_implicit_question_stream(context: UserContext) -> MessageOutputType:
    '''Generate and stream answers for implicit questions to be answered to make the answer comprehensive.'''

    time.sleep(1)

    start_of_chatbot_message = "Here's what I found to answer the question."
    time.sleep(0.5)
    yield start_of_chatbot_message

    answer = ''

    for _, response in stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_answer_to_implicit_question(),
        chain_type='qa_chain',
        model='gpt-3.5-turbo',
        verbose=False,
        docs=context.questions[-1].most_relevant_documents,
        question=context.get_current_implicit_question_to_be_answered()
    ):
        answer = response
        yield f'{start_of_chatbot_message}\n\n*{response}*'

    end_of_chatbot_message: str
    if 'Not enough information' not in answer:
        context.set_answer_to_current_implicit_question(answer)
        end_of_chatbot_message = 'Is this helpful?'
    else:
        end_of_chatbot_message = 'Would you like to answer it yourself?'

    yield f'{start_of_chatbot_message}\n\n*{answer}*\n\n{end_of_chatbot_message}'


def generate_answers_for_implicit_questions_stream(context: UserContext) -> MessageOutputType:
    '''Generate and stream answers for implicit questions to be answered to make the answer comprehensive.'''

    time.sleep(1)
    answers: list[str] = []

    question_context = context.questions[-1]
    for i, q in question_context.comprehensiveness.implicit_questions.items():
        start_of_sentence_for_answer = f'(**{i}**) **{q.question}**'
        answers.append(start_of_sentence_for_answer)
        time.sleep(0.5)
        yield answers

        for _, response in stream_from_llm_generation(
            prompt=get_prompt_template_for_generating_answer_to_implicit_question(),
            chain_type='qa_chain',
            model='gpt-3.5-turbo',
            verbose=False,
            docs=question_context.most_relevant_documents,
            question=q.question
        ):
            answers[-1] = f'{start_of_sentence_for_answer}\n\n*{response}*'
            yield answers

        if 'Not enough information' not in answers[-1]:
            q.answer = answers[-1]


def generate_final_answer_stream(context: UserContext) -> MessageOutputType:
    '''Generate and stream a final answer to a grant application question.'''

    question_context = context.questions[-1]
    comprehensiveness_context = question_context.comprehensiveness

    message = ''
    if not comprehensiveness_context.do_check:
        message = f'No final answer generated due to not having checked comprehensiveness.'
    elif len(comprehensiveness_context.implicit_questions) == 0:
        message = f'No final answer generated due to having no implicit questions.'
    elif sum(q.answer is not None for q in comprehensiveness_context.implicit_questions.values()) == 0:
        message = f'No final answer generated due to having answered none of the implicit questions.'

    if message != '':
        debug(message)
        return

    implicit_questions_answered = dict(filter(lambda elem: elem[1].answer is not None, comprehensiveness_context.implicit_questions.items()))
    num_implicit_questions = len(implicit_questions_answered)
    s = 's' if num_implicit_questions > 1 else ''

    start_of_sentence_for_final_answer = (
        f'Here is the final answer to **"{question_context.question}"** '
        f'after integrating answer{s} to **{num_implicit_questions}** implicit question{s}')

    final_answer = start_of_sentence_for_final_answer
    final_llm_response = ''

    for _, llm_response in stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_final_answer(),
        chain_type='llm_chain',
        verbose=True,
        question=question_context.question,
        answers_to_implicit_questions='\n\n'.join([q.answer for q in implicit_questions_answered.values()]),
        word_limit=question_context.word_limit,
        original_answer=question_context.answer
    ):
        final_llm_response = llm_response
        final_answer = f'{start_of_sentence_for_final_answer}:\n\n*{llm_response}*'
        yield final_answer

    comprehensiveness_context.revised_application_answer = final_llm_response

    comparison_answer = (
        f'The word limit requested was **{question_context.word_limit}** words.\n'
        f'The new answer contains **{len(final_llm_response.split())}** words.\n'
        f'For reference, the original answer, which contains **{len(question_context.answer.split())}** words, ' # type: ignore
        f'is the following:\n\n{question_context.answer}')

    time.sleep(0.5)
    yield [final_answer, comparison_answer]
    time.sleep(0.5)
