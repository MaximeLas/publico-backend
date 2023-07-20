from pydantic import BaseModel, Field

from langchain.callbacks import get_openai_callback
from langchain.chains.openai_functions import create_openai_fn_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from constants import OutputKeys
from helpers import *
from prompts import *


def generate_answer_to_question(vars: dict) -> str:
    '''Generate an answer to a grant application question.'''

    documents_chunks = get_documents_chunks_for_files(vars[OutputKeys.PRIOR_GRANT_APPLICATIONS])
    vector_store = Chroma.from_documents(documents=documents_chunks, embedding=OpenAIEmbeddings(client=None))

    if vars[OutputKeys.APPLICATION_QUESTION] == '':
        vars[OutputKeys.APPLICATION_ANSWER] = ''
        print(f'No application question provided, skipping generation of answer.\n')
        return 'No answer generated due to missing application question.'

    question_for_prompt = vars[OutputKeys.APPLICATION_QUESTION]
    most_relevant_documents = get_most_relevant_docs_in_vector_store_for_answering_question(
        vector_store=vector_store, question=question_for_prompt, n_results=1)

    vars[OutputKeys.MOST_RELEVANT_DOCUMENTS] = most_relevant_documents

    if vars[OutputKeys.WORD_LIMIT] and vars[OutputKeys.WORD_LIMIT].isdigit():
        question_for_prompt += f' ({vars[OutputKeys.WORD_LIMIT]} words)'

    chat_openai = ChatOpenAI(client=None, model_name='gpt-4', temperature=0)

    vars[OutputKeys.APPLICATION_ANSWER] = generate_answers_from_documents_for_question(
        most_relevant_documents, chat_openai, question_for_prompt)[0]

    print(f'Final answer for application question "{vars[OutputKeys.APPLICATION_QUESTION]}":\n\n{vars[OutputKeys.APPLICATION_ANSWER]}\n')

    return vars[OutputKeys.APPLICATION_ANSWER]


class ImplicitQuestion(BaseModel):
    '''An implicit question to be answered to fill in any missing information required to make the answer comprehensive.'''

    question: str = Field(
        ...,
        description="The question to be answered to fill in missing information."
    )

def check_for_comprehensiveness_fn(missing_information: str, implicit_questions: list[ImplicitQuestion]):
    '''
    Check for comprehensiveness of an answer to a grant application question.

    Args:
        missing_information: A description of the information that should be in a good grant application answer to the given question but is missing from this answer
        implicit_questions: List of questions to be answered to fill in any missing information required to make the answer comprehensive.
    '''

    print(f"Missing information: {missing_information}")
    for i, question in enumerate(implicit_questions):
        print(f"Question {i + 1}: {question.question}")


def check_for_comprehensiveness(vars: dict) -> str | None:
    '''Check for comprehensiveness of an answer to a grant application question using OpenAI functions.'''

    if vars[OutputKeys.CHECK_COMPREHENSIVENESS] != 'Yes':
        return None

    chat_openai = ChatOpenAI(client=None, model_name='gpt-4', temperature=0)
    prompt = get_prompt_template_for_comprehensiveness_check_openai_functions()

    with get_openai_callback() as cb:
        chain = create_openai_fn_chain([check_for_comprehensiveness_fn], chat_openai, prompt, verbose=True)
        response = chain.run(question=vars[OutputKeys.APPLICATION_QUESTION], answer=vars[OutputKeys.APPLICATION_ANSWER])
        print(f'Summary info OpenAI callback:\n{cb}\n')

    print(f'Response for check of comprehensiveness:\n\n{response}\n')

    assert type(response) == dict
    vars[OutputKeys.MISSING_INFORMATION] = response['missing_information']
    vars[OutputKeys.IMPLICIT_QUESTIONS] = response['implicit_questions']

    print(vars[OutputKeys.MISSING_INFORMATION])
    for i, q in enumerate(vars[OutputKeys.IMPLICIT_QUESTIONS]):
        print(f"Question {i + 1}: {q}")

    return vars[OutputKeys.MISSING_INFORMATION]


def generate_answers_for_implicit_questions(vars: dict) -> list[str]:
    '''Generate answers for implicit questions to be answered to fill in any missing information required to make the answer comprehensive.'''

    vars[OutputKeys.ANSWERS_TO_IMPLICIT_QUESTIONS] = []
    answers = []
    for i, question in enumerate(vars[OutputKeys.IMPLICIT_QUESTIONS]):
        with get_openai_callback() as cb:
            chain = load_qa_chain(
                llm=ChatOpenAI(client=None, model_name='gpt-4', temperature=0),
                chain_type='stuff',
                verbose=False,
                prompt=get_prompt_template_for_generating_answer_to_implicit_question()
            )
            answer = chain.run(input_documents=vars[OutputKeys.MOST_RELEVANT_DOCUMENTS], question=question)

            if 'Not enough information' not in answer:
                vars[OutputKeys.ANSWERS_TO_IMPLICIT_QUESTIONS].append(answer)
                print(f'Generated answer for "{question}":\n\n{answer}\n')
            else:
                print(f'Skipping including this answer as not enough information provided to answer question "{question}"\n')

            print(f'Summary info OpenAI callback:\n{cb}\n\n')
            answers.append(f'({i+1}) {question}\n\n{answer}')

    return answers


def generate_final_answer(vars: dict) -> list[str]:
    '''Generate a final answer to a grant application question.'''

    with get_openai_callback() as cb:
        chain = LLMChain(
            llm=ChatOpenAI(client=None, model_name='gpt-4', temperature=0),
            prompt=get_prompt_template_for_generating_final_answer(),
            verbose=True
        )
        response = chain.run(
            question=vars[OutputKeys.APPLICATION_QUESTION],
            answers_to_implicit_questions='\n\n'.join(vars[OutputKeys.ANSWERS_TO_IMPLICIT_QUESTIONS]),
            word_limit=vars[OutputKeys.WORD_LIMIT],
            original_answer=vars[OutputKeys.APPLICATION_ANSWER]
        )
        print(f'Summary info OpenAI callback:\n{cb}\n')

    print(f'Final answer for application question "{vars[OutputKeys.APPLICATION_QUESTION]}":\n\n{response}\n')

    return [f'Here is the final answer to "{vars[OutputKeys.APPLICATION_QUESTION]}":\n\n{response}',
            f'For reference, here is the original answer:\n\n{vars[OutputKeys.APPLICATION_ANSWER]}']
