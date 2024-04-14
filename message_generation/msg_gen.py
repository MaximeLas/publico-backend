from asyncio import Queue
import time

from devtools import debug

from langchain_community.callbacks import get_openai_callback
from langchain.chains.openai_functions import create_openai_fn_runnable
from langchain_openai import ChatOpenAI

from workflow.session_state import ImplicitQuestion, SessionState
from utilities.llm_streaming_utils import stream_from_llm_generation
from utilities.openai_functions_utils import function_for_comprehensiveness_check
from utilities.document_helpers import (
    add_files_to_vector_store,
    get_most_relevant_docs_in_vector_store_for_answering_question,
)
from configurations.constants import IS_DEV_MODE
from configurations.prompts import (
    get_prompt_template_for_generating_original_answer,
    get_prompt_template_for_comprehensiveness_check_openai_functions,
    get_prompt_template_for_generating_answer_to_implicit_question,
    get_prompt_template_for_generating_final_answer,
    get_prompt_template_for_user_guidance_post_answer
)
import logging

def generate_validation_message_following_files_upload(state: SessionState, queue: Queue) -> list[str]:
    '''Generate a validation message following a file upload.'''

    files = state.uploaded_files
    file_or_files = 'file' if len(files) == 1 else 'files'

    queue.put_nowait(
        f'You successfully uploaded **{len(files)}** {file_or_files}! 🎉\n\n' +
        'Now, on to your first grant application question!')

    add_files_to_vector_store(
        session_id=str(state.session_id),
        files=files,
        tokens_per_doc_chunk=state.get_num_of_tokens_per_doc_chunk())


def generate_answer_to_question_stream(state: SessionState, queue: Queue) -> None:
    '''Generate and stream an answer to a grant application question by streaming tokens from the LLM.'''

    question_state = state.get_last_question_context()
    if question_state.question is None:
        queue.put_nowait('No answer generated due to missing application question.')
        return

    most_relevant_documents = get_most_relevant_docs_in_vector_store_for_answering_question(
        session_id=str(state.session_id),
        question=question_state.question,
        n_results=state.get_num_of_doc_chunks_to_consider())

    intro_to_answer = 'Based on the information you provided, here\'s the best answer I could put together:\n\n'
    queue.put_nowait(intro_to_answer)

    def on_llm_end(answer: str, answer_formatted: str):
        state.set_answer_to_current_grant_application_question(answer, answer_formatted)
        time.sleep(0.15)
        queue.put_nowait(f'\n\nGenerated answer contains **{len(answer.split())}** words.\n\n')

    stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_original_answer(state.get_system_prompt_for_original_question()),
        queue=queue,
        on_llm_end=on_llm_end,
        chain_type='qa_chain',
        docs=most_relevant_documents,
        question=question_state.question,
        word_limit=question_state.word_limit
    )

def check_for_comprehensiveness(state: SessionState, queue: Queue) -> None:
    '''Check for comprehensiveness of an answer to a grant application question using OpenAI functions.'''

    queue.put_nowait('Give me a moment while I think about how to improve it ... 🔍\n\n')

    question_state = state.get_last_question_context()
    comprehensiveness_state = question_state.comprehensiveness

    with get_openai_callback() as cb:
        prompt = get_prompt_template_for_comprehensiveness_check_openai_functions()
        chat_openai = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0)
        chain = create_openai_fn_runnable([function_for_comprehensiveness_check], chat_openai, prompt)

        response = chain.invoke(
            dict(question=question_state.question, answer=question_state.answer.raw)
        )

        debug(**{'Summary info OpenAI callback': cb})

    comprehensiveness_state.missing_information = response['missing_information']

    questions = response['implicit_questions']
    if type(questions) is dict:
        comprehensiveness_state.implicit_questions = {
            i+1: ImplicitQuestion(q) for i, q in enumerate(questions.values())}
    elif type(questions) is list:
        comprehensiveness_state.implicit_questions = {
            i+1: ImplicitQuestion(q if type(q) is str else q['question']) for i, q in enumerate(questions)}
    else:
        raise ValueError(f'Unexpected type for implicit questions: {type(questions)}\n')

    #debug(**{f'Implicit question #{i}': q.question for i, q in comprehensiveness_state.implicit_questions.items()})

    queue.put_nowait(f'*{comprehensiveness_state.missing_information}*')

    time.sleep(0.15)
    queue.put_nowait('\n\nTo make the answer as strong as possible, I\'d include answers to the following questions:')

    time.sleep(0.15)
    for i, q in comprehensiveness_state.implicit_questions.items():
        time.sleep(0.1)
        queue.put_nowait(f'\n\n(**{i}**) **{q.question}**')


def generate_answer_for_implicit_question_stream(state: SessionState, queue: Queue) -> None:
    '''Generate and stream answers for implicit questions to be answered to make the answer comprehensive.'''

    start_of_chatbot_message = 'Here\'s what I found in your documents to answer this question:'
    queue.put_nowait(start_of_chatbot_message + '\n\n')

    if IS_DEV_MODE and state.user_has_changed_num_of_tokens():
        add_files_to_vector_store(
            session_id=str(state.session_id),
            files=state.uploaded_files,
            tokens_per_doc_chunk=state.get_num_of_tokens_per_doc_chunk())

    most_relevant_documents = get_most_relevant_docs_in_vector_store_for_answering_question(
        session_id=str(state.session_id),
        question=state.get_current_implicit_question(),
        n_results=state.get_num_of_doc_chunks_to_consider())

    def on_llm_end(answer: str, answer_formatted: str):
        if 'Not enough information' not in answer:
            state.set_answer_to_current_implicit_question(answer, answer_formatted)

    stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_answer_to_implicit_question(
            state.get_system_prompt_for_implicit_question()),
        queue=queue,
        on_llm_end=on_llm_end,
        chain_type='qa_chain',
        model='gpt-3.5-turbo',
        verbose=False,
        docs=most_relevant_documents,
        question=state.get_current_implicit_question()
    )



def generate_final_answer_stream(state: SessionState, queue: Queue) -> None:
    '''Generate and stream a final answer to a grant application question.'''

    question_context = state.get_last_question_context()
    comprehensiveness_state = question_context.comprehensiveness

    chatbot_msg = ''
    if len(comprehensiveness_state.implicit_questions) == 0:
        chatbot_msg = f'No final answer generated due to having no implicit questions.'
    elif not state.exists_answer_to_any_implicit_question():
        chatbot_msg = f'No final answer generated due to having answered none of the implicit questions.'

    if chatbot_msg != '':
        debug(chatbot_msg)
        queue.put_nowait('Something went wrong, please try again.')
        return

    implicit_questions_answered = dict(filter(lambda elem: elem[1].answer is not None, comprehensiveness_state.implicit_questions.items()))
    num_implicit_questions = len(implicit_questions_answered)
    s = 's' if num_implicit_questions > 1 else ''

    intro_to_final_answer = (
        f'Here is the final answer to **"{question_context.question}"** '
        f'after integrating answer{s} to **{num_implicit_questions}** implicit question{s}:\n\n')

    queue.put_nowait(intro_to_final_answer)

    def on_llm_end(answer: str, answer_formatted: str):
        state.set_revised_answer_to_current_grant_application_question(answer, answer_formatted)
        queue.put_nowait(f'\n\nThe final answer contains **{len(answer.split())}** words. The word limit is **{question_context.word_limit}** words.\n\n')

        time.sleep(0.05)
        '''comparison_msg = (
            f'For reference, the original answer, which contains **{len(question_context.answer.raw.split())}** words, is the following:\n\n' +
            f'{question_context.answer.formatted}')

        queue.put_nowait(comparison_msg)'''

    stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_final_answer(),
        queue=queue,
        on_llm_end=on_llm_end,
        chain_type='llm_chain',
        verbose=True,
        question=question_context.question,
        answers_to_implicit_questions='\n\n'.join([f'{q.question}\n{q.answer.raw}' for q in implicit_questions_answered.values()]),
        word_limit=question_context.word_limit,
        original_answer=question_context.answer.raw
    )


def generate_improved_answer_following_user_guidance_prompt(state: SessionState, queue: Queue) -> None:
    '''Generate and stream an improved answer to a grant application question following user guidance.'''

    question_context = state.get_last_question_context()
    comprehensiveness_state = question_context.comprehensiveness
    implicit_questions_answered = dict(filter(lambda elem: elem[1].answer is not None, question_context.comprehensiveness.implicit_questions.items()))

    intro_to_improved_answer = f'Here is the improved answer to **"{question_context.question}"**:\n\n'
    queue.put_nowait(intro_to_improved_answer)

    def on_llm_end(answer: str, answer_formatted: str):
        state.set_improved_answer(answer, answer_formatted)
        queue.put_nowait(f'\n\nThe improved answer contains **{len(answer.split())}** words. The word limit is **{question_context.word_limit}** words.')

    stream_from_llm_generation(
        prompt=get_prompt_template_for_user_guidance_post_answer(state.get_current_improvements()),
        queue=queue,
        on_llm_end=on_llm_end,
        chain_type='llm_chain',
        verbose=True,
        question=question_context.question,
        answers_to_implicit_questions='\n\n'.join([f'{q.question}\n{q.answer.raw}' for q in implicit_questions_answered.values()]),
        word_limit=question_context.word_limit,
        original_answer=question_context.answer.raw,
        answer=(
            comprehensiveness_state.revised_application_answer.raw
                if comprehensiveness_state.revised_application_answer
                else
            question_context.answer.raw)
    )
