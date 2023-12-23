import time

from devtools import debug

from langchain.callbacks import get_openai_callback
from langchain.chains.openai_functions import create_openai_fn_chain, create_structured_output_chain
from langchain.chat_models import ChatOpenAI

from chatbot_step import MessageOutputType
from constants import IS_DEV_MODE, GPT_MODEL
from context import ImplicitQuestion, UserContext
from helpers import (
    get_most_relevant_docs_in_vector_store_for_answering_question,
    get_vector_store_for_files
)
from llm_streaming_utils import stream_from_llm_generation
from openai_functions_utils import function_for_comprehensiveness_check
from prompts import (
    get_prompt_template_for_generating_original_answer,
    get_prompt_template_for_comprehensiveness_check_openai_functions,
    get_prompt_template_for_generating_answer_to_implicit_question,
    get_prompt_template_for_generating_final_answer
)



def generate_answer_to_question_stream(context: UserContext) -> MessageOutputType:
    '''Generate and stream an answer to a grant application question by streaming tokens from the LLM.'''

    # get the vector store for the uploaded files (will only update if files are not the same as before)
    context.uploaded_files.vector_store = get_vector_store_for_files(
        files=context.uploaded_files.files,
        tokens_per_doc_chunk=context.get_num_of_tokens_per_doc_chunk())

    question_context = context.get_last_question_context()
    if question_context.question is None:
        yield 'No answer generated due to missing application question.'
        return

    question_context.most_relevant_documents = get_most_relevant_docs_in_vector_store_for_answering_question(
        vector_store=context.uploaded_files.vector_store,
        question=question_context.question,
        n_results=context.get_num_of_doc_chunks_to_consider())

    intro_to_answer = 'Based on the information you provided, here\'s the best answer I could put together:'
    yield intro_to_answer

    chatbot_msg = intro_to_answer
    for _, llm_response in stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_original_answer(
            context.get_system_prompt_for_original_question()),
        chain_type='qa_chain',
        verbose=False,
        docs=question_context.most_relevant_documents,
        question=question_context.question,
        word_limit=question_context.word_limit
    ):
        answer_to_question = llm_response
        chatbot_msg = f'{intro_to_answer}\n\n*{answer_to_question}*'
        yield chatbot_msg

    question_context.answer = answer_to_question

    time.sleep(0.25)
    chatbot_msg += f'\n\nGenerated answer contains **{len(answer_to_question.split())}** words.'
    yield chatbot_msg




def check_for_comprehensiveness(context: UserContext) -> MessageOutputType:
    '''Check for comprehensiveness of an answer to a grant application question using OpenAI functions.'''

    chatbot_msg = 'Give me a moment while I think about how to improve it ... 🔍'
    yield chatbot_msg

    question_context = context.get_last_question_context()
    comprehensiveness_context = question_context.comprehensiveness

    with get_openai_callback() as cb:
        prompt = get_prompt_template_for_comprehensiveness_check_openai_functions()
        chat_openai = ChatOpenAI(client=None, model=GPT_MODEL, temperature=0)
        chain = create_openai_fn_chain([function_for_comprehensiveness_check], chat_openai, prompt, verbose=True)
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

    chatbot_msg = [chatbot_msg, f'*{comprehensiveness_context.missing_information}*']
    yield chatbot_msg

    time.sleep(0.25)
    chatbot_msg[1] += '\n\nTo make the answer as strong as possible, I\'d include answers to the following questions:\n'
    yield chatbot_msg

    time.sleep(0.25)
    implicit_questions = ''
    for i, q in comprehensiveness_context.implicit_questions.items():
        time.sleep(0.25)
        implicit_questions += f'\n(**{i}**) **{q.question}**'
        yield [chatbot_msg[0], chatbot_msg[1] + implicit_questions]



def generate_answer_for_implicit_question_stream(context: UserContext) -> MessageOutputType:
    '''Generate and stream answers for implicit questions to be answered to make the answer comprehensive.'''

    start_of_chatbot_message = 'Here\'s what I found in your documents to answer this question:'
    yield start_of_chatbot_message

    if IS_DEV_MODE and context.user_has_changed_num_of_tokens():
        context.uploaded_files.vector_store = get_vector_store_for_files(
            files=context.uploaded_files.files,
            tokens_per_doc_chunk=context.get_num_of_tokens_per_doc_chunk())

    most_relevant_documents = get_most_relevant_docs_in_vector_store_for_answering_question(
        vector_store=context.uploaded_files.vector_store, question=context.get_current_implicit_question(), n_results=4)

    answer_to_question = ''
    for _, llm_response in stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_answer_to_implicit_question(
            context.get_system_prompt_for_implicit_question()),
        chain_type='qa_chain',
        model='gpt-3.5-turbo',
        verbose=False,
        docs=most_relevant_documents,
        question=context.get_current_implicit_question()
    ):
        answer_to_question = llm_response
        yield f'{start_of_chatbot_message}\n\n*{answer_to_question}*'

    if 'Not enough information' not in answer_to_question:
        context.set_answer_to_current_implicit_question(answer_to_question)



def generate_final_answer_stream(context: UserContext) -> MessageOutputType:
    '''Generate and stream a final answer to a grant application question.'''

    question_context = context.get_last_question_context()
    comprehensiveness_context = question_context.comprehensiveness

    chatbot_msg = ''
    if len(comprehensiveness_context.implicit_questions) == 0:
        chatbot_msg = f'No final answer generated due to having no implicit questions.'
    elif not context.exists_answer_to_any_implicit_question():
        chatbot_msg = f'No final answer generated due to having answered none of the implicit questions.'

    if chatbot_msg != '':
        debug(chatbot_msg)
        yield 'Something went wrong, please try again.'
        return

    implicit_questions_answered = dict(filter(lambda elem: elem[1].answer is not None, comprehensiveness_context.implicit_questions.items()))
    num_implicit_questions = len(implicit_questions_answered)
    s = 's' if num_implicit_questions > 1 else ''

    intro_to_final_answer = (
        f'Here is the final answer to **"{question_context.question}"** '
        f'after integrating answer{s} to **{num_implicit_questions}** implicit question{s}:')

    yield intro_to_final_answer

    final_answer = ''
    for _, llm_response in stream_from_llm_generation(
        prompt=get_prompt_template_for_generating_final_answer(),
        chain_type='llm_chain',
        verbose=True,
        question=question_context.question,
        answers_to_implicit_questions='\n'.join([q.answer for q in implicit_questions_answered.values()]),
        word_limit=question_context.word_limit,
        original_answer=question_context.answer
    ):
        final_answer = llm_response
        chatbot_msg = f'{intro_to_final_answer}\n\n*{final_answer}*'
        yield chatbot_msg

    comprehensiveness_context.revised_application_answer = final_answer

    chatbot_msg += f'\n\nThe final answer contains **{len(final_answer.split())}** words. The word limit is **{question_context.word_limit}** words.'
    yield chatbot_msg

    time.sleep(0.25)
    comparison_msg = (
        f'For reference, the original answer, which contains **{len(question_context.answer.split())}** words, is the following:\n\n' +
        f'*{question_context.answer}*')

    yield [chatbot_msg, comparison_msg]
