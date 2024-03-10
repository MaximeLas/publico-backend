from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.schema.messages import (
    HumanMessage,
    SystemMessage
)

from workflow.session_state import Improvement



def get_prompt_template_for_generating_original_answer(system_prompt: str) -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to answer grant application question from documents
        Parameters:
            system_prompt: the system prompt to use for the chat model
        Returns:
            a prompt template for a chat model to answer a grant application question
    '''

    messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template('{question} ({word_limit} words)'),
    ]

    return ChatPromptTemplate.from_messages(messages)


def get_prompt_template_for_comprehensiveness_check_openai_functions() -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to check the comprehensiveness of a grant application answer using OpenAI functions
        Returns:
            a prompt template for a chat model to check the comprehensiveness of an answer
    '''

    sys_msg = (
        'I\'m going to present you with two pieces of information: a question on a grant application, '
        'and an answer that was written by a nonprofit organization to that question. '
        'Your job is to make the answer as comprehensive as possible. '
        'Once you see the answer written by the nonprofit, you will (a) identify any information that '
        'should be in a good grant application answer to the given question but is missing from this answer, '
        'and (b) come up with a list of questions (no more than five) for the author of the application that '
        'would elicit that missing information.'
    )
    prompt_msgs = [
        SystemMessage(content=sys_msg),
        HumanMessage(
            content="Make calls to the relevant function to record the missing information and implicit questions in the following input:"
        ),
        HumanMessagePromptTemplate.from_template(
            'Grant application question: {question}\n'
            '----------------\n'
            'Grant application answer: {answer}\n'
        ),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]

    return ChatPromptTemplate(messages=prompt_msgs, input_variables=["question", "answer"])


def get_prompt_template_for_generating_answer_to_implicit_question(system_prompt: str) -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to answer an implicit question
        Parameters:
            system_prompt: the system prompt to use for the chat model
        Returns:
            a prompt template for a chat model to answer an implicit question from documents
    '''

    messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template('{question}'),
    ]

    return ChatPromptTemplate.from_messages(messages)


def get_prompt_template_for_generating_final_answer() -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to generate a final answer to a grant application question
        Returns:
            a prompt template for a chat model to generate a final answer
    '''

    system_template = (
        'You will be given a paragraph which is an answer to a grant application question submitted by a nonprofit.\n'
        'The grant application questions was: "{question} ({word_limit} words)".\n'
        'Improve the answer by incorporating the potentially valuable information contained in the following lines:\n'
        '----------------\n'
        '{answers_to_implicit_questions}.\n'
        '----------------\n'
        'Start your message right away with the new answer without any leading words. '
        'Make sure to comply with the word limit stated in parentheses at the end of the grant application question as this is crucial! (but do not write the word count itself in the generated answer)'
    )

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('{original_answer}'),
    ]

    return ChatPromptTemplate.from_messages(messages)


def get_prompt_template_for_user_guidance_post_answer(improvements: list[Improvement]) -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to modify an answer according to user guidance
        Returns:
            a prompt template for a chat model to modify an answer according to user guidance
    '''

    messages = get_prompt_template_for_generating_final_answer().messages
    messages.append(AIMessagePromptTemplate.from_template('{answer}'))

    for improvement in improvements[:-1]:
        messages.append(HumanMessagePromptTemplate.from_template(improvement.user_prompt))
        messages.append(AIMessagePromptTemplate.from_template(improvement.improved_answer.original))

    messages.append(HumanMessagePromptTemplate.from_template(improvements[-1].user_prompt +
        ' (Just make sure to, once again, comply with the word limit of {word_limit} words as this is crucial!)'))

    return ChatPromptTemplate.from_messages(messages)
