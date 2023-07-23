from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.schema.messages import (
    HumanMessage,
    SystemMessage
)


def get_prompt_template_for_generating_original_answer() -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to answer grant application question from documents
        Returns:
            ChatPromptTemplate: prompt template for chat model to answer question
    '''
    
    system_template = (
        'You are going to help a nonprofit organization that is applying for a grant.\n'
        'Use the following pieces of context to respond to a grant application question '
        'in a way that provides a compelling and comprehensive answer from the perspective '
        'of a nonprofit organization applying for grant funding.\n'
        '----------------\n'
        '{context}'
    )

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('{question}'),
    ]

    return ChatPromptTemplate.from_messages(messages)


def get_prompt_template_for_comprehensiveness_check() -> PromptTemplate:
    '''
    Get a prompt template for a model to check the comprehensiveness of a grant application answer
        Returns:
            PromptTemplate: prompt template for model to check comprehensiveness of answer
    '''

    prompt_template = (
        'I\'m going to present you with two pieces of information: a question on a grant application, '
        'and an answer that was written by a nonprofit organization to that question. '
        'Your job is to make the answer as comprehensive as possible. '
        'Once you see the answer written by the nonprofit, you will (a) identify any information that '
        'should be in a good grant application answer to the given question but is missing from this answer, '
        'and (b) come up with a list of questions (no more than five) for the author of the application that '
        'would elicit that missing information.\n'
        '----------------\n'
        'Grant application question: {question}\n'
        '----------------\n'
        'Grant application answer: {answer}\n'
    )

    return PromptTemplate.from_template(prompt_template)


def get_prompt_template_for_comprehensiveness_check_openai_functions() -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to check the comprehensiveness of a grant application answer using OpenAI functions
        Returns:
            ChatPromptTemplate: prompt template for chat model to check comprehensiveness of answer
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


def get_prompt_template_for_generating_answer_to_implicit_question() -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to answer an implicit question
        Returns:
            ChatPromptTemplate: prompt template for chat model to answer implicit question
    '''
    system_template = (
        'You are a grantwriting expert who will be helping a non-profit organization applying for a grant. '
        'Please provide your best answer to the following question using the context provided. '
        'Be as concise as possible, using at most one or two lines. '
        'If you can\'t answer the question, don\'t make something up and simply answer the words \'Not enough information provided.\'.\n'
        '----------------\n'
        '{context}'
    )

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('{question}'),
    ]

    return ChatPromptTemplate.from_messages(messages)



def get_prompt_template_for_generating_final_answer() -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to generate a final answer to a grant application question
        Returns:
            ChatPromptTemplate: prompt template for chat model to generate final answer
    '''

    system_template = (
        'The following paragraph is an answer to a grant application question submitted by a nonprofit. '
        'The grant application questions was: "{question} ({word_limit} words)". '
        'Please improve the answer by incorporating the potentially valuable information contained in the following lines:\n'
        '----------------\n'
        '{answers_to_implicit_questions}.\n'
    )

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('Original answer ->\n{original_answer}'),
    ]

    return ChatPromptTemplate.from_messages(messages)
