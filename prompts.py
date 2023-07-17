from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)


def get_prompt_template_for_question() -> ChatPromptTemplate:
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
