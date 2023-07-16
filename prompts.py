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
