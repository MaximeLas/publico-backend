from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.schema.messages import (
    HumanMessage,
    SystemMessage
)

# single comment
# another comment
'''
multiple lines
of comments
'''

message_to_user_when_all_iqs_first_presented = (
    'Here\'s a list of questions that could improve the generated answer.\n'
    '{implicit_questions}\n'
    'Let\'s go through them one by one.'
)

message_to_user_when_presenting_first_iq_for_interaction = (
    'Let\'s start with the first question:\n'
    '{implicit_question_1}\n'
    'Do you want to answer this question in the revised answer?'
)

# User is presented with option "yes, let's answer it!" and "no, it's not relevant/I will answer it later in the application"

# If user selects no, move to next question

# If yes, then user is presented with three additional options

message_to_user_when_answering_first_relevant_iq = (
    'Here\'s what I found to answer the question.\n'
    '{model_answer_to_implicit_question_1}\n'
    'Is this helpful?\n'
)

# User selects "yes, that's good as is" or "yes, but I'd like to edit it" or "no, let me write one myself"
# If good as is, save answer to {approved_answers_to_implicit_questions}; otherwise:

message_to_user_to_revise_implicit_question = (
    'Okay, let\'s edit the proposed answer.\n'
    'You can edit the text in the box below, or type over it to replace it \n'
    'with any information you think better answers the question.'
)

# Once user has submitted edited/revised answer, save to {approved_answers_to_implicit_questions}; then, move on:

message_to_user_to_move_to_next_implicit_question = (
    'Great! Now that we\'ve answered that question, \n'
    'let\'s move on to the next.'
)
    
# Final prompt to model to incorporate all user-approved answers
def get_prompt_template_for_generating_final_answer() -> ChatPromptTemplate:
    '''
    Get a prompt template for a chat model to generate a final answer to a grant application question
        Returns:
            ChatPromptTemplate: prompt template for chat model to generate final answer
    '''

    system_template = (
        'You will be given a paragraph which is an answer to a grant application question submitted by a nonprofit.\n'
        'The grant application questions was: "{question} ({word_limit} words)".\n'
        'Improve the answer by incorporating the potentially valuable information contained in the following lines:\n'
        '----------------\n'
        '{approved_answers_to_implicit_questions}.\n'
        '----------------\n'
        'Start your message right away with the new answer without any leading words. '
        'Please make sure to comply with the word limit stated at the end of the grant application question.'
    )

example_prompt_to_user_for_an_implicit_question = (
    'This is the start of the prompt '
    'and it continues on and on and on.\n'
    'Here is the question:\n'
    '{implicit_question}\n'
    'Alright bye!'
)

# define below prompt in the future for step 3
prompt_to_use_for_anu_additional_information_in_the_end: str


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
        HumanMessagePromptTemplate.from_template('{question} ({word_limit} words)'),
    ]

    return ChatPromptTemplate.from_messages(messages)


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
        'You will be given a paragraph which is an answer to a grant application question submitted by a nonprofit.\n'
        'The grant application questions was: "{question} ({word_limit} words)".\n'
        'Improve the answer by incorporating the potentially valuable information contained in the following lines:\n'
        '----------------\n'
        '{answers_to_implicit_questions}.\n'
        '----------------\n'
        'Start your message right away with the new answer without any leading words. '
        'Please make sure to comply with the word limit stated at the end of the grant application question.'
    )

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('{original_answer}'),
    ]

    return ChatPromptTemplate.from_messages(messages)
