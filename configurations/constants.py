from enum import IntEnum, StrEnum, auto
import os

from gradio import utils


class StepID(StrEnum):
    START = auto()

    HAVE_MATERIALS_TO_SHARE = auto()
    UPLOAD_FILES = auto()

    ENTER_QUESTION = auto()
    ENTER_WORD_LIMIT = auto()

    ENTER_RAG_CONFIG_ORIGINAL_QUESTION = auto()
    ENTER_RAG_CONFIG_IMPLICIT_QUESTION = auto()
    GO_BACK_TO_CONFIG_STEP_ORIGINAL_QUESTION = auto()
    GO_BACK_TO_CONFIG_STEP_IMPLICIT_QUESTION = auto()

    GO_OVER_IMPLICIT_QUESTIONS = auto()

    # for each implicit question:
    DO_PROCEED_WITH_IMPLICIT_QUESTION = auto()
    SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT = auto()
    PROMPT_USER_TO_SUBMIT_ANSWER = auto()

    READY_TO_GENERATE_FINAL_ANSWER = auto()
    ASK_USER_IF_GUIDANCE_NEEDED = auto()
    USER_GUIDANCE_PROMPT = auto()

    DO_ANOTHER_QUESTION = auto()

    END = auto()

class Component(IntEnum):
    START = auto()
    YES = auto()
    NO = auto()
    FILES = auto()
    WORD_LIMIT = auto()
    GOOD_AS_IS = auto()
    EDIT_IT = auto()
    ADD_GUIDANCE = auto()
    OF_COURSE = auto()
    NUM_OF_TOKENS = auto()
    NUM_OF_DOCS = auto()

class ComponentLabel(StrEnum):
    CHATBOT = 'AI Grant Writing Coach 🚀'
    USER = 'User'
    SUBMIT = 'Submit'
    WORD_LIMIT = 'Word Limit'
    NUM_OF_TOKENS = '# Tokens'
    NUM_OF_DOCS = '# Documents'
    SYSTEM_PROMPT = 'System Prompt'
    EXAMPLES = 'Examples of grant application questions'
    START = 'I\'m ready!'
    YES = 'Yes'
    NO = 'No'
    FILES = 'Documents'
    UPLOAD = 'Upload'
    CLEAR = 'Clear'
    GOOD_AS_IS = 'Good as is!'
    EDIT_IT = 'Let me edit it'
    ADD_GUIDANCE = 'Let me add some guidance'
    OF_COURSE = 'Of course I\'m ready!'
    REPORT_MARKDOWN = 'Report'
    APPLICATION_ANSWERS = 'Application Answers - Click to Download'

class ComponentID(StrEnum):
    CHATBOT = auto()
    USER_TEXT_BOX = auto()
    NUMBER_1 = auto()
    NUMBER_2 = auto()
    SUBMIT_USER_INPUT_BTN = auto()
    EXAMPLES = auto()
    BTN_1 = auto()
    BTN_2 = auto()
    FILES = auto()
    UPLOAD_FILES_BTN = auto()
    CLEAR_FILES_BTN = auto()
    SUBMIT_FILES_BTN = auto()
    ANSWERS_MARKUP = auto()
    ANSWERS_FILE = auto()

DEFAULT_WORD_LIMIT = 150
DEFAULT_NUM_OF_TOKENS = 1000
DEFAULT_NUM_OF_DOC_CHUNKS = 2


GRANT_APPLICATION_QUESTIONS_EXAMPLES = [
    'What is your mission?',
    'Give me a background of your organization.',
    'What are your achievements to date?',
    'Where does this project fit within your organizational strategy and vision?',
    'What is your organization\'s approach to measuring impact?',
    'What are your organization\'s goals for the next 3-5 years?',
    'How is your organization building an inclusive workplace culture?',
    'What are your diversity, equity, and inclusion goals?'
]

PAGE_TITLE = "Publico.ai - Demo"

GPT_MODEL = 'gpt-4-turbo-preview' if os.getenv('GPT_MODEL', 'gpt-4') not in ('3.5', 'gpt-3.5', 'gpt-3.5-turbo') else 'gpt-3.5-turbo'
IS_DEV_MODE = os.getenv('DEV', 'False').lower() in ('true', 't', '1', 'yes')

SERVER_PORT = int(os.getenv('SERVER_PORT', 7860))
CREATE_LINK = utils.get_space() is None and os.getenv('CREATE_LINK', 'True').lower() in ('true', 't', '1', 'yes')
EXCLUDE_LOGO = os.getenv("EXCLUDE_LOGO", 'False').lower() in ('true', 't', '1', 'yes')

CHATBOT_HEIGHT = int(os.getenv('CHATBOT_HEIGHT', 580))
CHATBOT_LAYOUT = 'bubble' if os.getenv('CHATBOT_LAYOUT', 'panel').lower() in ('bubble', 'b') else 'panel'

SYSTEM_PROMPT_FOR_ANSWERING_ORIGINAL_QUESTION = (
    'You are going to help a nonprofit organization that is applying for a grant.\n'
    'Use the following pieces of context to respond to a grant application question '
    'in a way that provides a compelling and comprehensive answer from the perspective '
    'of a nonprofit organization applying for grant funding.\n'
    '----------------\n'
    '{context}\n'
    '----------------\n'
    'Make sure to comply with the word limit stated in parentheses at the end of the grant application question as this is crucial! (but do not write the word count itself in the generated answer)'
)

SYSTEM_PROMPT_FOR_ANSWERING_IMPLICIT_QUESTION = (
    'You are a grantwriting expert who will be helping a non-profit organization applying for a grant. '
    'Please provide your best answer to the following question using the context provided. '
    'Be as concise as possible, using at most one or two lines. '
    'If you can\'t answer the question, don\'t make something up and simply answer the words \'Not enough information provided.\'.\n'
    '----------------\n'
    '{context}'
)
