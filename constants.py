from enum import StrEnum, auto


class StepID(StrEnum):
    START = auto()
    HAVE_YOU_APPLIED_BEFORE = auto()
    UPLOAD_FILES = auto()
    ENTER_QUESTION = auto()
    ENTER_WORD_LIMIT = auto()
    DO_COMPREHENSIVENESS_CHECK = auto()
    # for each implicit question:
    DO_PROCEED_WITH_IMPLICIT_QUESTION = auto()
    SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT = auto()
    PROMPT_USER_TO_SUBMIT_ANSWER = auto()
    READY_TO_GENERATE_FINAL_ANSWER = auto() # at the end of the questions
    # end for
    DO_ANOTHER_QUESTION = auto()
    END = auto()


class ComponentLabel(StrEnum):
    CHATBOT = 'AI Grant Writing Coach'
    USER = 'User'
    SUBMIT = 'Submit'
    NUMBER = 'Number'
    EXAMPLES = 'Examples of grant application questions'
    START = 'Start'
    YES = 'Yes'
    NO = 'No'
    FILES = 'Documents'
    UPLOAD = 'Upload'
    CLEAR = 'Clear'
    GOOD_AS_IS = 'Good as is!'
    EDIT_IT = 'Let me edit it'


GRANT_APPLICATION_QUESTIONS_EXAMPLES = [
    'What is your mission?',
    'Give me a background of your organization.',
    'What are your achievements to date?',
    'Where does this project fit within your organizational strategy and vision?',
    'How is your organization building an inclusive workplace culture? What are your diversity, equity, and inclusion goals?',
    'How does the proposed project contribute to the foundation\'s funding priority of increasing diversity, equity, and inclusion (DEI)?',
    'What is your organization\'s approach to measuring impact?',
    'What are your organization\'s goals for the next 3-5 years?'
]
