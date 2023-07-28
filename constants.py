from enum import Enum, auto


class StepID(Enum):
    START = auto()
    HAVE_YOU_APPLIED_BEFORE = auto()
    UPLOAD_PRIOR_GRANT_APPLICATIONS = auto()
    ENTER_QUESTION = auto()
    ENTER_WORD_LIMIT = auto()
    DO_COMPREHENSIVENESS_CHECK = auto()
    DO_ANOTHER_QUESTION = auto()
    END = auto()


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
