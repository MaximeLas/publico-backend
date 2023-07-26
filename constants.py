from enum import Enum, auto


class UserInteractionType(Enum):
    YES_NO = auto()
    FILES = auto()
    UPLOAD = auto()
    SUBMIT = auto()
    CLEAR = auto()
    TEXT = auto()
    SUBMIT_TEXT = auto()
    START = auto()
    NONE = auto()

class ContextKeys(Enum):
    HAS_APPLIED_FOR_THIS_GRANT_BEFORE = auto()
    PRIOR_GRANT_APPLICATIONS = auto()
    APPLICATION_QUESTION = auto()
    WORD_LIMIT = auto()
    MOST_RELEVANT_DOCUMENTS = auto()
    APPLICATION_ANSWER = auto()
    CHECK_COMPREHENSIVENESS = auto()
    MISSING_INFORMATION = auto()
    IMPLICIT_QUESTIONS = auto()
    ANSWERS_TO_IMPLICIT_QUESTIONS = auto()
    TRY_WITH_ANOTHER_QUESTION = auto()

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
