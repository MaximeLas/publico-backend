from enum import Enum, auto


class UserInteractionType(Enum):
    YES_NO = auto()
    FILES = auto()
    UPLOAD = auto()
    SUBMIT = auto()
    CLEAR = auto()
    TEXT = auto()
    START = auto()
    NONE = auto()

class OutputKeys(Enum):
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
