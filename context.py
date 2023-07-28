
from dataclasses import dataclass, field
import tempfile

from langchain.docstore.document import Document

from constants import StepID

    

@dataclass
class ComprehensivenessCheckerContext:
    do_check: bool = False
    missing_information: str | None = None
    implicit_questions: list[str] = field(default_factory=list)
    answers_to_implicit_questions: dict[int, str] = field(default_factory=dict)
    wish_to_answer_implicit_questions: bool = True
    revised_application_answer: str | None = None


@dataclass
class GrantApplicationQuestionContext:
    question: str | None = None
    word_limit: str | None = None
    most_relevant_documents: list[Document] = field(default_factory=list)
    answer: str | None = None
    comprehensiveness: ComprehensivenessCheckerContext = field(default_factory=ComprehensivenessCheckerContext)


@dataclass
class UserContext:
    current_step_id: StepID = StepID.START
    questions: list[GrantApplicationQuestionContext] = field(default_factory=lambda: [GrantApplicationQuestionContext()])
    prior_grant_applications: list[str] = field(default_factory=list)


def set_prior_grant_applications(context: UserContext, files: list[tempfile._TemporaryFileWrapper]):
    context.prior_grant_applications = [file.name for file in files]
    return context

def set_grant_application_question(context: UserContext, question: str):
    context.questions[-1].question = question
    return context

def set_word_limit(context: UserContext, word_limit: str):
    context.questions[-1].word_limit = word_limit
    return context

def set_do_check_for_comprehensiveness(context: UserContext, yes_or_no: str):
    context.questions[-1].comprehensiveness.do_check = yes_or_no == 'Yes'
    return context

def set_current_step_id(context: UserContext, step_id: StepID):
    context.current_step_id = step_id
    return context
