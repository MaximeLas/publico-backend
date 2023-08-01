
from dataclasses import dataclass, field
import tempfile

from langchain.docstore.document import Document

    

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
    prior_grant_applications: list[str] = field(default_factory=list)
    questions: list[GrantApplicationQuestionContext] = field(default_factory=lambda: [GrantApplicationQuestionContext()])

    def set_prior_grant_applications(self, files: list[tempfile._TemporaryFileWrapper]):
        self.prior_grant_applications = [file.name for file in files]

    def set_grant_application_question(self, question: str):
        self.questions[-1].question = question

    def set_word_limit(self, word_limit: str):
        self.questions[-1].word_limit = word_limit

    def set_do_check_for_comprehensiveness(self, yes_or_no: str):
        self.questions[-1].comprehensiveness.do_check = yes_or_no == 'Yes'
