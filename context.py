
from dataclasses import dataclass, field
import tempfile

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

from constants import ComponentLabel

    
@dataclass
class ImplicitQuestion:
    #index: int
    question: str
    answer: str | None = None

@dataclass
class ComprehensivenessCheckerContext:
    do_check: bool = False
    missing_information: str | None = None
    implicit_questions: dict[int, ImplicitQuestion] = field(default_factory=dict)
    index_of_implicit_question_being_answered: int | None = None
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
class FilesStorageContext:
    files: list[str] = field(default_factory=list)
    vector_store: Chroma = None

@dataclass
class UserContext:
    uploaded_files: FilesStorageContext = field(default_factory=FilesStorageContext)
    questions: list[GrantApplicationQuestionContext] = field(default_factory=lambda: [GrantApplicationQuestionContext()])


    def add_new_question(self):
        self.questions.append(GrantApplicationQuestionContext())


    def get_last_question_context(self) -> GrantApplicationQuestionContext:
        return self.questions[-1]


    def set_uploaded_files(self, files: list[tempfile._TemporaryFileWrapper]):
        self.uploaded_files.files = [file.name for file in files]


    def set_grant_application_question(self, question: str):
        self.questions[-1].question = question


    def set_word_limit(self, word_limit: str):
        self.questions[-1].word_limit = word_limit


    def set_do_check_for_comprehensiveness(self, yes_or_no: str):
        self.questions[-1].comprehensiveness.do_check = yes_or_no == ComponentLabel.YES


    def get_index_of_implicit_question_being_answered(self) -> int | None:
        return self.questions[-1].comprehensiveness.index_of_implicit_question_being_answered


    def get_current_implicit_question_to_be_answered(self) -> str:
        index = self.get_index_of_implicit_question_being_answered()
        if not index:
            raise Exception('No implicit question currently being answered')
        else:
            return self.questions[-1].comprehensiveness.implicit_questions[index].question
    
    
    def get_answer_of_current_implicit_question_to_be_answered(self) -> str | None:
        index = self.get_index_of_implicit_question_being_answered()
        if not index:
            raise Exception('No implicit question currently being answered')
        else:
            return self.questions[-1].comprehensiveness.implicit_questions[index].answer


    def set_answer_to_current_implicit_question(self, answer: str):
        index = self.get_index_of_implicit_question_being_answered()
        implicit_questions = self.questions[-1].comprehensiveness.implicit_questions

        if not index or index > len(implicit_questions):
            raise Exception('Cannot set answer as no implicit question currently being answered')

        implicit_questions[index].answer = answer


    def get_next_implicit_question_to_be_answered(self) -> str:
        comprehensiveness = self.questions[-1].comprehensiveness
        if not comprehensiveness.implicit_questions:
            raise Exception('No implicit questions present')

        index = self.get_index_of_implicit_question_being_answered()
        if index == len(comprehensiveness.implicit_questions):
            raise Exception('No more implicit questions to answer')
        elif not index:
            comprehensiveness.index_of_implicit_question_being_answered = 1
            return comprehensiveness.implicit_questions[1].question
        else:
            comprehensiveness.index_of_implicit_question_being_answered += 1
            return comprehensiveness.implicit_questions[index + 1].question

    
    def has_more_implcit_questions_to_answer(self) -> bool:
        index = self.get_index_of_implicit_question_being_answered()
        return index is None or index < len(self.questions[-1].comprehensiveness.implicit_questions)
