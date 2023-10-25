
from dataclasses import dataclass, field
import tempfile

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

from constants import (
    ComponentLabel,
    DEFAULT_NUM_OF_DOC_CHUNKS,
    DEFAULT_NUM_OF_TOKENS, 
    IS_DEV_MODE, 
    SYSTEM_PROMPT_FOR_ANSWERING_ORIGINAL_QUESTION,
    SYSTEM_PROMPT_FOR_ANSWERING_IMPLICIT_QUESTION
)
    
@dataclass
class ImplicitQuestion:
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
class TestConfigContext:
    system_prompt: str = None
    num_of_tokens_per_doc_chunk: int = 1000
    dev_has_changed_num_of_tokens: bool = False
    num_of_doc_chunks_to_consider: int = 4



@dataclass
class UserContext:
    uploaded_files: FilesStorageContext = field(default_factory=FilesStorageContext)
    questions: list[GrantApplicationQuestionContext] = field(default_factory=lambda: [GrantApplicationQuestionContext()])
    test_config: TestConfigContext = field(default_factory=TestConfigContext) if IS_DEV_MODE else None


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


    def set_test_config_params(self, prompt: str, num_of_tokens: str, num_of_doc_chunks: str):
        self.test_config.system_prompt = prompt
        self.test_config.dev_has_changed_num_of_tokens = self.test_config.num_of_tokens_per_doc_chunk != num_of_tokens
        self.test_config.num_of_tokens_per_doc_chunk = num_of_tokens
        self.test_config.num_of_doc_chunks_to_consider = num_of_doc_chunks


    def get_system_prompt_for_original_question(self) -> str:
        return self.test_config.system_prompt if IS_DEV_MODE else SYSTEM_PROMPT_FOR_ANSWERING_ORIGINAL_QUESTION


    def get_system_prompt_for_implicit_question(self) -> str:
        return self.test_config.system_prompt if IS_DEV_MODE else SYSTEM_PROMPT_FOR_ANSWERING_IMPLICIT_QUESTION


    def get_num_of_tokens_per_doc_chunk(self) -> int:
        return self.test_config.num_of_tokens_per_doc_chunk if IS_DEV_MODE else DEFAULT_NUM_OF_TOKENS


    def user_has_changed_num_of_tokens(self) -> bool:
        return self.test_config.dev_has_changed_num_of_tokens


    def get_num_of_doc_chunks_to_consider(self) -> int:
        return self.test_config.num_of_doc_chunks_to_consider if IS_DEV_MODE else DEFAULT_NUM_OF_DOC_CHUNKS


    def set_do_check_for_comprehensiveness(self, yes_or_no: str):
        self.questions[-1].comprehensiveness.do_check = yes_or_no == ComponentLabel.YES


    def get_index_of_implicit_question_being_answered(self) -> int | None:
        return self.questions[-1].comprehensiveness.index_of_implicit_question_being_answered


    def get_current_implicit_question(self) -> str:
        if (index := self.get_index_of_implicit_question_being_answered()) is not None:
            return self.questions[-1].comprehensiveness.implicit_questions[index].question
        else:
            raise Exception('No implicit question currently being answered')
    
    
    def get_answer_of_current_implicit_question(self) -> str | None:
        if (index := self.get_index_of_implicit_question_being_answered()) is not None:
            return self.questions[-1].comprehensiveness.implicit_questions[index].answer
        else:
            raise Exception('No implicit question currently being answered')


    def exists_answer_to_current_implicit_question(self) -> bool:
        if (index := self.get_index_of_implicit_question_being_answered()) is not None:
            return self.questions[-1].comprehensiveness.implicit_questions[index].answer is not None
        else:
            raise Exception('No implicit question currently being answered')


    def set_answer_to_current_implicit_question(self, answer: str):
        index = self.get_index_of_implicit_question_being_answered()
        implicit_questions = self.questions[-1].comprehensiveness.implicit_questions

        if not index or index > len(implicit_questions):
            raise Exception('Cannot set answer as no implicit question currently being answered')

        implicit_questions[index].answer = answer


    def get_next_implicit_question(self) -> str:
        comprehensiveness = self.questions[-1].comprehensiveness

        if (index := self.get_index_of_implicit_question_being_answered()) is not None:
            if index == len(comprehensiveness.implicit_questions):
                raise Exception('No more implicit questions to answer')

            comprehensiveness.index_of_implicit_question_being_answered += 1
            return comprehensiveness.implicit_questions[index + 1].question
        else:
            comprehensiveness.index_of_implicit_question_being_answered = 1
            return comprehensiveness.implicit_questions[1].question

    
    def has_more_implcit_questions_to_answer(self) -> bool:
        index = self.get_index_of_implicit_question_being_answered()

        return index is None or index < len(self.questions[-1].comprehensiveness.implicit_questions)


    def exists_answer_to_any_implicit_question(self) -> bool:
        return any([question.answer for question in self.questions[-1].comprehensiveness.implicit_questions.values()])
