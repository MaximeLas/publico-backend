
from dataclasses import dataclass, field
import tempfile

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

from configurations.constants import (
    DEFAULT_NUM_OF_DOC_CHUNKS,
    DEFAULT_NUM_OF_TOKENS, 
    IS_DEV_MODE, 
    SYSTEM_PROMPT_FOR_ANSWERING_ORIGINAL_QUESTION,
    SYSTEM_PROMPT_FOR_ANSWERING_IMPLICIT_QUESTION
)



@dataclass
class TextFormat:
    original: str = ''
    formatted: str = ''


@dataclass
class ImplicitQuestion:
    question: str
    answer: TextFormat | None = None


@dataclass
class ComprehensivenessCheckerContext:
    missing_information: str | None = None
    implicit_questions: dict[int, ImplicitQuestion] = field(default_factory=dict)
    index_of_implicit_question_being_answered: int | None = None
    wish_to_answer_implicit_questions: bool = True
    revised_application_answer: TextFormat | None = None


@dataclass
class Improvement:
    user_prompt: str
    improved_answer: TextFormat


@dataclass
class PolishContext:
    improvements: list[Improvement] = field(default_factory=list)


@dataclass
class GrantApplicationQuestionContext:
    question: str | None = None
    word_limit: str | None = None
    most_relevant_documents: list[Document] = field(default_factory=list)
    answer: TextFormat | None = None
    comprehensiveness: ComprehensivenessCheckerContext = field(default_factory=ComprehensivenessCheckerContext)
    polish: PolishContext = field(default_factory=PolishContext)

    def get_original_answer(self, format: bool) -> str | None:
        if self.answer:
            return self.answer.formatted if format else self.answer.original
        else:
            return None

    def get_revised_answer(self, format: bool) -> str | None:
        if self.comprehensiveness.revised_application_answer:
            return (
                self.comprehensiveness.revised_application_answer.formatted
                    if format else
                self.comprehensiveness.revised_application_answer.original
            )
        else:
            return None

    def get_last_improved_answer(self, format: bool) -> str | None:
        if self.polish.improvements:
            return (
                self.polish.improvements[-1].improved_answer.formatted
                    if format else
                self.polish.improvements[-1].improved_answer.original
            )
        else:
            return None


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
    system_prompt: str = None


@dataclass
class AppContext:
    uploaded_files: FilesStorageContext = field(default_factory=FilesStorageContext)
    questions: list[GrantApplicationQuestionContext] = field(default_factory=list)
    full_application: TextFormat = field(default_factory=TextFormat)
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


    def set_answer_to_current_grant_application_question(
        self,
        original: str,
        formatted: str | None = None
    ):
        self.questions[-1].answer = TextFormat(original, formatted)


    def get_index_of_implicit_question_being_answered(self) -> int | None:
        return self.questions[-1].comprehensiveness.index_of_implicit_question_being_answered


    def get_current_implicit_question(self) -> str:
        if (index := self.get_index_of_implicit_question_being_answered()) is not None:
            return self.questions[-1].comprehensiveness.implicit_questions[index].question
        else:
            raise Exception('No implicit question currently being answered')
    
    
    def get_answer_of_current_implicit_question(self) -> str | None:
        if (index := self.get_index_of_implicit_question_being_answered()) is not None:
            return self.questions[-1].comprehensiveness.implicit_questions[index].answer.original
        else:
            raise Exception('No implicit question currently being answered')


    def exists_answer_to_current_implicit_question(self) -> bool:
        if (index := self.get_index_of_implicit_question_being_answered()) is not None:
            return self.questions[-1].comprehensiveness.implicit_questions[index].answer is not None
        else:
            raise Exception('No implicit question currently being answered')


    def set_answer_to_current_implicit_question(
        self,
        original: str,
        formatted: str | None = None
    ):
        index = self.get_index_of_implicit_question_being_answered()
        implicit_questions = self.questions[-1].comprehensiveness.implicit_questions

        if not index or index > len(implicit_questions):
            raise Exception('Cannot set answer as no implicit question currently being answered')

        implicit_questions[index].answer = TextFormat(original, formatted)


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


    def set_revised_answer_to_current_grant_application_question(
        self,
        original: str,
        formatted: str | None = None
    ):
        self.questions[-1].comprehensiveness.revised_application_answer = TextFormat(original, formatted)


    def get_current_user_guidance_prompt(self) -> str:
        return self.questions[-1].polish.improvements[-1].user_prompt


    def set_user_guidance_prompt(self, prompt: str):
        self.questions[-1].polish.improvements.append(Improvement(user_prompt=prompt, improved_answer=None))


    def get_current_improvements(self) -> list[Improvement]:
        return self.questions[-1].polish.improvements


    def set_improved_answer(self, original: str, formatted: str | None = None):
        self.questions[-1].polish.improvements[-1].improved_answer = TextFormat(original, formatted)


    def is_allowed_to_add_more_guidance(self) -> bool:
        return len(self.questions[-1].polish.improvements) < 3


    def get_completed_application(self) -> tuple[str, str]:
        report_original = ''
        report_formatted = ''
        for i, question in enumerate(self.questions):
            report_original += f'Question {i+1}: {question.question}'
            report_formatted += f'## Question {i+1}\n **{question.question}**'

            if question.word_limit:
                report_original += f' ({question.word_limit} words)'
                report_formatted += f' ({question.word_limit} words)'

            answer_original = (
                question.get_last_improved_answer(False) or
                question.get_revised_answer(False) or
                question.get_original_answer(False) or
                ''
            )
            answer_formatted = (
                question.get_last_improved_answer(True) or
                question.get_revised_answer(True) or
                question.get_original_answer(True) or
                ''
            )

            if answer_original:
                report_original += f'\n\n{answer_original}\n({len(answer_original.split())} words)\n\n'
                report_formatted += f'\n\n{answer_formatted}\n({len(answer_formatted.split())} words)\n\n'

        if self.full_application.original == report_original:
            return None, None

        self.full_application.original = report_original
        self.full_application.formatted = report_formatted

        return report_original, report_formatted


    ''' Test Config methods '''

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
