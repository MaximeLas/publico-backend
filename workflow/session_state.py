
from dataclasses import dataclass, field
import datetime

from langchain.docstore.document import Document
from pydantic import UUID4

from configurations.constants import (
    DEFAULT_NUM_OF_DOC_CHUNKS,
    DEFAULT_NUM_OF_TOKENS, 
    IS_DEV_MODE, 
    SYSTEM_PROMPT_FOR_ANSWERING_ORIGINAL_QUESTION,
    SYSTEM_PROMPT_FOR_ANSWERING_IMPLICIT_QUESTION,
    StepID
)


@dataclass
class ImplicitQuestion:
    question: str
    answer: str | None = None


@dataclass
class ComprehensivenessCheckerContext:
    missing_information: str | None = None
    implicit_questions: list[ImplicitQuestion] = field(default_factory=list)
    index_of_implicit_question_being_answered: int | None = None
    wish_to_answer_implicit_questions: bool = True
    revised_application_answer: str | None = None


@dataclass
class Improvement:
    user_prompt: str
    improved_answer: str


@dataclass
class PolishContext:
    improvements: list[Improvement] = field(default_factory=list)

@dataclass
class EditedAnswer:
    time: datetime.datetime
    previous_answer: str
    new_answer: str

@dataclass
class GrantApplicationQuestionContext:
    question: str | None = None
    word_limit: str | None = None
    answer: str | None = None # change this to original_answer
    comprehensiveness: ComprehensivenessCheckerContext = field(default_factory=ComprehensivenessCheckerContext)
    polish: PolishContext = field(default_factory=PolishContext)
    edited_answers: list[EditedAnswer] = field(default_factory=list)
    current_answer: str | None = None # TODO: point this to the last answer in current workflow

    def get_original_answer(self) -> str | None:
        return self.answer

    def get_revised_answer(self) -> str | None:
        return self.comprehensiveness.revised_application_answer

    def get_last_improved_answer(self,) -> str | None:
        if self.polish.improvements:
            return self.polish.improvements[-1].improved_answer
        else:
            return None


@dataclass
class TestConfigContext:
    system_prompt: str = None
    num_of_tokens_per_doc_chunk: int = 1000
    dev_has_changed_num_of_tokens: bool = False
    num_of_doc_chunks_to_consider: int = 4
    system_prompt: str = None


@dataclass
class SessionState:
    session_id: str
    user_id: str
    uploaded_files: list[str] = field(default_factory=list)
    questions: list[GrantApplicationQuestionContext] = field(default_factory=list)
    current_step_id: StepID = StepID.START
    chat_history: list[list] = field(default_factory=list)
    last_user_input = None
    test_config: TestConfigContext = field(default_factory=TestConfigContext) if IS_DEV_MODE else None


    def add_new_question(self):
        self.questions.append(GrantApplicationQuestionContext())


    def get_last_question_context(self) -> GrantApplicationQuestionContext:
        return self.questions[-1]

    def get_index_of_last_question(self) -> int:
        return len(self.questions) - 1


    def set_uploaded_files(self, files: list[str]):
        if files is None:
            files = ['./PBRC.txt', './PBRC2.txt']

        self.uploaded_files = files


    def set_grant_application_question(self, question: str):
        self.questions[-1].question = question


    def set_word_limit(self, word_limit: str):
        self.questions[-1].word_limit = word_limit


    def set_answer_to_current_grant_application_question(self, answer: str):
        self.questions[-1].answer = answer


    def get_index_of_implicit_question_being_answered(self) -> int | None:
        return self.questions[-1].comprehensiveness.index_of_implicit_question_being_answered


    def get_current_implicit_question(self) -> str:
        if (index := self.get_index_of_implicit_question_being_answered()) is not None:
            return self.questions[-1].comprehensiveness.implicit_questions[index].question
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

        if index is None or index > len(implicit_questions):
            print(f'index: {index}, len: {len(implicit_questions)}')
            raise Exception('Cannot set answer as no implicit question currently being answered')

        implicit_questions[index].answer = answer


    def get_next_implicit_question_and_index(self) -> tuple[str, int]:
        comprehensiveness = self.questions[-1].comprehensiveness

        if (index := self.get_index_of_implicit_question_being_answered()) is not None:
            if index == len(comprehensiveness.implicit_questions) - 1:
                raise Exception('No more implicit questions to answer')

            comprehensiveness.index_of_implicit_question_being_answered += 1
            return comprehensiveness.implicit_questions[index + 1].question, index + 1
        else:
            comprehensiveness.index_of_implicit_question_being_answered = 0
            return comprehensiveness.implicit_questions[0].question, 0

    
    def has_more_implcit_questions_to_answer(self) -> bool:
        index = self.get_index_of_implicit_question_being_answered()

        return index is None or index < len(self.questions[-1].comprehensiveness.implicit_questions) - 1


    def exists_answer_to_any_implicit_question(self) -> bool:
        return any([question.answer for question in self.questions[-1].comprehensiveness.implicit_questions])


    def set_revised_answer_to_current_grant_application_question(self, answer: str):
        self.questions[-1].comprehensiveness.revised_application_answer = answer


    def get_current_user_guidance_prompt(self) -> str:
        return self.questions[-1].polish.improvements[-1].user_prompt


    def set_user_guidance_prompt(self, prompt: str):
        self.questions[-1].polish.improvements.append(Improvement(user_prompt=prompt, improved_answer=None))


    def get_current_improvements(self) -> list[Improvement]:
        return self.questions[-1].polish.improvements


    def set_improved_answer(self, answer: str):
        self.questions[-1].polish.improvements[-1].improved_answer = answer


    def is_allowed_to_add_more_guidance(self) -> bool:
        return len(self.questions[-1].polish.improvements) < 3


    def edit_last_question(self, question_index: int, answer: str):
        question = self.questions[question_index]
        question.edited_answers.append(EditedAnswer(
            time=datetime.datetime.now(),
            previous_answer=question.current_answer,
            new_answer=answer
        ))

        question.current_answer = answer


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
