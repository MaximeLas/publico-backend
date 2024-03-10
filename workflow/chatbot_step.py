from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from configurations.constants import Component, StepID
from workflow.session_state import SessionState
from workflow.step_decider import StepDecider



MessageOutputType = str | list[str] | Iterator[str | list[str] | None] | None
GenerateMessageFunc = Callable[[SessionState], MessageOutputType]
GenerateMessageFuncList = list[GenerateMessageFunc]
ComponentPropertiesType = dict[str, str]


class EditorContentType(Enum):
    QUESTION = auto()
    WORD_LIMIT = auto()
    ANSWER = auto()


@dataclass
class InitialChatbotMessage:
    message: str
    extract_formatting_variables_func: Callable[['SessionState'], any] = lambda _: dict()

    def get_formatted_message(self, state: SessionState) -> Iterator[str]:
        #for response_so_far in self.extract_formatting_variables_func(state):
        if isinstance(response_so_far := self.extract_formatting_variables_func(state), dict):
            return self.message.format(**response_so_far)
        else:
            return self.message.format(response=response_so_far)


@dataclass
class ChatbotStep():
    initial_chatbot_message: InitialChatbotMessage
    next_step_decider: StepDecider | dict[str, StepDecider]
    components: set[Component] | Callable[[SessionState], set[Component]] = field(default_factory=set)
    initialize_step_func: Callable[[SessionState], None] = lambda _: None
    save_event_outcome_fn: Callable[[SessionState, Any], None] | None = None
    generate_chatbot_messages_fns: GenerateMessageFuncList | dict[str, GenerateMessageFuncList] = field(default_factory=lambda: defaultdict(list))
    updated_editor_contents: set[EditorContentType] = field(default_factory=set)

    def determine_next_step(self, trigger: str, state: SessionState) -> StepID:
        next_step_decider = (
            self.next_step_decider
                if isinstance(self.next_step_decider, StepDecider)
                else
            self.next_step_decider[trigger])

        return next_step_decider.determine_next_step(state)

    def get_components(self, state: SessionState) -> set[Component]:
        return self.components if isinstance(self.components, set) else self.components(state)

    def get_generate_chatbot_messages_fns_for_trigger(self, trigger: str) -> list[GenerateMessageFunc]:
        fns = self.generate_chatbot_messages_fns
        return fns if isinstance(fns, list) else fns.get(trigger, [])
