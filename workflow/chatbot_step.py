from asyncio import Queue
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from configurations.constants import Component, StepID
from workflow.session_state import SessionState
from workflow.step_decider import StepDecider

class EditorContentType(Enum):
    QUESTION = auto()
    WORD_LIMIT = auto()
    ANSWER = auto()

GenerateMsgFns = list[Callable[[SessionState, Queue], None]]

@dataclass
class ChatbotStep():
    initial_chatbot_message: str | Callable[[SessionState], str]
    components: set[Component] | Callable[[SessionState], set[Component]]
    save_event_outcome_fn: Callable[[SessionState, Any], None] | None = None
    next_step_decider: StepDecider | dict[Component, StepDecider] | None = None
    initialize_step_func: Callable[[SessionState], None] = lambda _: None
    generate_chatbot_messages_fns: GenerateMsgFns | dict[Component, GenerateMsgFns] = field(default_factory=list)
    updated_editor_contents: set[EditorContentType] = field(default_factory=set)

    def get_initial_chatbot_message(self, state: SessionState) -> str:
        return (
            self.initial_chatbot_message(state)
                if callable(self.initial_chatbot_message)
                else
            self.initial_chatbot_message)

    def determine_next_step(self, state: SessionState) -> StepID:
        next_step_decider = (
            self.next_step_decider
                if isinstance(self.next_step_decider, StepDecider)
                else
            self.next_step_decider[state.last_user_input])

        return next_step_decider.determine_next_step(state)

    def get_components(self, state: SessionState) -> set[Component]:
        return (
            self.components(state)
                if callable(self.components)
                else
            self.components)

    def get_generate_chatbot_messages_fns_for_trigger(self, trigger: Component | None) -> GenerateMsgFns:
        fns = self.generate_chatbot_messages_fns
        if isinstance(fns, dict):
            assert trigger is not None and isinstance(trigger, Component)
            return fns.get(trigger, [])
        else:
            return fns
