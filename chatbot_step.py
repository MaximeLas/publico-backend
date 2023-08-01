from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from devtools import debug

from gradio.blocks import Block

from constants import StepID
from context import UserContext



MessageOutputType = str | Iterator[str] | list[str] | Iterator[list[str]]
GenerateMessageFunc = Callable[[UserContext], MessageOutputType]
GenerateMessageFuncList = list[GenerateMessageFunc]


@dataclass
class EventOutcomeSaver:
    save_fn: Callable[[UserContext, Any], None]
    component_name: str | None


@dataclass
class StepDecider:
    next_step: StepID

    def determine_next_step(self, context: UserContext) -> StepID:
        return self.next_step


@dataclass
class ConditionalStepDecider(StepDecider):
    alternative_step: StepID
    condition: Callable[['UserContext'], bool]

    def determine_next_step(self, context: UserContext) -> StepID:
        return self.next_step if self.condition(context) else self.alternative_step


@dataclass
class ChatbotStep():
    initial_chatbot_message: str | MessageOutputType
    step_decider: StepDecider | dict[str, StepDecider]
    components: list[Block] = field(default_factory=list)
    save_event_outcome: EventOutcomeSaver | None = None
    generate_chatbot_messages_fns: GenerateMessageFuncList | dict[str, GenerateMessageFuncList] = field(default_factory=lambda: defaultdict(list))
    retrieve_relevant_vars_func: Callable[[UserContext], dict] = lambda _: dict()

    def determine_next_step(self, trigger: str, context: UserContext) -> StepID:
        step_decider = (
            self.step_decider
                if isinstance(self.step_decider, StepDecider)
                else
            self.step_decider[trigger])
        debug(step_decider.determine_next_step(context))
        return step_decider.determine_next_step(context)


    def get_generate_chatbot_messages_fns_for_trigger(self, trigger: str) -> list[GenerateMessageFunc]:
        fns = self.generate_chatbot_messages_fns
        return fns if isinstance(fns, list) else fns[trigger]
    
