from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from gradio.blocks import Block

from constants import StepID
from context import UserContext



MessageOutputType = str | list[str] | Iterator[str | list[str] | None] | None
GenerateMessageFunc = Callable[[UserContext], MessageOutputType]
GenerateMessageFuncList = list[GenerateMessageFunc]


@dataclass
class EventOutcomeSaver:
    save_fn: Callable[[UserContext, Any], None]
    component_name: str | None


@dataclass
class NextStepDecider:
    next_step: StepID

    def determine_next_step(self, context: UserContext) -> StepID:
        return self.next_step


@dataclass
class ConditionalNextStepDecider(NextStepDecider):
    alternative_step: StepID
    condition: Callable[['UserContext'], bool]

    def determine_next_step(self, context: UserContext) -> StepID:
        return self.next_step if self.condition(context) else self.alternative_step


@dataclass
class InitialChatbotMessage:
    message: str
    extract_formatting_variables_func: Callable[[UserContext], Iterator] = lambda _: (yield dict())

    def get_formatted_message(self, context: UserContext) -> Iterator[str]:
        for response in self.extract_formatting_variables_func(context):
            if isinstance(response, dict):
                yield self.message.format(**response)
            else:
                yield self.message.format(response=response)


@dataclass
class ChatbotStep():
    initial_chatbot_message: InitialChatbotMessage
    next_step_decider: NextStepDecider | dict[str, NextStepDecider]
    components: list[Block] = field(default_factory=list)
    initialize_step_func: Callable[[UserContext], None] = lambda _: None
    save_event_outcome: EventOutcomeSaver | None = None
    generate_chatbot_messages_fns: GenerateMessageFuncList | dict[str, GenerateMessageFuncList] = field(default_factory=lambda: defaultdict(list))
    retrieve_relevant_vars_func: Callable[[UserContext], dict] = lambda _: dict()

    def determine_next_step(self, trigger: str, context: UserContext) -> StepID:
        next_step_decider = (
            self.next_step_decider
                if isinstance(self.next_step_decider, NextStepDecider)
                else
            self.next_step_decider[trigger])

        return next_step_decider.determine_next_step(context)


    def get_generate_chatbot_messages_fns_for_trigger(self, trigger: str) -> list[GenerateMessageFunc]:
        fns = self.generate_chatbot_messages_fns
        return fns if isinstance(fns, list) else fns[trigger]
