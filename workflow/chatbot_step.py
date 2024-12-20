from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from configurations.constants import ComponentID, StepID
from workflow.app_context import AppContext
from workflow.step_decider import StepDecider



MessageOutputType = str | list[str] | Iterator[str | list[str] | None] | None
GenerateMessageFunc = Callable[[AppContext], MessageOutputType]
GenerateMessageFuncList = list[GenerateMessageFunc]
ComponentPropertiesType = dict[str, str]



@dataclass
class InitialChatbotMessage:
    message: str
    extract_formatting_variables_func: Callable[['AppContext'], Iterator] = lambda _: (yield dict())

    def get_formatted_message(self, context: AppContext) -> Iterator[str]:
        for response_so_far in self.extract_formatting_variables_func(context):
            if isinstance(response_so_far, dict):
                yield self.message.format(**response_so_far)
            else:
                yield self.message.format(response=response_so_far)


@dataclass
class ChatbotStep():
    initial_chatbot_message: InitialChatbotMessage
    next_step_decider: StepDecider | dict[str, StepDecider]
    components: dict[ComponentID, ComponentPropertiesType | Callable[['AppContext'], ComponentPropertiesType]] = field(default_factory=dict)
    initialize_step_func: Callable[[AppContext], None] = lambda _: None
    save_event_outcome_fn: Callable[[AppContext, Any], None] | None = None
    generate_chatbot_messages_fns: GenerateMessageFuncList | dict[str, GenerateMessageFuncList] = field(default_factory=lambda: defaultdict(list))
    retrieve_relevant_vars_func: Callable[[AppContext], dict] = lambda _: dict()

    def determine_next_step(self, trigger: str, context: AppContext) -> StepID:
        next_step_decider = (
            self.next_step_decider
                if isinstance(self.next_step_decider, StepDecider)
                else
            self.next_step_decider[trigger])

        return next_step_decider.determine_next_step(context)


    def get_generate_chatbot_messages_fns_for_trigger(self, trigger: str) -> list[GenerateMessageFunc]:
        fns = self.generate_chatbot_messages_fns
        return fns if isinstance(fns, list) else fns.get(trigger, [])
