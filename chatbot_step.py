from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from gradio.blocks import Block

from constants import StepID
from context import UserContext


MessageGenerationType = str | Iterator[str] | list[str] | Iterator[list[str]]
GenerateMessageFnType = Callable[[UserContext], MessageGenerationType]
GenerateMessageFnsType = list[GenerateMessageFnType] | defaultdict[str, list[GenerateMessageFnType]]


@dataclass
class ChatbotStep():
    initial_message: str
    next_step: StepID | dict[str, StepID]
    components: list[Block] = field(default_factory=list)
    store_in_context_fn: tuple[Callable[[UserContext, Any], UserContext], str | None] | None = None
    generate_chatbot_messages_fns: GenerateMessageFnsType = field(default_factory=list)


    def determine_next_step(self, trigger: str) -> StepID:
        return self.next_step if isinstance(self.next_step, StepID) else self.next_step[trigger]


    def determine_generate_chatbot_messages_fns(self, trigger: str) -> list[GenerateMessageFnType]:
        fns = self.generate_chatbot_messages_fns
        return fns if isinstance(fns, list) else fns[trigger]
