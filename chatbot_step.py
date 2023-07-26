from abc import ABC
from collections.abc import Callable, Iterator
from types import GeneratorType

from constants import UserInteractionType, ContextKeys


MessageGenerationType = str | Iterator[str] | list[str] | Iterator[list[str]]
GenerateMessageFnType = Callable[[dict], MessageGenerationType]



class ChatbotStep(ABC):
    def __init__(
        self,
        user_interaction_types: list[UserInteractionType],
        initial_message: str,
        context_key: ContextKeys,
        generate_message_fns: list[GenerateMessageFnType] = []
    ):
        self._user_interaction_types = user_interaction_types
        self._initial_message = initial_message
        self._output_key = context_key
        self._generate_message_fns = generate_message_fns
    
    @property
    def user_interaction_types(self) -> list[UserInteractionType]:
        return self._user_interaction_types

    @property
    def initial_message(self) -> str:
        return self._initial_message
    
    @property
    def context_key(self) -> ContextKeys:
        return self._output_key

    @property
    def generate_message_fns(self) -> list[GenerateMessageFnType]:
        return self._generate_message_fns


    def generate_chatbot_message(self, chatbot_history, context):
        new_chatbot_messages = []
        for fn in self.generate_message_fns:
            if (response := fn(context)) is not None:
                current_new_messages = []
                if type(response) is GeneratorType:
                    for streamed_so_far in response:
                        current_new_messages = (
                            [[streamed_so_far, None]]
                                if type(streamed_so_far) is str else
                            [[message, None] for message in streamed_so_far])
                        yield chatbot_history + new_chatbot_messages + current_new_messages
                else:
                    current_new_messages = (
                        [[response, None]]
                            if type(response) is str else
                        [[message, None] for message in response])
                    yield chatbot_history + new_chatbot_messages + current_new_messages
                new_chatbot_messages += current_new_messages

        yield chatbot_history + new_chatbot_messages


class FilesStep(ChatbotStep):
    def __init__(
        self,
        kind_of_document: str,
        **kwargs
    ):
        super().__init__(
            user_interaction_types=[
                UserInteractionType.FILES,
                UserInteractionType.UPLOAD,
                UserInteractionType.SUBMIT,
                UserInteractionType.CLEAR
            ],
            **kwargs)

        self._kind_of_document = kind_of_document
    
    @property
    def initial_message(self):
        return self._initial_message.format(kind_of_document=self._kind_of_document)
    
    @property
    def kind_of_document(self):
        return self._kind_of_document

class TextStep(ChatbotStep):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            user_interaction_types=[UserInteractionType.TEXT, UserInteractionType.SUBMIT_TEXT],
            **kwargs)

class StartStep(ChatbotStep):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(user_interaction_types=[UserInteractionType.START], **kwargs)

class YesNoStep(ChatbotStep):
    def __init__(
        self,
        next_step_if_yes: Callable[[int], int] = lambda step: step + 1,
        next_step_if_no: Callable[[int], int] = lambda step: step + 1,
        **kwargs
    ):
        super().__init__(user_interaction_types=[UserInteractionType.YES_NO], **kwargs)
        self._next_step_if_yes = next_step_if_yes
        self._next_step_if_no = next_step_if_no

    def go_to_step(self, yes_or_no: str, step: int) -> int:
        return self._next_step_if_yes(step) if yes_or_no == 'Yes' else self._next_step_if_no(step)
