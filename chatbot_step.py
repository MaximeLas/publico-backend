from abc import ABC
from collections.abc import Iterator
from typing import Callable

from constants import UserInteractionType, ContextKeys

MessageGenerationType = str | Iterator[str] | list[str] | Iterator[list[str]]
GenerateMessageFnType = Callable[[dict], MessageGenerationType]

class ChatbotStep(ABC):
    def __init__(
        self,
        user_interaction_types: list[UserInteractionType],
        message: str,
        context_key: ContextKeys,
        generate_message_fns: list[GenerateMessageFnType] = []
    ):
        self._user_interaction_types = user_interaction_types
        self._message = message
        self._output_key = context_key
        self._generate_message_fns = generate_message_fns
    
    @property
    def user_interaction_types(self) -> list[UserInteractionType]:
        return self._user_interaction_types

    @property
    def message(self) -> str:
        return self._message
    
    @property
    def context_key(self) -> ContextKeys:
        return self._output_key

    @property
    def generate_message_fns(self) -> list[GenerateMessageFnType]:
        return self._generate_message_fns

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
    def message(self):
        return self._message.format(kind_of_document=self._kind_of_document)
    
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
        steps_to_skip_if_yes: int = 0,
        steps_to_skip_if_no: int = 0,
        **kwargs
    ):
        super().__init__(user_interaction_types=[UserInteractionType.YES_NO], **kwargs)
        self._steps_to_skip_if_yes = steps_to_skip_if_yes
        self._steps_to_skip_if_no = steps_to_skip_if_no

    def steps_to_skip(self, yes_or_no: str) -> int:
        return self._steps_to_skip_if_yes if yes_or_no == 'Yes' else self._steps_to_skip_if_no
