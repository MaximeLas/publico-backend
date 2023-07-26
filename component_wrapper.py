from abc import ABC
import time

from gradio.events import EventListenerMethod, Dependency
from gradio.components import Component
import gradio as gr

from constants import UserInteractionType


class ComponentWrapper(ABC):
    trigger_index = 0
    
    def __init__(
        self,
        user_interaction_type: UserInteractionType,
        component: Component,
        user_action: EventListenerMethod | None = None,
        first_actions_after_trigger: list[dict] = [],
        proceed_to_next_step: bool = True
    ):
        self._user_interaction_type = user_interaction_type
        self._component = component
        self._user_action = user_action
        self._first_actions_after_trigger = first_actions_after_trigger
        self._proceed_to_next_step = proceed_to_next_step
    
    @property
    def user_interaction_type(self) -> UserInteractionType:
        return self._user_interaction_type

    @property
    def component(self) -> Component:
        return self._component

    @property
    def user_action(self) -> EventListenerMethod | None:
        return self._user_action

    @property
    def first_actions_after_trigger(self) -> list[dict]:
        return self._first_actions_after_trigger

    @property
    def proceed_to_next_step(self) -> bool:
        return self._proceed_to_next_step


    def chain_first_actions_after_trigger(self, **kwargs) -> Dependency:
        def print_trigger_index() -> None:
            ComponentWrapper.trigger_index += 1
            component_name = self._component.label or self._component.value # type: ignore
            component_type = type(self._component).__name__
            print(f'-- {ComponentWrapper.trigger_index} -- Triggered \'{component_name}\' {component_type}\n')

        assert self.user_action is not None, f'Cannot chain first actions after trigger for {self._component} as user_action is None'

        trigger = self.user_action(print_trigger_index).then(**kwargs)
        for action in self._first_actions_after_trigger:
            trigger = trigger.then(**action)
            trigger = trigger.then(fn=lambda: time.sleep(0.5))
        return trigger


class StartWrapper(ComponentWrapper):
    def __init__(
        self,
        start_btn: gr.Button,
        **kwargs
    ):
        super().__init__(
            user_interaction_type=UserInteractionType.START,
            component=start_btn,
            user_action = getattr(start_btn, 'click'),
            **kwargs)


class YesNoWrapper(ComponentWrapper):
    def __init__(
        self,
        yes_no_btn: gr.Button,
        **kwargs
    ):
        super().__init__(
            user_interaction_type=UserInteractionType.YES_NO,
            component=yes_no_btn,
            user_action=getattr(yes_no_btn, 'click'),
            **kwargs)


class FilesWrapper(ComponentWrapper):
    def __init__(
        self,
        files: gr.Files,
        **kwargs
    ):
        super().__init__(
            user_interaction_type=UserInteractionType.FILES,
            component=files,
            **kwargs)


class UploadWrapper(ComponentWrapper):
    def __init__(
        self,
        upload_btn: gr.UploadButton,
        **kwargs
    ):
        super().__init__(
            user_interaction_type=UserInteractionType.UPLOAD,
            component=upload_btn,
            user_action=getattr(upload_btn, 'upload'),
            proceed_to_next_step=False,
            **kwargs)


class SubmitWrapper(ComponentWrapper):
    def __init__(
        self,
        submit_btn: gr.Button,
        **kwargs
    ):
        super().__init__(
            user_interaction_type=UserInteractionType.SUBMIT,
            component=submit_btn,
            user_action=getattr(submit_btn, 'click'),
            **kwargs)


class ClearWrapper(ComponentWrapper):
    def __init__(
        self,
        clear_btn: gr.Button, # replace with gr.ClearButton once hf issue fixed
        **kwargs
    ):
        super().__init__(
            user_interaction_type=UserInteractionType.CLEAR,
            component=clear_btn,
            user_action=getattr(clear_btn, 'click'),
            proceed_to_next_step=False,
            **kwargs)


class TextWrapper(ComponentWrapper):
    def __init__(
        self,
        text_box: gr.Textbox,
        **kwargs
    ):
        super().__init__(
            user_interaction_type=UserInteractionType.TEXT,
            component=text_box,
            user_action=getattr(text_box, 'submit'),
            **kwargs)


class SubmitTextButtonWrapper(ComponentWrapper):
    def __init__(
        self,
        submit_text_btn: gr.Button,
        **kwargs
    ):
        super().__init__(
            user_interaction_type=UserInteractionType.SUBMIT_TEXT,
            component=submit_text_btn,
            user_action=getattr(submit_text_btn, 'click'),
            **kwargs)
