from abc import ABC

from gradio.events import EventListenerMethod
import gradio as gr

from constants import UserInteractionType


class ComponentWrapper(ABC):
    trigger_index = 0
    
    def __init__(
        self,
        user_interaction_type: UserInteractionType,
        component: gr.Button | gr.Textbox | gr.Files | gr.UploadButton,
        trigger_to_proceed: EventListenerMethod,
        first_actions_after_trigger: list[dict] = []
    ):
        self._user_interaction_type = user_interaction_type
        self._component = component
        self._trigger_to_proceed = trigger_to_proceed
        self._first_actions_after_trigger = first_actions_after_trigger
    
    @property
    def user_interaction_type(self):
        return self._user_interaction_type

    @property
    def component(self):
        return self._component

    def get_trigger_to_proceed(self):
        def print_trigger_index():
            ComponentWrapper.trigger_index += 1
            component_name = self._component.label or self._component.value
            component_type = type(self._component).__name__
            print(f'-- {ComponentWrapper.trigger_index} -- Triggered \'{component_name}\' {component_type}\n')

        trigger = self._trigger_to_proceed(print_trigger_index)

        for action in self._first_actions_after_trigger:
            trigger = trigger.then(**action)

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
            trigger_to_proceed = getattr(start_btn, 'click'),
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
            trigger_to_proceed=getattr(yes_no_btn, 'click'),
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
            trigger_to_proceed=getattr(files, 'clear'),
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
            trigger_to_proceed=getattr(upload_btn, 'upload'),
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
            trigger_to_proceed=getattr(submit_btn, 'click'),
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
            trigger_to_proceed=getattr(clear_btn, 'click'),
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
            trigger_to_proceed=getattr(text_box, 'submit'),
            **kwargs)
