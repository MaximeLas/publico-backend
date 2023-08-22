from abc import ABC
from dataclasses import dataclass, field
import time
from typing import Callable, NotRequired, Sequence, TypedDict

from gradio.events import EventListenerMethod, Dependency
from gradio.components import IOComponent
import gradio as gr


class EventParameters(TypedDict):
    fn: Callable
    inputs: NotRequired[IOComponent | Sequence[IOComponent]]
    outputs: NotRequired[IOComponent | Sequence[IOComponent]]



@dataclass
class ComponentWrapper(ABC):
    component: IOComponent
    name: str = field(init=False)
    user_action: str | None = field(init=False)
    handle_user_action: EventParameters | None = None
    proceed_to_next_step: bool = True
    trigger_index = 0

    def __post_init__(self):
        self.name = self.component.label or self.component.value


    def get_initial_chain_following_trigger(self) -> Dependency:
        def print_trigger_index() -> None:
            if self.name == 'Start':
                # setting/resetting to 1
                ComponentWrapper.trigger_index = 1
            else:
                ComponentWrapper.trigger_index += 1

            print(f'\n-- {ComponentWrapper.trigger_index} -- Triggered \'{self.name}\' {type(self.component).__name__}\n')

        assert self.user_action is not None, f'Cannot chain first actions after trigger for {self.component} as user_action is None'

        event_listener_method: EventListenerMethod = getattr(self.component, self.user_action)
        return event_listener_method(print_trigger_index)


    def chain_handle_user_action(self,  trigger: Dependency) -> Dependency:
        if self.handle_user_action:
            trigger = trigger.then(
                **self.handle_user_action
            ).then(
                fn=lambda: time.sleep(0.5)
            )

        return trigger




@dataclass
class ButtonWrapper(ComponentWrapper):
    component: gr.Button
    user_action = 'click'



@dataclass
class FilesWrapper(ComponentWrapper):
    component: gr.Files
    user_action = None

    def __post_init__(self):
        super().__post_init__()
        self.proceed_to_next_step = False



@dataclass
class UploadButtonWrapper(ComponentWrapper):
    component: gr.UploadButton
    user_action = 'upload'

    def __post_init__(self):
        super().__post_init__()
        self.proceed_to_next_step = False



@dataclass
class ClearButtonWrapper(ComponentWrapper):
    component: gr.ClearButton
    user_action = 'click'

    def __post_init__(self):
        super().__post_init__()
        self.proceed_to_next_step = False



@dataclass
class TextboxWrapper(ComponentWrapper):
    component: gr.Textbox
    user_action = 'submit'



@dataclass
class NumberWrapper(ComponentWrapper):
    component: gr.Number
    user_action = 'submit'
