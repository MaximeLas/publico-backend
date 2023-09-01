from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import Callable, NotRequired, Sequence, TypedDict

from gradio.events import EventListenerMethod, Dependency
from gradio.components import IOComponent
import gradio as gr

from constants import ComponentLabel, StepID


class EventParameters(TypedDict):
    fn: Callable
    inputs: NotRequired[Sequence[IOComponent]]
    outputs: NotRequired[Sequence[IOComponent]]



@dataclass
class ComponentWrapper(ABC):
    component: IOComponent
    name: str = field(init=False)
    user_action: str = field(init=False)
    handle_user_action: EventParameters
    proceed_to_next_step: bool = True
    trigger_index = 0

    def __post_init__(self):
        self.name = self.component.label or self.component.value


    @staticmethod
    def print_trigger_info(component_name: str, current_step_id: StepID) -> None:
        if component_name == ComponentLabel.START:
            # setting/resetting to 1
            ComponentWrapper.trigger_index = 1
        else:
            ComponentWrapper.trigger_index += 1

        now = datetime.fromtimestamp(time.time(), tz=None).strftime("%d-%m-%Y %H:%M:%S")
        print(f"\n-- {now} -- {ComponentWrapper.trigger_index} -- Triggered '{component_name}' -- Step '{current_step_id}'\n")


    def get_component_trigger(self) -> EventListenerMethod:
        return getattr(self.component, self.user_action)


    def chain_handle_user_action(self,  trigger: Dependency) -> Dependency:
        if self.handle_user_action:
            trigger = trigger.then(
                **self.handle_user_action
            ).then(
                fn=lambda: time.sleep(0.25)
            )

        return trigger




@dataclass
class ButtonWrapper(ComponentWrapper):
    component: gr.Button
    user_action = 'click'



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
