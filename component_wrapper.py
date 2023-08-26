from abc import ABC
from dataclasses import dataclass, field
import time
from typing import Callable, NotRequired, Sequence, TypedDict

from gradio.events import EventListenerMethod, Dependency
from gradio.components import IOComponent
import gradio as gr

from chatbot_workflow import WorkflowState


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


    @staticmethod
    def print_trigger_info(component_name: str, workflow_state: WorkflowState) -> None:
        if component_name == 'Start':
            # setting/resetting to 1
            ComponentWrapper.trigger_index = 1
        else:
            ComponentWrapper.trigger_index += 1

        print(f"\n-- {ComponentWrapper.trigger_index} -- Triggered '{component_name}' -- Step '{workflow_state.current_step_id}'\n")


    def get_component_trigger(self) -> EventListenerMethod:
        assert self.user_action is not None, f'Cannot chain first actions after trigger for {self.component} as user_action is None'

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
