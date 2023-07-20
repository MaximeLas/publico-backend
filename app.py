from abc import ABC
from enum import Enum, auto
from typing import Callable

import gradio as gr
from gradio.events import EventListenerMethod

from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()

from constants import *
from helpers import *
from prompts import *
from comprehensiveness import *

# check if sqlite3 version is too old and install newer version if necessary in order to avoid error:
# Error: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
import sqlite3
if sqlite3.sqlite_version_info < (3, 35, 0):
    print(f'sqlite3 version is too old version={sqlite3.sqlite_version_info}')
    print(f'installing sqlite3 version >= 3.35.0\n')

    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])

    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

    print(f'\ninstalled sqlite3 version={sys.modules["sqlite3"].sqlite_version_info}\n')

from pip._internal.operations import freeze
pkgs = freeze.freeze()
for pkg in pkgs: print(pkg)

class ChatbotStep(ABC):
    def __init__(
        self,
        user_interaction_type: UserInteractionType,
        message: str,
        output_key: OutputKeys,
        generate_output_fns: list[Callable[[dict], str]] = []
    ):
        self._user_interaction_type = user_interaction_type
        self._message = message
        self._output_key = output_key
        self._generate_output_fns = generate_output_fns
    
    @property
    def user_interaction_type(self):
        return self._user_interaction_type

    @property
    def message(self):
        return self._message
    
    @property
    def output_key(self):
        return self._output_key

    @property
    def generate_output_fns(self):
        return self._generate_output_fns

class FilesStep(ChatbotStep):
    def __init__(
        self,
        kind_of_document: str,
        **kwargs
    ):
        super().__init__(user_interaction_type=UserInteractionType.FILES, **kwargs)
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
        super().__init__(user_interaction_type=UserInteractionType.TEXT, **kwargs)

class StartStep(ChatbotStep):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(user_interaction_type=UserInteractionType.START, **kwargs)

class YesNoStep(ChatbotStep):
    def __init__(
        self,
        steps_to_skip_if_yes: int = 0,
        steps_to_skip_if_no: int = 0,
        **kwargs
    ):
        super().__init__(user_interaction_type=UserInteractionType.YES_NO, **kwargs)
        self._steps_to_skip_if_yes = steps_to_skip_if_yes
        self._steps_to_skip_if_no = steps_to_skip_if_no

    def steps_to_skip(self, yes_or_no: str) -> int:
        return self._steps_to_skip_if_yes if yes_or_no == 'Yes' else self._steps_to_skip_if_no
        

class ComponentWrapper(ABC):
    trigger_index = 0
    
    def __init__(
        self,
        user_interaction_type: UserInteractionType,
        component: gr.Button | gr.Textbox | gr.Files,
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
            trigger_to_proceed = getattr(yes_no_btn, 'click'),
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
            trigger_to_proceed = getattr(files, 'change'),
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
            trigger_to_proceed = getattr(text_box, 'submit'),
            **kwargs)


# define dict to store output variables from chatbot steps
OUTPUT_VARIABLES = {}

# define chatbot steps and their properties
CHATBOT_STEPS = [
    YesNoStep( # 0
        message="Have you applied for this grant before?",
        output_key=OutputKeys.HAS_APPLIED_FOR_THIS_GRANT_BEFORE,
        steps_to_skip_if_no=1),
    FilesStep( # 1
        message="That's very useful! Please upload your {kind_of_document}.",
        output_key=OutputKeys.PRIOR_GRANT_APPLICATIONS,
        kind_of_document="prior grant application(s)"),
    TextStep( # 2
        message="Now, on to the first question! Please let me know what the first application question is, or copy and paste it from the application portal.",
        output_key=OutputKeys.APPLICATION_QUESTION),
    TextStep( # 3
        message="What is the word limit?",
        output_key=OutputKeys.WORD_LIMIT,
        generate_output_fns=[generate_answer_to_question]),
    YesNoStep( # 4
        message="Do you want to check the comprehensiveness of the generated answer?",
        output_key=OutputKeys.CHECK_COMPREHENSIVENESS,
        generate_output_fns=[check_for_comprehensiveness, generate_answers_for_implicit_questions, generate_final_answer])]


def save_to_output_variable(output_key: OutputKeys, value):
    OUTPUT_VARIABLES[output_key] = value
    print(f'{output_key}: {OUTPUT_VARIABLES[output_key]}\n')


def handle_user_interaction(user_message, chat_history, step: int):
    text_step: TextStep = CHATBOT_STEPS[step]

    # save user message to output variable
    save_to_output_variable(text_step.output_key, user_message)
    
    # update chat history with user message
    new_chat_history = chat_history[:-1] + [[chat_history[-1][0], user_message]]

    # generate output if necessary and update chat history with it
    if (fns := text_step.generate_output_fns) is not None:
        for fn in fns:
            if (response := fn(OUTPUT_VARIABLES)) is not None:
                if type(response) == list:
                    for chatbot_response in response:
                        new_chat_history += [[chatbot_response, None]]
                else:
                    new_chat_history += [[response, None]]
    
    return '', new_chat_history


def handle_yes_no_interaction(yes_or_no: str, chat_history, step: int):
    yes_no_step: YesNoStep = CHATBOT_STEPS[step]

    # save YES or NO to output variable
    save_to_output_variable(yes_no_step.output_key, yes_or_no)

    # generate output if necessary and update chat history with it
    if (fns := yes_no_step.generate_output_fns) is not None:
        for fn in fns:
            if (response := fn(OUTPUT_VARIABLES)) is not None:
                if type(response) == list:
                    for chatbot_response in response:
                        chat_history += [[chatbot_response, None]]
                else:
                    chat_history += [[response, None]]
    
    return chat_history, step + yes_no_step.steps_to_skip(yes_or_no)


def handle_files_uploaded(files: list, step: int, chat_history: list[list]):
    files_step = CHATBOT_STEPS[step]

    # save file names to output variable
    save_to_output_variable(files_step.output_key, [file.name for file in files])

    # iterate over files and print their names
    for file in files: print(f'File uploaded: {file.name.split("/")[-1]}')
    print()

    # update chat history with validation message
    validation_message = f'You successfully uploaded {len(files)} {files_step.kind_of_document}! 🎉'

    return chat_history + [[validation_message, None]]


def is_visible_in_current_user_interaction(component_wrapper: ComponentWrapper, step: int):
    # if step is out of bounds, return False to hide component
    if not 0 <= step < len(CHATBOT_STEPS):
        return False

    # if step is in bounds, return True if component's user interaction type matches current step's user interaction type
    return component_wrapper.user_interaction_type is CHATBOT_STEPS[step].user_interaction_type

CSS ="""
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; }
"""

with gr.Blocks() as demo:
    # create state variable to keep track of current chatbot step
    step_var = gr.State(-1)

    # create chatbot component
    chatbot = gr.Chatbot(elem_id="chatbot", height=800)
    
    with gr.Row():
        # create component wrappers for each component in the chatbot UI
        
        # start button component
        start_btn = StartWrapper(start_btn=gr.Button("Start", visible=True))

        # yes/no button component
        yes_no_btns = []
        for value in ['Yes', 'No']:
            btn_component = gr.Button(value, visible=False)
            yes_no_btns.append(YesNoWrapper(
                yes_no_btn=btn_component,
                first_actions_after_trigger=[{
                    'fn': handle_yes_no_interaction,
                    'inputs': [btn_component, chatbot, step_var],
                    'outputs': [chatbot, step_var]
                }]))

        # user text box component
        text_box_component = gr.Textbox(label="User", lines=3, visible=False, interactive=True, placeholder="Type your message here")
        user_text_box = TextWrapper(
            text_box=text_box_component,
            first_actions_after_trigger=[{
                'fn': handle_user_interaction,
                'inputs': [text_box_component, chatbot, step_var],
                'outputs': [text_box_component, chatbot]
            }])
        
        # files component
        files_component = gr.Files(label='Documents', visible=False, interactive=True)
        files = FilesWrapper(
            files=files_component,
            first_actions_after_trigger=[{
                'fn': handle_files_uploaded,
                'inputs': [files_component, step_var, chatbot],
                'outputs': chatbot
            }])

    components = [start_btn, yes_no_btns[0], yes_no_btns[1], files, user_text_box]

    for component in components:
        component.get_trigger_to_proceed().then(
            # increment step
            fn=lambda step: step + 1,
            inputs=step_var,
            outputs=step_var
        ).then(
            # stream chatbot message to chatbot component if step is within chatbot steps range else stream default message
            fn=lambda chat_history, step: chat_history + ([[CHATBOT_STEPS[step].message, None]] if 0 <= step < len(CHATBOT_STEPS) else [['End of demo, thanks for participating!', None]]),
            inputs=[chatbot, step_var],
            outputs= chatbot
        ).then(
            # update visibility of components based on current chatbot step user interaction type
            fn=lambda step: [gr.update(visible=is_visible_in_current_user_interaction(component, step)) for component in components],
            inputs=step_var,
            outputs=[component.component for component in components]
        )

demo.launch()