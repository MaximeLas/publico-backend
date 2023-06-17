from enum import Enum, auto
from strenum import StrEnum
from typing import Callable

import gradio as gr

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

from helpers import *


def generate_answer_to_question(vars: dict) -> str:
    documents_chunks = get_documents_chunks_for_txt_files(vars[OutputKeys.PRIOR_GRANT_APPLICATIONS])
    vector_store = Chroma.from_documents(documents=documents_chunks, embedding=OpenAIEmbeddings(client=None))
    
    question_for_prompt = vars[OutputKeys.APPLICATION_QUESTION]
    if vars[OutputKeys.WORD_COUNT] and vars[OutputKeys.WORD_COUNT].isdigit():
        question_for_prompt += f' (Word Count: {vars[OutputKeys.WORD_COUNT]})'

    most_relevant_documents = get_most_relevant_docs_in_vector_store_for_answering_question(
        vector_store=vector_store, question=question_for_prompt, n_results=1)

    chatopenai = ChatOpenAI(client=None, model_name='gpt-4', temperature=0.1)

    final_answer = generate_answers_from_documents_for_question(most_relevant_documents, chatopenai, vars[OutputKeys.APPLICATION_QUESTION])[0]

    print(f'Final answer for application question "{vars[OutputKeys.APPLICATION_QUESTION]}":\n{final_answer}')
    return final_answer


class UserInteractionType(Enum):
    YES_NO = auto()
    FILES = auto()
    TEXT = auto()
    START = auto()
    NONE = auto()

class OutputKeys(Enum):
    HAS_APPLIED_FOR_THIS_GRANT_BEFORE = auto()
    PRIOR_GRANT_APPLICATIONS = auto()
    APPLICATION_QUESTION = auto()
    WORD_COUNT = auto()

class ChatbotStep(ABC):
    def __init__(
        self,
        user_interaction_type: UserInteractionType,
        message: str,
        output_key: OutputKeys,
        generate_output_fn: Callable[[dict], str] | None = None
    ):
        self._user_interaction_type = user_interaction_type
        self._message = message
        self._output_key = output_key
        self._generate_output_fn = generate_output_fn
    
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
    def generate_output_fn(self):
        return self._generate_output_fn

class FilesStep(ChatbotStep):
    def __init__(
        self,
        message: str,
        output_key: OutputKeys,
        kind_of_document: str
    ):
        super().__init__(UserInteractionType.FILES, message, output_key)
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
        message: str,
        output_key: OutputKeys,
        generate_output_fn: Callable[[dict], str] | None = None
    ):
        super().__init__(UserInteractionType.TEXT, message, output_key, generate_output_fn)

class StartStep(ChatbotStep):
    def __init__(
        self,
        message: str,
        output_key: OutputKeys
    ):
        super().__init__(UserInteractionType.START, message, output_key)

class YesNoStep(ChatbotStep):
    def __init__(
        self,
        message: str,
        output_key: OutputKeys
    ):
        super().__init__(UserInteractionType.YES_NO, message, output_key)

    
class ComponentWrapper:
    trigger_index = 0
    
    def __init__(
        self,
        component: gr.Button | gr.Textbox | gr.Files,
        trigger_function_name: str,
        user_interaction_type: UserInteractionType,
        first_actions_after_trigger: list[dict] = []
    ):
        self.component = component
        self.trigger_to_proceed = getattr(component, trigger_function_name)
        self.user_interaction_type = user_interaction_type
        self.first_actions_after_trigger = first_actions_after_trigger
        
    def set_first_actions_after_trigger(self, first_actions_after_trigger):
        self.first_actions_after_trigger = first_actions_after_trigger

    def get_trigger_to_proceed(self):
        def print_trigger_index():
            ComponentWrapper.trigger_index += 1
            print(f'\n-- {ComponentWrapper.trigger_index} -- Triggered {type(self.component).__name__}')

        trigger = self.trigger_to_proceed(print_trigger_index)

        for action in self.first_actions_after_trigger:
            trigger = trigger.then(**action)

        return trigger


chatbot_steps = [
    YesNoStep( # 0
        message="Have you applied for this grant before?",
        output_key=OutputKeys.HAS_APPLIED_FOR_THIS_GRANT_BEFORE),
    FilesStep( # 1
        message="That's very useful! Please upload your {kind_of_document}.",
        output_key=OutputKeys.PRIOR_GRANT_APPLICATIONS,
        kind_of_document="prior grant application(s)"),
    TextStep( # 2
        message="Now, on to the first question! Please let me know what the first application question is, or copy and paste it from the application portal.",
        output_key=OutputKeys.APPLICATION_QUESTION),
    TextStep( # 3
        message="What is the word count?",
        output_key=OutputKeys.WORD_COUNT,
        generate_output_fn=generate_answer_to_question)]


OUTPUT_VARIABLES: dict = {}
def handle_user_interaction(user_message, chat_history, step: int):
    if chatbot_steps[step].output_key:
        OUTPUT_VARIABLES[chatbot_steps[step].output_key] = user_message
        print(f'{chatbot_steps[step].output_key}: {OUTPUT_VARIABLES[chatbot_steps[step].output_key]}')
    
    new_chat_history = chat_history[:-1] + [[chat_history[-1][0], user_message]]

    gen_output_fn = chatbot_steps[step].generate_output_fn
    if gen_output_fn is not None:
        generated_output = gen_output_fn(OUTPUT_VARIABLES)
        new_chat_history += [[generated_output, None]]
    
    return '', new_chat_history


def handle_files_uploaded(files: list, step: int, chat_history: list[list]):
    # iterate over files and print their names
    for file in files: print(f'File uploaded: {file.name.split("/")[-1]}')

    files_step = chatbot_steps[step]

    output_var_for_files = files_step.output_key
    OUTPUT_VARIABLES[output_var_for_files] = [file.name for file in files]
    print(f'{output_var_for_files}: {OUTPUT_VARIABLES[output_var_for_files]}')

    validation_message = f'You successfully uploaded {len(files)} {files_step.kind_of_document}! 🎉'

    return chat_history + [[validation_message, None]]
    

def is_visible_in_current_user_interaction(component_wrapper: ComponentWrapper, step: int):
    if 0 <= step < len(chatbot_steps):
        return component_wrapper.user_interaction_type is chatbot_steps[step].user_interaction_type
    else:
        return False


with gr.Blocks() as demo:
    step_var = gr.State(-1)
    chatbot = gr.Chatbot()
    
    with gr.Row():
        start_btn = ComponentWrapper(
            component=gr.Button("Start", visible=True),
            trigger_function_name='click',
            user_interaction_type=UserInteractionType.START)
        
        yes_btn = ComponentWrapper(
            component=gr.Button("Yes", visible=False),
            trigger_function_name='click',
            user_interaction_type=UserInteractionType.YES_NO)
        
        no_btn = ComponentWrapper(
            component=gr.Button("No", visible=False),
            trigger_function_name='click',
            user_interaction_type=UserInteractionType.YES_NO)
        
        user_text_box = ComponentWrapper(
            component=gr.Textbox(label="User", lines=3, visible=False, interactive=True, placeholder="Type your message here"),
            trigger_function_name='submit',
            user_interaction_type=UserInteractionType.TEXT)
        user_text_box.set_first_actions_after_trigger([{
            'fn': handle_user_interaction,
            'inputs': [user_text_box.component, chatbot, step_var],
            'outputs': [user_text_box.component, chatbot]
        }])
         
        files = ComponentWrapper(
            component=gr.Files(label='Documents', visible=False, interactive=True),
            trigger_function_name='change',
            user_interaction_type=UserInteractionType.FILES)
        files.set_first_actions_after_trigger([{
            'fn': handle_files_uploaded,
            'inputs': [files.component, step_var, chatbot],
            'outputs': chatbot
        }])

    components = [start_btn, yes_btn, no_btn, files, user_text_box]

    for component in components:
        component.get_trigger_to_proceed().then(
            # increment step
            fn=lambda step: step + 1,
            inputs=step_var,
            outputs=step_var
        ).then(
            # stream chatbot message to chatbot component
            fn=lambda chat_history, step: chat_history + [[chatbot_steps[step].message, None]] if step > -1 and step < len(chatbot_steps) else chat_history,
            inputs=[chatbot, step_var],
            outputs= chatbot
        ).then(
            # update visibility of components based on current chatbot step user interaction type
            fn=lambda step: [gr.update(visible=is_visible_in_current_user_interaction(component, step)) for component in components],
            inputs=step_var,
            outputs=[component.component for component in components]
        )

demo.launch()