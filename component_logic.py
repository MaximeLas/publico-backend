from typing import Callable
import tempfile

import gradio as gr

from settings import CHATBOT_STEPS, CONTEXT
from constants import ContextKeys
from chatbot_step import TextStep, YesNoStep, FilesStep


def generate_chatbot_messages(generate_output_fns: list[Callable[[dict], str | list[str] | None]]) -> list[list[str]]:
    new_chatbot_messages = []
    for fn in generate_output_fns:
        if (response := fn(CONTEXT)) is not None:
            if type(response) == list:
                for chatbot_response in response:
                    new_chatbot_messages += [[chatbot_response, None]]
            else:
                new_chatbot_messages += [[response, None]]

    return new_chatbot_messages

def save_to_context(context_key: ContextKeys, value):
    CONTEXT[context_key] = value
    print(f'{context_key}: {CONTEXT[context_key]}\n')



def handle_text_submitted(user_message, chat_history: list[list], step: int):
    text_step = CHATBOT_STEPS[step]
    assert type(text_step) is TextStep

    # save user message to context variable
    save_to_context(text_step.context_key, user_message)
    
    # update chat history with user message
    new_chat_history = chat_history[:-1] + [[chat_history[-1][0], user_message]]

    # generate output if necessary and update chat history with it
    if (fns := text_step.generate_output_fns) is not None:
        new_chat_history += generate_chatbot_messages(fns)
    
    return '', new_chat_history


def handle_yes_no_clicked(yes_or_no: str, chat_history: list[list], step: int):
    yes_no_step = CHATBOT_STEPS[step]
    assert type(yes_no_step) is YesNoStep

    # save YES or NO to context variable
    save_to_context(yes_no_step.context_key, yes_or_no)

    # generate output if necessary and update chat history with it
    if (fns := yes_no_step.generate_output_fns) is not None:
        chat_history += generate_chatbot_messages(fns)
    
    return chat_history, step + yes_no_step.steps_to_skip(yes_or_no)


def handle_files_uploaded(files_uploaded: list[tempfile._TemporaryFileWrapper], files_present: list[tempfile._TemporaryFileWrapper] | None):
    new_files = []

    if files_present is not None:
        for file in files_present:
            new_files.append(file.name)

    for file in files_uploaded:
        if file.name not in new_files:
            new_files.append(file.name)
        else:
            print(f'{file.name.rsplit("/", 1)[-1]} already present in uploaded files')

    print(f'Total files uploaded so far: {len(new_files)}\n')

    return new_files, gr.update(interactive=True), gr.update(interactive=True)


def handle_files_submitted(files: list, chat_history: list[list], step: int):
    files_step = CHATBOT_STEPS[step]
    assert type(files_step) is FilesStep

    # save file names to context variable
    save_to_context(files_step.context_key, [file.name for file in files])

    # iterate over files and print their names
    for file in files: print(f'File uploaded: {file.name.split("/")[-1]}')
    print()

    # update chat history with validation message
    validation_message = f'You successfully uploaded {len(files)} {files_step.kind_of_document}! 🎉'

    return chat_history + [[validation_message, None]]


def stream_next_step_chatbot_message(chat_history: list[list], step: int):
    step += 1
    if 0 <= step < len(CHATBOT_STEPS):
        chat_history += [[CHATBOT_STEPS[step].message, None]]
    else:
        chat_history += [['End of demo, thanks for participating!', None]]

    return chat_history, step
