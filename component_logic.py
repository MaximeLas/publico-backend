import tempfile

import gradio as gr

from settings import CHATBOT_STEPS
from constants import ContextKeys
from chatbot_step import TextStep, YesNoStep, FilesStep



def add_to_context(context: dict, context_key: ContextKeys, value: str | list[str]) -> dict:
    '''Add a value to the context dictionary and print the value added to the context'''
    context[context_key] = value
    print(f'{context_key}: {context[context_key]}\n')
    return context


def handle_text_submitted(
    user_message: str,
    chat_history: list[list],
    step: int,
    context: dict
):
    text_step = CHATBOT_STEPS[step]
    assert type(text_step) is TextStep

    # update chat history with user message
    chat_history[-1][1] = user_message

    return '', chat_history, add_to_context(context, text_step.context_key, user_message)


def handle_yes_no_clicked(
    yes_or_no: str,
    chat_history: list[list],
    step: int,
    context: dict
):
    yes_no_step = CHATBOT_STEPS[step]
    assert type(yes_no_step) is YesNoStep

    # update chat history with user selection of yes or no
    chat_history[-1][1] = yes_or_no

    step = yes_no_step.go_to_step(yes_or_no, step)
    
    return chat_history, step, add_to_context(context, yes_no_step.context_key, yes_or_no)


def handle_files_uploaded(
    new_files_uploaded: list[tempfile._TemporaryFileWrapper],
    current_files_uploaded: list[tempfile._TemporaryFileWrapper] | None
):
    # get all file names uploaded so far
    all_files = [file.name for file in current_files_uploaded] if current_files_uploaded is not None else []

    # add new file names to list of all file names if not already present
    for file in new_files_uploaded:
        if file.name not in all_files:
            all_files.append(file.name)
        else:
            print(f'{file.name.rsplit("/", 1)[-1]} already present in uploaded files')

    print(f'Total files uploaded so far: {len(all_files)}\n')

    return all_files, gr.update(interactive=True), gr.update(interactive=True)


def handle_files_submitted(
    files: list[tempfile._TemporaryFileWrapper],
    step: int,
    context: dict
):
    files_step = CHATBOT_STEPS[step]
    assert type(files_step) is FilesStep

    # iterate over files and print their names
    for file in files: print(f'File uploaded: {file.name.split("/")[-1]}')
    print()

    return add_to_context(context, files_step.context_key, [file.name for file in files])


def stream_next_step_chatbot_message(chat_history: list[list], step: int):
    return (
        # if there are more steps, then stream the next step's message and increment the step counter
        [chat_history + [[CHATBOT_STEPS[step + 1].message, None]], step + 1, gr.skip()]
            if step + 1 < len(CHATBOT_STEPS) else
        # if there are no more steps, then end the chat and reset the step counter and context
        [chat_history + [['End of demo, thanks for participating!', None]], -1, {}]
    )
