from collections.abc import Iterator
from types import GeneratorType
import tempfile

import gradio as gr

from settings import CHATBOT_STEPS, CONTEXT
from constants import ContextKeys
from chatbot_step import TextStep, YesNoStep, FilesStep, GenerateMessageFnType



def save_to_context(context_key: ContextKeys, value: str | list[str]):
    CONTEXT[context_key] = value
    print(f'{context_key}: {CONTEXT[context_key]}\n')


def generate_chatbot_messages(generate_message_fns: list[GenerateMessageFnType]) -> Iterator[list[list[str]]]:
    new_chatbot_messages = []
    for fn in generate_message_fns:
        if (response := fn(CONTEXT)) is not None:
            current_new_messages = []
            if type(response) is GeneratorType:
                for streamed_so_far in response:
                    current_new_messages = (
                        [[streamed_so_far, None]]
                            if type(streamed_so_far) is str else
                        [[message, None] for message in streamed_so_far])
                    yield new_chatbot_messages + current_new_messages
            else:
                current_new_messages = (
                    [[response, None]]
                        if type(response) is str else
                    [[message, None] for message in response])
                yield new_chatbot_messages + current_new_messages
            new_chatbot_messages += current_new_messages
    
    print('Done with generate_chatbot_messages\n')


def handle_text_submitted(
    user_message: str,
    chat_history: list[list],
    step: int
):
    text_step = CHATBOT_STEPS[step]
    assert type(text_step) is TextStep

    save_to_context(text_step.context_key, user_message)

    # update chat history with user message
    chat_history[-1][1] = user_message

    if (fns := text_step.generate_message_fns) != []:
        # generate new message(s) and update chat history accordingly
        for messages in generate_chatbot_messages(fns):
            yield '', chat_history + messages
    else:
        yield '', chat_history


def handle_yes_no_clicked(
    yes_or_no: str,
    chat_history: list[list],
    step: int
):
    yes_no_step = CHATBOT_STEPS[step]
    assert type(yes_no_step) is YesNoStep

    # save YES or NO to context variable
    save_to_context(yes_no_step.context_key, yes_or_no)

    steps_to_skip = yes_no_step.steps_to_skip(yes_or_no)
    
    if (fns := yes_no_step.generate_message_fns) != []:
        # generate new message(s) and update chat history accordingly
        for new_messages in generate_chatbot_messages(fns):
            yield chat_history + new_messages, step + steps_to_skip
    else:
        yield chat_history, step + steps_to_skip


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
    chat_history: list[list],
    step: int
):
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
    chat_history += (
        [[CHATBOT_STEPS[step + 1].message, None]]
            if step + 1 < len(CHATBOT_STEPS) else
        [['End of demo, thanks for participating!', None]])

    return chat_history, step + 1
