import tempfile
from devtools import debug

import gradio as gr

from chatbot_workflow import WorkflowState
from constants import DEFAULT_NUMBER, ComponentLabel



def handle_btn_clicked(
    btn_label: str,
    chat_history: list[list],
    workflow_state: WorkflowState
):
    chat_history[-1][1] = f'**{btn_label}**'

    text_box = workflow_state.context.get_answer_of_current_implicit_question_to_be_answered() if btn_label == ComponentLabel.EDIT_IT else gr.skip()

    return chat_history, text_box


def handle_submit(
    user_message: str,
    number: str,
    chat_history: list[list]
):
    if user_message != '':
        # update chat history with user message
        chat_history[-1][1] = f'**{user_message}**'
        return None, gr.skip(), chat_history
    else:
        # update chat history with number submitted
        chat_history[-1][1] = f'**{str(number)}**'
        return gr.skip(), DEFAULT_NUMBER, chat_history


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

    debug(**{'Total files uploaded so far': len(all_files)})

    return all_files, gr.update(interactive=True), gr.update(interactive=True)


def handle_files_submitted(
    files: list[tempfile._TemporaryFileWrapper]
):
    # print file names
    debug(**{f'File #{i+1} uploaded': file.name.split("/")[-1] for i, file in enumerate(files)})
