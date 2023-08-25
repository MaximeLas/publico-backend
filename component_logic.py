import tempfile
from devtools import debug

import gradio as gr
from gradio.blocks import Block

from chatbot_workflow import WorkflowState
from component_wrapper import ButtonWrapper, ClearButtonWrapper, ComponentWrapper, FilesWrapper, NumberWrapper, TextboxWrapper, UploadButtonWrapper
from constants import DEFAULT_NUMBER, ComponentID, ComponentLabel



def handle_btn_clicked(
    btn_label: str,
    chat_history: list[list],
    workflow_state: WorkflowState
):
    chat_history[-1][1] = f'**{btn_label}**'

    text_box = workflow_state.context.get_answer_of_current_implicit_question() if btn_label == ComponentLabel.EDIT_IT else gr.skip()

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


def initialize_component_wrappers(
    components: dict[ComponentID, Block],
    workflow_state: WorkflowState) -> ComponentWrapper:
    # create wrappers for each component and define the actions to be executed after being triggered, if any

    create_btn = lambda btn_component: ButtonWrapper(
        component=btn_component,
        handle_user_action={
            'fn': handle_btn_clicked,
            'inputs': [btn_component, components[ComponentID.CHATBOT], workflow_state],
            'outputs': [components[ComponentID.CHATBOT], components[ComponentID.USER_TEXT_BOX]]})
    btn1 = create_btn(components[ComponentID.BTN_1])
    btn2 = create_btn(components[ComponentID.BTN_2])

    files = FilesWrapper(component=components[ComponentID.FILES])

    upload_btn = UploadButtonWrapper(
        component=components[ComponentID.UPLOAD_FILES_BTN],
        handle_user_action={
            'fn': handle_files_uploaded,
            'inputs': [components[ComponentID.UPLOAD_FILES_BTN], components[ComponentID.FILES]],
            'outputs': [components[ComponentID.FILES], components[ComponentID.SUBMIT_FILES_BTN], components[ComponentID.CLEAR_FILES_BTN]]
        })

    submit_files_btn = ButtonWrapper(
        component=components[ComponentID.SUBMIT_FILES_BTN],
        handle_user_action={
            'fn': lambda files: debug(**{f'File #{i+1} uploaded': file.name.split("/")[-1] for i, file in enumerate(files)}),
            'inputs': components[ComponentID.FILES]
        })

    clear_btn = ClearButtonWrapper(
        component=components[ComponentID.CLEAR_FILES_BTN],
        handle_user_action={
            'fn': lambda: [gr.update(interactive=False)] * 2,
            'outputs': [components[ComponentID.SUBMIT_FILES_BTN], components[ComponentID.CLEAR_FILES_BTN]]})

    submit_btn = ButtonWrapper(
        component=components[ComponentID.SUBMIT_USER_INPUT_BTN],
        handle_user_action={
            'fn': handle_submit,
            'inputs': [components[ComponentID.USER_TEXT_BOX], components[ComponentID.NUMBER], components[ComponentID.CHATBOT]],
            'outputs': [components[ComponentID.USER_TEXT_BOX], components[ComponentID.NUMBER], components[ComponentID.CHATBOT]]})

    user_text_box = TextboxWrapper(
        component=components[ComponentID.USER_TEXT_BOX],
        handle_user_action=submit_btn.handle_user_action)

    number = NumberWrapper(
        component=components[ComponentID.NUMBER],
        handle_user_action=submit_btn.handle_user_action)

    return [btn1, btn2, files, upload_btn, submit_files_btn, clear_btn, submit_btn, user_text_box, number]
