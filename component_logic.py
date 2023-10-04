import tempfile
from devtools import debug

import gradio as gr
from gradio.blocks import Block

from chatbot_workflow import WorkflowState
from component_wrapper import ButtonWrapper, ClearButtonWrapper, ComponentWrapper, NumberWrapper, TextboxWrapper, UploadButtonWrapper
from constants import DEFAULT_WORD_LIMIT, ComponentID, ComponentLabel



def handle_btn_clicked(
    btn_label: str,
    chat_history: list[list],
    workflow_state: WorkflowState
):
    chat_history += [[f'**{btn_label}**', None]]

    text_box = workflow_state.context.get_answer_of_current_implicit_question() if btn_label == ComponentLabel.EDIT_IT else gr.skip()

    if save_fn := workflow_state.current_step.save_event_outcome_fn:
        save_fn(workflow_state.context, btn_label)

    return chat_history, text_box, workflow_state


def handle_submit(
    user_message: str,
    number_1: int,
    number_2: int,
    chat_history: list[list],
    workflow_state: WorkflowState
):
    values_to_save: list[str | int]

    # check which input was submitted
    input_submitted: str
    if user_message != '':
        values_to_save = [user_message]
        input_submitted = f'**{user_message}**'
        user_message = None
    elif number_2 == 0:
        values_to_save = [number_1]
        input_submitted = f'**{str(number_1)}**'
        number_1 = DEFAULT_WORD_LIMIT
    else:
        values_to_save = [number_1, number_2]
        input_submitted = f'**{str(number_1)}**\n**{str(number_2)}**'
        number_1 = DEFAULT_WORD_LIMIT
        number_2 = 0

    chat_history += [[input_submitted, None]]
    workflow_state.current_step.save_event_outcome_fn(workflow_state.context, *values_to_save)

    return user_message, number_1, number_2, chat_history, workflow_state


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

    return all_files, gr.Button(interactive=True), gr.ClearButton(interactive=True)


def handle_files_submitted(
    files: list[tempfile._TemporaryFileWrapper],
    workflow_state: WorkflowState
):
    debug(**{f'File #{i+1} uploaded': file.name.split("/")[-1] for i, file in enumerate(files)})
    workflow_state.current_step.save_event_outcome_fn(workflow_state.context, files)

    return workflow_state



def create_component_wrappers(
    components: dict[ComponentID, Block],
    workflow_state: WorkflowState
) -> list[ComponentWrapper]:
    # create wrappers for each component and define the actions to be executed after being triggered, if any

    create_btn = lambda btn_component: ButtonWrapper(
        component=btn_component,
        handle_user_action={
            'fn': handle_btn_clicked,
            'inputs': [btn_component, components[ComponentID.CHATBOT], workflow_state],
            'outputs': [components[ComponentID.CHATBOT], components[ComponentID.USER_TEXT_BOX], workflow_state]})

    btn1 = create_btn(components[ComponentID.BTN_1])
    btn2 = create_btn(components[ComponentID.BTN_2])

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
            'fn': handle_files_submitted,
            'inputs': [components[ComponentID.FILES], workflow_state],
            'outputs': workflow_state
        })

    clear_btn = ClearButtonWrapper(
        component=components[ComponentID.CLEAR_FILES_BTN],
        handle_user_action={
            'fn': lambda: [gr.Button(interactive=False), gr.ClearButton(interactive=False)],
            'outputs': [components[ComponentID.SUBMIT_FILES_BTN], components[ComponentID.CLEAR_FILES_BTN]]})

    submit_btn = ButtonWrapper(
        component=components[ComponentID.SUBMIT_USER_INPUT_BTN],
        handle_user_action={
            'fn': handle_submit,
            'inputs': [components[ComponentID.USER_TEXT_BOX], components[ComponentID.NUMBER_1], components[ComponentID.NUMBER_2], components[ComponentID.CHATBOT], workflow_state],
            'outputs': [components[ComponentID.USER_TEXT_BOX], components[ComponentID.NUMBER_1], components[ComponentID.NUMBER_2], components[ComponentID.CHATBOT], workflow_state]})

    user_text_box = TextboxWrapper(
        component=components[ComponentID.USER_TEXT_BOX],
        handle_user_action=submit_btn.handle_user_action)

    number_1 = NumberWrapper(
        component=components[ComponentID.NUMBER_1],
        handle_user_action=submit_btn.handle_user_action)

    number_2 = NumberWrapper(
        component=components[ComponentID.NUMBER_2],
        handle_user_action=submit_btn.handle_user_action)

    return [btn1, btn2, upload_btn, submit_files_btn, clear_btn, submit_btn, user_text_box, number_1, number_2]
