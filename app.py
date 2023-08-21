from functools import partial

import gradio as gr
from gradio.components import IOComponent

# initialize langchain llm cache
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()


from chatbot_step import ChatbotStep
from chatbot_workflow import WorkflowManager, WorkflowState, modify_context, show_initial_chatbot_message, find_and_store_event_value, generate_chatbot_messages_from_trigger, update_workflow_step
from component_logic import (
    handle_edit_it_clicked,
    handle_good_as_is_clicked,
    handle_number_submitted,
    handle_write_one_myself_clicked,
    handle_yes_no_clicked,
    handle_files_uploaded,
    handle_files_submitted,
    handle_text_submitted
)
from component_wrapper import (
    ButtonWrapper,
    ComponentWrapper,
    FilesWrapper,
    NumberWrapper,
    StartWrapper,
    YesNoWrapper,
    UploadWrapper,
    SubmitWrapper,
    ClearWrapper,
    TextWrapper,
    SubmitTextButtonWrapper
)
from constants import ComponentLabel, StepID, GRANT_APPLICATION_QUESTIONS_EXAMPLES



files_component = gr.Files(label='Documents', visible=False, interactive=False, file_types=['.docx', '.txt'])
with gr.Blocks() as demo:
    # create a workflow manager which contains all the chatbot steps and keeps track of the current step as well as the user context
    workflow_manager = WorkflowManager()
    workflow_state = gr.State(WorkflowState(workflow_manager.current_step_id, workflow_manager.context))

    with gr.Row():
        # create chatbot component
        chatbot = gr.Chatbot(
            value=[[workflow_manager.get_current_step().initial_chatbot_message.message, None]],
            label=ComponentLabel.CHATBOT,
            show_share_button=True,
            height=600)

    with gr.Row():
        with gr.Column():
            # user text box component
            user_text_box_component = gr.Textbox(label=ComponentLabel.USER, visible=False, interactive=True, lines=3, show_copy_button=True, placeholder='Type your message here')
            # submit text button component
            submit_text_btn_component = gr.Button(value=ComponentLabel.SUBMIT_TEXT, variant='primary', visible=False)

            # number component
            number_component = gr.Number(value=30, precision=0, label=ComponentLabel.NUMBER, visible=False, interactive=True)
            # submit number component
            submit_number_btn_component = gr.Button(value=ComponentLabel.SUBMIT_NUMBER, variant='primary', visible=False)

        
        with gr.Row(visible=False) as examples_row:
            # examples component (not a true gradio 'Component' techincally)
            examples=gr.Examples(
                examples=GRANT_APPLICATION_QUESTIONS_EXAMPLES,
                inputs=user_text_box_component,
                label=ComponentLabel.EXAMPLES)

    with gr.Row():
        # start button component
        start_btn_component = gr.Button(value=ComponentLabel.START, variant='primary', visible=True)
        # yes/no button component
        yes_btn_component = gr.Button(value=ComponentLabel.YES, variant='primary', visible=False)
        no_btn_component = gr.Button(value=ComponentLabel.NO, variant='stop', visible=False)
        # upload button component
        upload_btn_component = gr.UploadButton(label=ComponentLabel.UPLOAD, variant='primary', visible=False, file_types=['.docx', '.txt'], file_count='multiple')
        # clear button component
        clear_btn_component=gr.ClearButton(value=ComponentLabel.CLEAR, variant='stop', visible=False, interactive=False)
        # submit files button component
        submit_files_btn_component=gr.Button(value=ComponentLabel.SUBMIT_FILES, variant='primary', visible=False, interactive=False)
        # buttons for handling generated answer to implicit question
        good_as_is_btn_component = gr.Button(value=ComponentLabel.GOOD_AS_IS, variant='primary', visible=False)
        edit_it_btn_component = gr.Button(value=ComponentLabel.EDIT_IT, variant='primary', visible=False)
        write_one_myself_btn_component = gr.Button(value=ComponentLabel.WRITE_ONE_MYSELF, variant='primary', visible=False)
        # of course button component
        of_course_btn_component = gr.Button(value=ComponentLabel.OF_COURSE, variant='primary', visible=False)

    with gr.Row() as row:
        # files component
        files_component = gr.Files(label=ComponentLabel.FILES, visible=False, interactive=False, file_types=['.docx', '.txt'])

    # specify that the clear button should clear the files component
    clear_btn_component.add(files_component) # make the clear button clear the files


    # define a list of tuples of the form (step_id, list of components for that step)
    step_components = [
        (StepID.START, [start_btn_component]),
        (StepID.HAVE_YOU_APPLIED_BEFORE, [yes_btn_component, no_btn_component]),
        (StepID.UPLOAD_FILES, [upload_btn_component, files_component, submit_files_btn_component, clear_btn_component]),
        (StepID.ENTER_QUESTION, [user_text_box_component, submit_text_btn_component, examples_row]),
        (StepID.ENTER_WORD_LIMIT, [number_component, submit_number_btn_component]),
        (StepID.DO_COMPREHENSIVENESS_CHECK, [yes_btn_component, no_btn_component]),
        (StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION, [yes_btn_component, no_btn_component]),
        (StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT, [
            yes_btn_component, no_btn_component, good_as_is_btn_component, edit_it_btn_component, write_one_myself_btn_component]),
        (StepID.PROMPT_USER_TO_SUBMIT_ANSWER, [user_text_box_component, submit_text_btn_component]),
        (StepID.READY_TO_GENERATE_FINAL_ANSWER, [of_course_btn_component]),
        (StepID.DO_ANOTHER_QUESTION, [yes_btn_component, no_btn_component])
    ]

    # set the components for each step
    for step_id, components in step_components:
        workflow_manager.get_step(step_id).components = components


    # create wrappers for each component and define the actions to be executed after being triggered, if any
    start_btn = StartWrapper(component=start_btn_component)

    create_yes_no_btn = lambda yes_no_btn_component: YesNoWrapper(
        component=yes_no_btn_component,
        handle_user_action={
            'fn': handle_yes_no_clicked,
            'inputs': [yes_no_btn_component, chatbot],
            'outputs': chatbot})
    yes_btn = create_yes_no_btn(yes_btn_component)
    no_btn = create_yes_no_btn(no_btn_component)

    files = FilesWrapper(component=files_component)

    upload_btn = UploadWrapper(
        component=upload_btn_component,
        handle_user_action={
            'fn': handle_files_uploaded,
            'inputs': [upload_btn_component, files_component],
            'outputs': [files_component, submit_files_btn_component, clear_btn_component]
        })

    submit_files_btn = SubmitWrapper(
        component=submit_files_btn_component,
        handle_user_action={
            'fn': handle_files_submitted,
            'inputs': files_component
        })

    clear_btn = ClearWrapper(
        component=clear_btn_component,
        handle_user_action={
            'fn': lambda: [gr.update(interactive=False)] * 2,
            'outputs': [submit_files_btn_component, clear_btn_component]})

    user_text_box = TextWrapper(
        component=user_text_box_component,
        handle_user_action={
            'fn': handle_text_submitted,
            'inputs': [user_text_box_component, chatbot],
            'outputs': [user_text_box_component, chatbot]})

    submit_text_btn = SubmitTextButtonWrapper(
        component=submit_text_btn_component,
        handle_user_action=user_text_box.handle_user_action)

    number = NumberWrapper(
        component=number_component,
        handle_user_action={
            'fn': handle_number_submitted,
            'inputs': [number_component, chatbot],
            'outputs': [number_component, chatbot]})

    submit_number_btn = SubmitTextButtonWrapper(
        component=submit_number_btn_component,
        handle_user_action=number.handle_user_action)

    good_as_is_btn = ButtonWrapper(
        component=good_as_is_btn_component,
        handle_user_action={
            'fn': handle_good_as_is_clicked,
            'inputs': [good_as_is_btn_component, chatbot],
            'outputs': [chatbot]})

    edit_it_btn = ButtonWrapper(
        component=edit_it_btn_component,
        handle_user_action={
            'fn': handle_edit_it_clicked,
            'inputs': [edit_it_btn_component, chatbot, workflow_state],
            'outputs': [chatbot, user_text_box_component]})

    write_one_myself_btn = ButtonWrapper(
        component=write_one_myself_btn_component,
        handle_user_action={
            'fn': handle_write_one_myself_clicked,
            'inputs': [write_one_myself_btn_component, chatbot],
            'outputs': [chatbot]})

    of_course_btn = ButtonWrapper(component=of_course_btn_component)

    components: list[ComponentWrapper] = [
        start_btn, yes_btn, no_btn,
        files, upload_btn, submit_files_btn, clear_btn,
        user_text_box, submit_text_btn,
        number, submit_number_btn,
        good_as_is_btn, edit_it_btn, write_one_myself_btn,
        of_course_btn]

    internal_components: list[IOComponent] = [component.component for component in components]
    internal_components_with_row: list[IOComponent | gr.Row] = internal_components + [examples_row]


    def update_component_visibility(
        internal_components_with_row: list[IOComponent | gr.Row],
        steps: dict[StepID, ChatbotStep],
        workflow_state: WorkflowState
    ) -> list:
        '''Update visibility of components based on current step'''

        return [
            gr.update(
                visible=(component in steps[workflow_state.current_step_id].components and (
                    # current sort-of hard-coded solution to show the correct buttons for that one step
                    workflow_state.current_step_id is not StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT or (
                        component in (yes_btn_component, no_btn_component) and
                        workflow_state.context.get_answer_of_current_implicit_question_to_be_answered() is None
                    ) or (
                        component in (good_as_is_btn_component, edit_it_btn_component, write_one_myself_btn_component) and
                        workflow_state.context.get_answer_of_current_implicit_question_to_be_answered() is not None
                    )
                ))
            )
            for component in internal_components_with_row
        ]


    def control_components_visibility(num_of_components: int, proceed: bool) -> list:
        '''Update the visibility of components based on the 'proceed' value.'''

        return [
            gr.update(visible=False) if proceed else gr.skip() for _ in range(num_of_components)]


    for c in components:
        if c.user_action is None:
            continue

        chain = c.get_initial_chain_following_trigger(
        ).then(
            # update visibility of components based on current chatbot step and user interaction type of component
            fn=partial(control_components_visibility, len(internal_components_with_row)),
            inputs=gr.State(c.proceed_to_next_step),
            outputs=internal_components_with_row # type: ignore
        ).then(
            # print current step id
            fn=lambda workflow_state: print(f"Current step: '{workflow_state.current_step_id}'\n"),
            inputs=workflow_state
        )

        if c.proceed_to_next_step:
            # if we proceed then we store the value of the relevant component in the context, if defined
            # for the current step (e.g. store the user's reply to a question)
            chain = chain.then(
                fn=partial(find_and_store_event_value, workflow_manager.steps),
                inputs=[workflow_state, c.component, gr.State(internal_components), *internal_components],
                outputs=workflow_state,
            )

        if c.handle_user_action is not None:
            # handle user action as defined by the component wrapper (e.g. upload files, submit text)
            chain = chain.then(**c.handle_user_action)

        if not c.proceed_to_next_step:
            continue

        chain.then(
            # generate any chatbot messages for current step (e.g. validation message following files upload, llm answer to question)
            fn=partial(generate_chatbot_messages_from_trigger, workflow_manager.steps),
            inputs=[workflow_state, c.component, chatbot],
            outputs=chatbot
        ).then(
            # update chatbot step (e.g. move to next step if user has submitted files)
            fn=partial(update_workflow_step, workflow_manager.steps),
            inputs=[workflow_state, c.component],
            outputs=workflow_state
        ).then(
            # modify the context (e.g. add new question to context if we're on the 'enter question' step)
            fn=partial(modify_context, workflow_manager.steps),
            inputs=workflow_state,
            outputs=workflow_state
        ).then(
            # show the initial (chatbot) message of the next step
            fn=partial(show_initial_chatbot_message, workflow_manager.steps),
            inputs=[workflow_state, chatbot],
            outputs=[workflow_state, chatbot]
        ).then(
            # update visibility of components by making visible only those components relevant to the next step
            fn=partial(update_component_visibility, internal_components_with_row, workflow_manager.steps),
            inputs=workflow_state,
            outputs=internal_components_with_row # type: ignore
        )


if __name__ == '__main__':
    demo.queue().launch()
