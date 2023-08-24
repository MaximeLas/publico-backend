from functools import partial
from typing import Callable

import gradio as gr
from gradio.components import IOComponent

# initialize langchain llm cache
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()


from chatbot_step import ChatbotStep
from chatbot_workflow import WorkflowManager, WorkflowState, modify_context, show_initial_chatbot_message, find_and_store_event_value, generate_chatbot_messages_from_trigger, update_workflow_step
from component_logic import (
    handle_btn_clicked,
    handle_submit,
    handle_files_uploaded,
    handle_files_submitted,
)
from component_wrapper import (
    ButtonWrapper,
    ComponentWrapper,
    FilesWrapper,
    NumberWrapper,
    UploadButtonWrapper,
    ClearButtonWrapper,
    TextboxWrapper
)
from constants import ComponentLabel, StepID, GRANT_APPLICATION_QUESTIONS_EXAMPLES



files_component = gr.Files(label='Documents', visible=False, interactive=False, file_types=['.docx', '.txt'])
with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.lime)) as demo:
    # create a workflow manager which contains all the chatbot steps and keeps track of the current step as well as the user context
    workflow_manager = WorkflowManager()
    workflow_state = gr.State(WorkflowState(workflow_manager.current_step_id, workflow_manager.context))

    with gr.Row():
        # create chatbot component
        chatbot = gr.Chatbot(
            value=[[workflow_manager.get_current_step().initial_chatbot_message.message, None]],
            label=ComponentLabel.CHATBOT,
            show_share_button=True,
            show_copy_button=True,
            height=600)

    with gr.Row():
        with gr.Column():
            # user text box component
            user_text_box_component = gr.Textbox(label=ComponentLabel.USER, visible=False, interactive=True, lines=3, show_copy_button=True, placeholder='Type your message here')
            # number component
            number_component = gr.Number(value=30, precision=0, label=ComponentLabel.NUMBER, visible=False, interactive=True)
            # submit button component
            submit_component = gr.Button(value=ComponentLabel.SUBMIT, variant='primary', visible=False)

        
        with gr.Row(visible=False) as examples_row:
            # examples component (not a true gradio 'Component' techincally)
            examples=gr.Examples(
                examples=GRANT_APPLICATION_QUESTIONS_EXAMPLES,
                inputs=user_text_box_component,
                label=ComponentLabel.EXAMPLES)

    with gr.Row():
        # button components
        btn1_component = gr.Button(value=ComponentLabel.START, variant='primary')
        btn2_component = gr.Button(visible=False)
        # upload button component
        upload_btn_component = gr.UploadButton(label=ComponentLabel.UPLOAD, variant='primary', visible=False, file_types=['.docx', '.txt'], file_count='multiple')
        # clear button component
        clear_btn_component=gr.ClearButton(value=ComponentLabel.CLEAR, variant='stop', visible=False, interactive=False)
        # submit files button component
        submit_files_btn_component=gr.Button(value=ComponentLabel.SUBMIT, variant='primary', visible=False, interactive=False)

    with gr.Row() as row:
        # files component
        files_component = gr.Files(label=ComponentLabel.FILES, visible=False, interactive=False, file_types=['.docx', '.txt'])

    # specify that the clear button should clear the files component
    clear_btn_component.add(files_component) # make the clear button clear the files

    # specify properties for each button we'll use
    start_btn_props = dict(value=ComponentLabel.START, variant='primary')
    yes_btn_props = dict(value=ComponentLabel.YES, variant='primary')
    no_btn_props = dict(value=ComponentLabel.NO, variant='stop')
    good_as_is_props = dict(value=ComponentLabel.GOOD_AS_IS, variant='primary')
    edit_it_props = dict(value=ComponentLabel.EDIT_IT, variant='primary')

    # define a list of tuples of the form (step_id, list of components for that step)
    step_components = [
        (StepID.START, {btn1_component: start_btn_props}),
        (StepID.HAVE_YOU_APPLIED_BEFORE, {btn1_component: yes_btn_props, btn2_component: no_btn_props}),
        (StepID.UPLOAD_FILES, {upload_btn_component: {}, files_component: {}, submit_files_btn_component: {}, clear_btn_component: {}}),
        (StepID.ENTER_QUESTION, {user_text_box_component: {}, submit_component: {}, examples_row: {}}),
        (StepID.ENTER_WORD_LIMIT, {number_component: {}, submit_component: {}}),
        (StepID.DO_COMPREHENSIVENESS_CHECK, {btn1_component: yes_btn_props, btn2_component: no_btn_props}),
        (StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION, {btn1_component: yes_btn_props, btn2_component: no_btn_props}),
        (StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT, {
            btn1_component: (lambda context: yes_btn_props if context.get_answer_of_current_implicit_question_to_be_answered() is None else good_as_is_props),
            btn2_component: (lambda context: no_btn_props if context.get_answer_of_current_implicit_question_to_be_answered() is None else edit_it_props)}),
        (StepID.PROMPT_USER_TO_SUBMIT_ANSWER, {user_text_box_component: {}, submit_component: {}}),
        (StepID.READY_TO_GENERATE_FINAL_ANSWER, {btn1_component: yes_btn_props}),
        (StepID.DO_ANOTHER_QUESTION, {btn1_component: yes_btn_props, btn2_component: no_btn_props})
    ]

    # set the components for each step
    for step_id, components in step_components:
        workflow_manager.get_step(step_id).components = components

    def handle_btn_clicked(
        btn_label: str,
        chat_history: list[list],
        workflow_state: WorkflowState
    ):
        chat_history[-1][1] = f'**{btn_label}**'
        
        return {
            chatbot: chat_history,
            user_text_box_component: workflow_state.context.get_answer_of_current_implicit_question_to_be_answered() if btn_label == ComponentLabel.EDIT_IT else gr.skip()
        }
        '''if btn_label == ComponentLabel.EDIT_IT:
            outputs['user_text_box_component'] = workflow_state.context.get_answer_of_current_implicit_question_to_be_answered()

        return {'chatbot': chat_history} | ({'user_text_box_component': workflow_state.context.get_answer_of_current_implicit_question_to_be_answered()} if btn_label == ComponentLabel.EDIT_IT else {})'''
    # create wrappers for each component and define the actions to be executed after being triggered, if any
    create_btn = lambda btn_component: ButtonWrapper(
        component=btn_component,
        handle_user_action={
            'fn': handle_btn_clicked,
            'inputs': [btn_component, chatbot, workflow_state],
            'outputs': [chatbot, user_text_box_component]})
    btn1 = create_btn(btn1_component)
    btn2 = create_btn(btn2_component)

    files = FilesWrapper(component=files_component)

    upload_btn = UploadButtonWrapper(
        component=upload_btn_component,
        handle_user_action={
            'fn': handle_files_uploaded,
            'inputs': [upload_btn_component, files_component],
            'outputs': [files_component, submit_files_btn_component, clear_btn_component]
        })

    submit_files_btn = ButtonWrapper(
        component=submit_files_btn_component,
        handle_user_action={
            'fn': handle_files_submitted,
            'inputs': files_component
        })

    clear_btn = ClearButtonWrapper(
        component=clear_btn_component,
        handle_user_action={
            'fn': lambda: [gr.update(interactive=False)] * 2,
            'outputs': [submit_files_btn_component, clear_btn_component]})

    submit_btn = ButtonWrapper(
        component=submit_component,
        handle_user_action={
            'fn': handle_submit,
            'inputs': [user_text_box_component, number_component, chatbot],
            'outputs': [user_text_box_component, number_component, chatbot]})

    user_text_box = TextboxWrapper(
        component=user_text_box_component,
        handle_user_action=submit_btn.handle_user_action)

    number = NumberWrapper(
        component=number_component,
        handle_user_action=submit_btn.handle_user_action)


    components: list[ComponentWrapper] = [
        btn1, btn2,
        files, upload_btn, submit_files_btn, clear_btn,
        user_text_box, submit_btn,
        number]

    internal_components: list[IOComponent] = [component.component for component in components]
    internal_components_with_row: list[IOComponent | gr.Row] = internal_components + [examples_row]


    def update_component_visibility(
        steps: dict[StepID, ChatbotStep],
        workflow_state: WorkflowState
    ) -> list:
        '''Update visibility of components based on current step'''
        return {
            component: gr.update(visible=True, **(properties if not isinstance(properties, Callable) else properties(workflow_state.context)))
            for component, properties in steps[workflow_state.current_step_id].components.items()}


    def control_components_visibility(num_of_components: int, proceed: bool) -> list:
        '''Update the visibility of components based on the 'proceed' value.'''

        return [
            gr.update(visible=False) if proceed else gr.skip() for _ in range(num_of_components)]


    for c in components:
        if c.user_action is None:
            continue

        chain = c.get_component_trigger()(
            # print info about the component that was triggered and the current step
            fn=c.print_trigger_info,
            inputs=[c.component, workflow_state]
        ).then(
            # update visibility of components based on current chatbot step and user interaction type of component
            fn=partial(control_components_visibility, len(internal_components_with_row)),
            inputs=gr.State(c.proceed_to_next_step),
            outputs=internal_components_with_row # type: ignore
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
            fn=partial(update_component_visibility, workflow_manager.steps),
            inputs=workflow_state,
            outputs=internal_components_with_row # type: ignore
        )


if __name__ == '__main__':
    demo.queue().launch()


# 5 sec load / immediately start - 8 components
# 5 sec load / immediately start - 10 components
# 6 sec load / 1.5 sec start - 11 components
# 8 sec load / 3 sec start - 12 components
# 8 sec load / 5.5 sec start - 13 components
# 12 sec load / 10.5 sec start - 14 components
# 21 sec load / 20 sec start - all 15 components
# 18 sec load / 18 sec start - all 15 components + only some, around half, of the event
