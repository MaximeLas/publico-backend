from functools import partial
from typing import Callable

import gradio as gr
from gradio.blocks import Block
from gradio.components import IOComponent

# initialize langchain llm cache
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()

from chatbot_workflow import (
    WorkflowManager,
    WorkflowState,
    modify_context,
    show_initial_chatbot_message,
    find_and_store_event_value,
    generate_chatbot_messages_from_trigger,
    update_workflow_step
)
from component_logic import (
    initialize_component_wrappers,
)
from component_wrapper import (
    ComponentWrapper,
)
from constants import ComponentID, ComponentLabel, StepID, GRANT_APPLICATION_QUESTIONS_EXAMPLES



# create a workflow manager which contains all the chatbot steps and keeps track of the current step as well as the user context
workflow_manager = WorkflowManager()

with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.lime)) as demo:
    chatbot = workflow_manager.get_component(ComponentID.CHATBOT)
    user_text_box_component = workflow_manager.get_component(ComponentID.USER_TEXT_BOX)
    number_component = workflow_manager.get_component(ComponentID.NUMBER)
    submit_user_input_component = workflow_manager.get_component(ComponentID.SUBMIT_USER_INPUT_BTN)
    btn_1_component = workflow_manager.get_component(ComponentID.BTN_1)
    btn_2_component = workflow_manager.get_component(ComponentID.BTN_2)
    btn_3_component = workflow_manager.get_component(ComponentID.BTN_3)
    files_component = workflow_manager.get_component(ComponentID.FILES)
    upload_files_btn_component = workflow_manager.get_component(ComponentID.UPLOAD_FILES_BTN)
    submit_files_btn_component = workflow_manager.get_component(ComponentID.SUBMIT_FILES_BTN)
    clear_files_btn_component = workflow_manager.get_component(ComponentID.CLEAR_FILES_BTN)
    clear_files_btn_component.add(files_component) # make the clear button clear the files

    with gr.Row():
        chatbot.render()

    with gr.Row():
        btn_1_component.render()
        btn_2_component.render()

    with gr.Row():
        with gr.Column():
            user_text_box_component.render()
            number_component.render()
            submit_user_input_component.render()
            upload_files_btn_component.render()
            clear_files_btn_component.render()
            submit_files_btn_component.render()

        files_component.render()

        with gr.Row(visible=False) as examples_row:
            examples=gr.Examples(
                examples=GRANT_APPLICATION_QUESTIONS_EXAMPLES,
                inputs=user_text_box_component,
                label=ComponentLabel.EXAMPLES)

        workflow_manager.components[ComponentID.EXAMPLES] = examples_row
        workflow_manager.steps[StepID.ENTER_QUESTION].components[ComponentID.EXAMPLES] = {}


    workflow_state = gr.State(WorkflowState(workflow_manager.get_step(StepID.START)))
    component_wrappers: list[ComponentWrapper] = initialize_component_wrappers(workflow_manager.components, workflow_state)
    internal_components: list[IOComponent] = [component.component for component in component_wrappers]
    internal_components_with_row: list[Block] = internal_components + [examples_row]

    def make_components_in_current_step_visible(
        components: dict[ComponentID, Block],
        workflow_state: WorkflowState
    ) -> list:
        '''Update visibility of components based on current step'''

        return {
            components[component_id]: gr.update(visible=True, **(
                properties
                    if not isinstance(properties, Callable)
                    else
                properties(workflow_state.context))
            ) for component_id, properties in workflow_state.current_step.components.items()}


    for c in component_wrappers:
        if c.user_action is None:
            continue

        chain = c.get_component_trigger()(
            # print info about the component that was triggered and the current step
            fn=c.print_trigger_info,
            inputs=[c.component, workflow_state]
        ).then(
            # if we proceed to the next step we make all components invisible while we perform the actions defined by the component wrapper
            fn=lambda proceed: [gr.update(visible=False) if proceed else gr.skip() for _ in range(len(internal_components_with_row))],
            inputs=gr.State(c.proceed_to_next_step),
            outputs=internal_components_with_row
        )

        if c.proceed_to_next_step:
            # if we proceed then we store the value of the relevant component in the context, if defined
            # for the current step (e.g. store the user's reply to a question)
            chain = chain.then(
                fn=find_and_store_event_value,
                inputs=[workflow_state, c.component, gr.State(internal_components), *internal_components],
                outputs=workflow_state)

        if c.handle_user_action is not None:
            # handle user action as defined by the component wrapper (e.g. upload files, submit text)
            chain = chain.then(**c.handle_user_action)

        if not c.proceed_to_next_step:
            continue

        chain.then(
            # generate any chatbot messages for current step (e.g. validation message following files upload, llm answer to question)
            fn=generate_chatbot_messages_from_trigger,
            inputs=[workflow_state, c.component, chatbot],
            outputs=chatbot
        ).then(
            # update chatbot step (e.g. move to next step if user has submitted files)
            fn=partial(update_workflow_step, workflow_manager.steps),
            inputs=[workflow_state, c.component],
            outputs=workflow_state
        ).then(
            # modify the context (e.g. add new question to context if we're on the 'enter question' step)
            fn=modify_context,
            inputs=workflow_state,
            outputs=workflow_state
        ).then(
            # show the initial (chatbot) message of the next step
            fn=show_initial_chatbot_message,
            inputs=[workflow_state, chatbot],
            outputs=[workflow_state, chatbot]
        ).then(
            # make components in the next step visible (e.g. show the 'yes' and 'no' buttons if we're on the 'have you applied before' step)
            fn=partial(make_components_in_current_step_visible, workflow_manager.components),
            inputs=workflow_state,
            outputs=internal_components_with_row
        )


if __name__ == '__main__':
    demo.queue().launch()
