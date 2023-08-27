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
from component_logic import initialize_component_wrappers
from component_wrapper import ComponentWrapper
from constants import (
    ComponentID,
    ComponentLabel,
    StepID,
    GRANT_APPLICATION_QUESTIONS_EXAMPLES
)



# create a workflow manager which contains all the chatbot steps and keeps track of the current step as well as the user context
workflow_manager = WorkflowManager()

with gr.Blocks(css="custom.css", theme=gr.themes.Default(primary_hue=gr.themes.colors.lime)) as demo:
    title = gr.HTML(
        """<h1><img src="file/publico_logo_no_circle.jpeg"></h1>""",
        elem_id="title")

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
    ) -> dict:
        '''Update visibility of components based on current step'''

        return {
            components[component_id]: gr.update(visible=True, **(
                properties
                    if not isinstance(properties, Callable)
                    else
                properties(workflow_state.context))
            ) for component_id, properties in workflow_state.current_step.components.items()}

    def handle_proceed_to_next_step(steps, workflow_state, component, chatbot):
        # 1. Generate chatbot messages
        for chatbot in generate_chatbot_messages_from_trigger(workflow_state, component, chatbot):
            yield chatbot, workflow_state

        # 2. Update chatbot step
        workflow_state = update_workflow_step(steps, workflow_state, component)

        # 3. Modify the context
        workflow_state = modify_context(workflow_state)

        # 4. Show initial chatbot message
        for workflow_state, chatbot in show_initial_chatbot_message(workflow_state, chatbot):
            yield chatbot, workflow_state



    # Calling unified_handler within the loop
    for c in component_wrappers:
        if c.user_action is None:
            continue

        chain = c.get_component_trigger()(
            # print info about the component that was triggered and the current step
            fn=ComponentWrapper.print_trigger_info,
            inputs=[c.component, workflow_state]
        )
        if c.proceed_to_next_step:
            chain.then(
                # if we proceed to the next step we make all components invisible while we perform the actions defined by the component wrapper
                fn=lambda: [gr.update(visible=False) for _ in range(len(internal_components_with_row))],
                outputs=internal_components_with_row
            ).then(
                fn=find_and_store_event_value,
                inputs=[workflow_state, c.component, gr.State(internal_components), *internal_components],
                outputs=workflow_state
            )

        # handle user action as defined by the component wrapper (e.g. upload files, submit text)
        chain = chain.then(**c.handle_user_action)

        if c.proceed_to_next_step:
            # Chain the new unified function here
            chain = chain.then(
                partial(handle_proceed_to_next_step, workflow_manager.steps),
                inputs=[
                    workflow_state,
                    c.component,
                    chatbot
                ],
                outputs=[chatbot, workflow_state]
            ).then(
                # make components in the next step visible (e.g. show the 'yes' and 'no' buttons if we're on the 'have you applied before' step)
                fn=partial(make_components_in_current_step_visible, workflow_manager.components),
                inputs=workflow_state,
                outputs=internal_components_with_row
            )


if __name__ == '__main__':
    demo.queue().launch()
