from functools import partial
import time
import warnings

import gradio as gr

# initialize langchain llm cache
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()

from workflow.chatbot_workflow import (
    WorkflowManager,
    WorkflowState,
    modify_context,
    get_initial_chatbot_message,
    generate_chatbot_messages_from_trigger,
    update_visibility_of_components_in_current_step,
    update_workflow_step
)
from components.component_logic import create_component_wrappers
from components.component_wrapper import ComponentWrapper, EventParameters
from configurations.constants import (
    ComponentID,
    ComponentLabel,
    StepID,
    GRANT_APPLICATION_QUESTIONS_EXAMPLES,
    PAGE_TITLE,
    SERVER_PORT,
    CREATE_LINK,
    EXCLUDE_LOGO
)

from configurations.settings import update_sqlite3_if_necessary


# create a workflow manager which contains all the chatbot steps and keeps track of the current step as well as the user context
workflow_manager = WorkflowManager()

with gr.Blocks(css='css/custom.css', theme=gr.themes.Default(primary_hue=gr.themes.colors.lime), title=PAGE_TITLE) as demo:
    # TODO: Fix logo not showing up or remove it
    #if not EXCLUDE_LOGO:
    #    title = gr.HTML(value='<h1><img src="file/publico_logo_no_circle.jpeg"></h1>', elem_id='title')

    chatbot = workflow_manager.get_component(ComponentID.CHATBOT)
    user_text_box_component = workflow_manager.get_component(ComponentID.USER_TEXT_BOX)
    number_1_component = workflow_manager.get_component(ComponentID.NUMBER_1)
    number_2_component = workflow_manager.get_component(ComponentID.NUMBER_2)
    submit_user_input_component = workflow_manager.get_component(ComponentID.SUBMIT_USER_INPUT_BTN)
    btn_1_component = workflow_manager.get_component(ComponentID.BTN_1)
    btn_2_component = workflow_manager.get_component(ComponentID.BTN_2)
    files_component = workflow_manager.get_component(ComponentID.FILES)
    upload_files_btn_component = workflow_manager.get_component(ComponentID.UPLOAD_FILES_BTN)
    submit_files_btn_component = workflow_manager.get_component(ComponentID.SUBMIT_FILES_BTN)
    clear_files_btn_component = workflow_manager.get_component(ComponentID.CLEAR_FILES_BTN)
    clear_files_btn_component.add(files_component) # make the clear button clear the files
    answers_markup = workflow_manager.get_component(ComponentID.ANSWERS_MARKUP)
    answers_file = workflow_manager.get_component(ComponentID.ANSWERS_FILE)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot.render()

            with gr.Row():
                btn_1_component.render()
                btn_2_component.render()

            with gr.Row():
                with gr.Column():
                    user_text_box_component.render()
                    number_1_component.render()
                    number_2_component.render()
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

        with gr.Column(scale=2):
            answers_file.render()
            answers_markup.render()



    def handle_proceed_to_next_step(
        workflow_manager: WorkflowManager,
        handle_user_action: EventParameters,
        proceed_to_next_step: bool,
        workflow: WorkflowState,
        component: str,
        chat_history: list[list],
        *handle_user_action_inputs
    ):
        # 1. Print info about the component that was triggered and the current step
        ComponentWrapper.print_trigger_info(component, workflow.current_step_id)

        # 2. Handle user action as defined by the component wrapper (e.g. upload files, submit text)
        fn_output_values = handle_user_action['fn'](*handle_user_action_inputs)
        fn_output_components = handle_user_action['outputs']
        outputs_dict = (
            dict(zip(fn_output_components, fn_output_values))
                if isinstance(fn_output_components, tuple) or isinstance(fn_output_components, list)
                else
            {fn_output_components: fn_output_values}
        )
        yield outputs_dict

        if not proceed_to_next_step:
            return

        # update chat history if it was modified in the handle_user_action function
        chat_history = outputs_dict.get(chatbot, chat_history)

        # 3. Make current step's components invisible
        if invisible_components := update_visibility_of_components_in_current_step(
            workflow_manager.components, workflow, False):
            yield invisible_components

        # 4. Generate chatbot messages
        new_chat_history = chat_history
        for chatbot_messages in generate_chatbot_messages_from_trigger(workflow, component):
            new_chat_history = chat_history + chatbot_messages
            yield {chatbot: new_chat_history}

        # 5. Update Application Answers (markdown and txt file)
        report_original, report_formatted = workflow.context.get_completed_application()
        if report_original:
            yield {
                answers_markup: report_formatted,
                answers_file: gr.update(value=generate_grant_application_txt_file(report_original), visible=True)
            }

        # 6. Update chatbot step
        update_workflow_step(workflow_manager.steps, workflow, component)

        # 7. Modify the context
        modify_context(workflow)

        # 8. Show initial chatbot message
        time.sleep(0.25)
        for initial_chatbot_message in get_initial_chatbot_message(workflow):
            yield {chatbot: new_chat_history + initial_chatbot_message}

        # 9. Make next step's components visible
        if visible_components := update_visibility_of_components_in_current_step(
            workflow_manager.components, workflow, True):
            yield visible_components


    workflow_state = gr.State(WorkflowState(workflow_manager.get_step(StepID.START)))
    for c in create_component_wrappers(workflow_manager.components, workflow_state):
        chain = c.get_component_trigger()(
            fn=partial(handle_proceed_to_next_step, workflow_manager, c.handle_user_action),
            inputs=[
                gr.State(c.proceed_to_next_step),
                workflow_state,
                c.component,
                chatbot,
                *(c.handle_user_action.get('inputs', []))
            ],
            outputs=[workflow_state, *(list(workflow_manager.components.values()))]
        )

    def generate_grant_application_txt_file(report: str) -> str:
        file_name = f'grant_application_{time.strftime("%Y-%m-%d_%H-%M-%S")}.txt'

        with open(file_name, 'w') as file:
            file.write(report)

        return file_name

    @chatbot.like(inputs=None, outputs=None)
    def vote(data: gr.LikeData):
        print(f'You {"upvoted" if data.liked else "downvoted"} this response:\n{data.value}\n')


if __name__ == '__main__':
    update_sqlite3_if_necessary()
    warnings.filterwarnings('ignore') # ignore the gr.update deprecated warnings

    demo.queue().launch(
        favicon_path='assets/favicon.ico',
        server_port=SERVER_PORT,
        share=CREATE_LINK,
        allowed_paths=['assets/*']
    )
