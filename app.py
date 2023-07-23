from typing import Callable

import gradio as gr

from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()

from constants import *
from helpers import *
from prompts import *
from comprehensiveness import *
from chatbot_step import *
from component_wrapper import *


# define dict to store output variables from chatbot steps
OUTPUT_VARIABLES = {}

# define chatbot steps and their properties
CHATBOT_STEPS: list[ChatbotStep] = [
    YesNoStep( # 0
        message="Have you applied for this grant before?",
        output_key=OutputKeys.HAS_APPLIED_FOR_THIS_GRANT_BEFORE,
        steps_to_skip_if_no=1),
    FilesStep( # 1
        message="That's very useful! Please upload your {kind_of_document}.",
        output_key=OutputKeys.PRIOR_GRANT_APPLICATIONS,
        kind_of_document="prior grant application(s)"),
    TextStep( # 2
        message="Now, on to the first question! Please let me know what the first application question is, or copy and paste it from the application portal.",
        output_key=OutputKeys.APPLICATION_QUESTION),
    TextStep( # 3
        message="What is the word limit?",
        output_key=OutputKeys.WORD_LIMIT,
        generate_output_fns=[generate_answer_to_question]),
    YesNoStep( # 4
        message="Do you want to check the comprehensiveness of the generated answer?",
        output_key=OutputKeys.CHECK_COMPREHENSIVENESS,
        generate_output_fns=[check_for_comprehensiveness, generate_answers_for_implicit_questions, generate_final_answer])]


def save_to_output_variable(output_key: OutputKeys, value):
    OUTPUT_VARIABLES[output_key] = value
    print(f'{output_key}: {OUTPUT_VARIABLES[output_key]}\n')


def generate_chatbot_messages(generate_output_fns: list[Callable[[dict], str | list[str] | None]]) -> list[list[str]]:
    new_chatbot_messages = []
    for fn in generate_output_fns:
        if (response := fn(OUTPUT_VARIABLES)) is not None:
            if type(response) == list:
                for chatbot_response in response:
                    new_chatbot_messages += [[chatbot_response, None]]
            else:
                new_chatbot_messages += [[response, None]]

    return new_chatbot_messages


def handle_user_interaction(user_message, chat_history, step: int):
    text_step = CHATBOT_STEPS[step]
    assert type(text_step) is TextStep

    # save user message to output variable
    save_to_output_variable(text_step.output_key, user_message)
    
    # update chat history with user message
    new_chat_history = chat_history[:-1] + [[chat_history[-1][0], user_message]]

    # generate output if necessary and update chat history with it
    if (fns := text_step.generate_output_fns) is not None:
        new_chat_history += generate_chatbot_messages(fns)
    
    return '', new_chat_history


def handle_yes_no_interaction(yes_or_no: str, chat_history, step: int):
    yes_no_step = CHATBOT_STEPS[step]
    assert type(yes_no_step) is YesNoStep

    # save YES or NO to output variable
    save_to_output_variable(yes_no_step.output_key, yes_or_no)

    # generate output if necessary and update chat history with it
    if (fns := yes_no_step.generate_output_fns) is not None:
        chat_history += generate_chatbot_messages(fns)
    
    return chat_history, step + yes_no_step.steps_to_skip(yes_or_no)


def handle_files_uploaded_from_files(files: list, step: int, chat_history: list[list]):
    files_step = CHATBOT_STEPS[step]
    assert type(files_step) is FilesStep

    # save file names to output variable
    save_to_output_variable(files_step.output_key, [file.name for file in files])

    # iterate over files and print their names
    for file in files: print(f'File uploaded: {file.name.split("/")[-1]}')
    print()

    # update chat history with validation message
    validation_message = f'You successfully uploaded {len(files)} {files_step.kind_of_document}! 🎉'

    return chat_history + [[validation_message, None]]


def handle_files_uploaded_from_btn(files_uploaded: list, files_present):
    new_files = []
    if files_present is not None:
        if type(files_present) is list:
            for file in files_present:
                new_files.append(file.name)
        else:
            new_files.append(files_present.name)

    for file in files_uploaded:
        if file.name not in new_files:
            new_files.append(file.name)
        else:
            print(f'{file.name.rsplit("/", 1)[-1]} already present')

    print(f'Total files uploaded so far: {len(new_files)}\n')

    make_btn_interactive_if_needed = gr.update(interactive=len(new_files)>0)
    return new_files, make_btn_interactive_if_needed, make_btn_interactive_if_needed


def submit_files(files: list, step: int, chat_history: list[list]):
    files_step = CHATBOT_STEPS[step]
    assert type(files_step) is FilesStep

    # save file names to output variable
    save_to_output_variable(files_step.output_key, [file.name for file in files])

    # iterate over files and print their names
    for file in files: print(f'File uploaded: {file.name.split("/")[-1]}')
    print()

    # update chat history with validation message
    validation_message = f'You successfully uploaded {len(files)} {files_step.kind_of_document}! 🎉'

    return chat_history + [[validation_message, None]]


def stream_chatbot_message_for_next_step_if_needed(user_interaction_type: UserInteractionType, chat_history: list[list], step: int):
    if user_interaction_type not in [UserInteractionType.UPLOAD, UserInteractionType.CLEAR]:
        step += 1
        if 0 <= step < len(CHATBOT_STEPS):
            chat_history += [[CHATBOT_STEPS[step].message, None]]
        else:
            chat_history += [['End of demo, thanks for participating!', None]]

    return chat_history, step


def is_visible_in_current_user_interaction(component_wrapper: ComponentWrapper, step: int):
    # if step is out of bounds, return False to hide component
    if not 0 <= step < len(CHATBOT_STEPS):
        return False

    # if step is in bounds, return True if component's user interaction type matches current step's user interaction type
    return component_wrapper.user_interaction_type in CHATBOT_STEPS[step].user_interaction_types



with gr.Blocks() as demo:
    # create state variable to keep track of current chatbot step
    step_var = gr.State(-1)

    # create components for each component in the chatbot UI

    # create chatbot component
    chatbot = gr.Chatbot(label='AI Grant Writing Coach').style(height=800) # once hf is fixed set height in constructor instead of style

    with gr.Row():
        # start button component
        start_btn = StartWrapper(start_btn=gr.Button(value='Start', variant='primary', visible=True))

        # yes/no button component
        yes_btn_component = gr.Button(value='Yes', variant='primary', visible=False)
        no_btn_component = gr.Button(value='No', variant='stop', visible=False)

        # user text box component
        user_text_box_component = gr.Textbox(label='User', lines=3, visible=False, interactive=True, placeholder='Type your message here')

    with gr.Row():
        with gr.Column():
            # upload button component
            upload_btn_component = gr.UploadButton(label='Upload', variant='primary', visible=False, file_types=['.docx', '.txt'], file_count='multiple', scale=1)

            # submit button component
            submit_btn_component=gr.Button(value='Submit', variant='primary', visible=False, interactive=False, scale=1)

            # clear button component
            clear_btn_component=gr.Button(value='Clear', variant='stop', visible=False, interactive=False, scale=1)
            #clear_btn_component=gr.ClearButton(value='Clear', variant='stop', visible=False, interactive=False, scale=1)

        # files component
        files_component = gr.Files(label='Documents', visible=False, interactive=False, file_types=['.docx', '.txt'], scale=3)

        import json
        clear_btn_component.click(None, [], [files_component], _js=f"() => {json.dumps([files_component.postprocess(None)])}")
        # clear_btn_component.add(files_component) # make the clear button clear the files
    
    with gr.Row():
        # examples component (not a true gradio component in fact)
        examples=gr.Examples(
            examples=[
                'What is your mission?',
                'Give me a background of your organization.',
                'What are your achievements to date?',
                'Where does this project fit within your organizational strategy and vision?',
                'How is your organization building an inclusive workplace culture? What are your diversity, equity, and inclusion goals?',
                'How does the proposed project contribute to the foundation\'s funding priority of increasing diversity, equity, and inclusion (DEI)?',
                'What is your organization\'s approach to measuring impact?',
                'What are your organization\'s goals for the next 3-5 years?',
            ],
            inputs=user_text_box_component,
            label='Examples of grant application questions')

    # create wrappers for each component in the chatbot UI and define their first actions after trigger (i.e. after user interaction)
    create_yes_no_btn = lambda yes_no_btn_component: YesNoWrapper(
        yes_no_btn=yes_no_btn_component,
        first_actions_after_trigger=[{
            'fn': handle_yes_no_interaction,
            'inputs': [yes_no_btn_component, chatbot, step_var],
            'outputs': [chatbot, step_var]
        }])
    yes_btn = create_yes_no_btn(yes_btn_component)
    no_btn = create_yes_no_btn(no_btn_component)

    files = FilesWrapper(
        files=files_component,
        first_actions_after_trigger=[{
            'fn': handle_files_uploaded_from_files,
            'inputs': [files_component, step_var, chatbot],
            'outputs': chatbot
        }])

    upload_btn = UploadWrapper(
        upload_btn=upload_btn_component,
        first_actions_after_trigger=[{
            'fn': handle_files_uploaded_from_btn,
            'inputs': [upload_btn_component, files_component],
            'outputs': [files_component, submit_btn_component, clear_btn_component]
        }])

    submit_btn = SubmitWrapper(
        submit_btn=submit_btn_component,
        first_actions_after_trigger=[{
            'fn': submit_files,
            'inputs': [files_component, step_var, chatbot],
            'outputs': chatbot
        }])

    clear_btn = ClearWrapper(
        clear_btn=clear_btn_component,
        first_actions_after_trigger=[{
            'fn': lambda: [gr.update(interactive=False), gr.update(interactive=False)],
            'outputs': [submit_btn_component, clear_btn_component]
        }])

    user_text_box = TextWrapper(
        text_box=user_text_box_component,
        first_actions_after_trigger=[{
            'fn': handle_user_interaction,
            'inputs': [user_text_box_component, chatbot, step_var],
            'outputs': [user_text_box_component, chatbot]
        }])


    components: list[ComponentWrapper] = [start_btn, yes_btn, no_btn, files, upload_btn, submit_btn, clear_btn, user_text_box]

    for component in components:
        component.get_trigger_to_proceed(
        ).then(
            # update chatbot step if needed and stream chatbot message for next step if needed
            fn=stream_chatbot_message_for_next_step_if_needed,
            inputs=[gr.State(component.user_interaction_type), chatbot, step_var],
            outputs= [chatbot, step_var]
        ).then(
            # update visibility of components based on current chatbot step and user interaction type of component
            fn=lambda step: [gr.update(visible=is_visible_in_current_user_interaction(component, step)) for component in components],
            inputs=step_var,
            outputs=[component.component for component in components]
        )

if __name__ == '__main__':
    demo.launch()
