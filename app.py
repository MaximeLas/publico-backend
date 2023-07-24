import importlib.metadata

import gradio as gr

from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()

from settings import CHATBOT_STEPS
from constants import *
from helpers import *
from prompts import *
from comprehensiveness import *
from chatbot_step import *
from component_wrapper import *
from component_logic import *



# define chatbot steps and their properties
CHATBOT_STEPS += [
    YesNoStep( # 0
        message="Have you applied for this grant before?",
        context_key=ContextKeys.HAS_APPLIED_FOR_THIS_GRANT_BEFORE,
        steps_to_skip_if_no=1),
    FilesStep( # 1
        message="That's very useful! Please upload your {kind_of_document}.",
        context_key=ContextKeys.PRIOR_GRANT_APPLICATIONS,
        kind_of_document="prior grant application(s)"),
    TextStep( # 2
        message="Now, on to the first question! Please let me know what the first application question is, or copy and paste it from the application portal.",
        context_key=ContextKeys.APPLICATION_QUESTION),
    TextStep( # 3
        message="What is the word limit?",
        context_key=ContextKeys.WORD_LIMIT,
        generate_output_fns=[generate_answer_to_question]),
    YesNoStep( # 4
        message="Do you want to check the comprehensiveness of the generated answer?",
        context_key=ContextKeys.CHECK_COMPREHENSIVENESS,
        generate_output_fns=[check_for_comprehensiveness, generate_answers_for_implicit_questions, generate_final_answer])]



with gr.Blocks() as demo:
    with gr.Row():
        # create chatbot component
        if importlib.metadata.version('gradio') < '3.34.0':
            chatbot = gr.Chatbot(label='AI Grant Writing Coach', show_share_button=True).style(height=800)
        else:
            chatbot = gr.Chatbot(label='AI Grant Writing Coach', show_share_button=True, height=800)

    with gr.Row():
        with gr.Column():
            # user text box component
            user_text_box_component = gr.Textbox(label='User', visible=False, interactive=True, lines=3, placeholder='Type your message here')
            # submit text button component
            submit_text_btn_component = gr.Button(value='Submit', variant='primary', visible=False)
        
        with gr.Row(visible=False) as examples_row:
            # examples component (not a true gradio 'Component' techincally)
            examples=gr.Examples(
                examples=GRANT_APPLICATION_QUESTIONS_EXAMPLES,
                inputs=user_text_box_component,
                label='Examples of grant application questions')

    with gr.Row():
        # start button component
        start_btn = StartWrapper(start_btn=gr.Button(value='Start', variant='primary', visible=True))
        # yes/no button component
        yes_btn_component = gr.Button(value='Yes', variant='primary', visible=False)
        no_btn_component = gr.Button(value='No', variant='stop', visible=False)
        # upload button component
        upload_btn_component = gr.UploadButton(label='Upload', variant='primary', visible=False, file_types=['.docx', '.txt'], file_count='multiple')
        # clear button component
        ClearButtonType = gr.Button if importlib.metadata.version('gradio') < '3.35.0' else gr.ClearButton
        clear_btn_component=ClearButtonType(value='Clear', variant='stop', visible=False, interactive=False)
        # submit button component
        submit_btn_component=gr.Button(value='Submit', variant='primary', visible=False, interactive=False)

    with gr.Row() as row:
        # files component
        files_component = gr.Files(label='Documents', visible=False, interactive=False, file_types=['.docx', '.txt'])

    # specify that the clear button should clear the files component
    if type(clear_btn_component) is gr.ClearButton:
        clear_btn_component.add(files_component) # make the clear button clear the files
    else:
        # current workaround until hg supports gradio>=3.35.0 in which ClearButton was introduced
        import json
        clear_btn_component.click(None, [], [files_component], _js=f"() => {json.dumps([files_component.postprocess(None)])}")


    # create state variable to keep track of current chatbot step
    step_var = gr.State(-1)

    # create wrappers for each component in the chatbot UI and define their first actions after trigger (i.e. after user interaction)
    create_yes_no_btn = lambda yes_no_btn_component: YesNoWrapper(
        yes_no_btn=yes_no_btn_component,
        first_actions_after_trigger=[{
            'fn': handle_yes_no_clicked,
            'inputs': [yes_no_btn_component, chatbot, step_var],
            'outputs': [chatbot, step_var]}])
    yes_btn = create_yes_no_btn(yes_btn_component)
    no_btn = create_yes_no_btn(no_btn_component)

    files = FilesWrapper(files=files_component)

    upload_btn = UploadWrapper(
        upload_btn=upload_btn_component,
        first_actions_after_trigger=[{
            'fn': handle_files_uploaded,
            'inputs': [upload_btn_component, files_component],
            'outputs': [files_component, submit_btn_component, clear_btn_component]
        }])

    submit_btn = SubmitWrapper(
        submit_btn=submit_btn_component,
        first_actions_after_trigger=[{
            'fn': handle_files_submitted,
            'inputs': [files_component, chatbot, step_var],
            'outputs': chatbot}])

    clear_btn = ClearWrapper(
        clear_btn=clear_btn_component,
        first_actions_after_trigger=[{
            'fn': lambda: [gr.update(interactive=False), gr.update(interactive=False)],
            'outputs': [submit_btn_component, clear_btn_component]}])

    user_text_box = TextWrapper(
        text_box=user_text_box_component,
        first_actions_after_trigger=[{
            'fn': handle_text_submitted,
            'inputs': [user_text_box_component, chatbot, step_var],
            'outputs': [user_text_box_component, chatbot]}])

    submit_text_btn = SubmitTextButtonWrapper(
        submit_text_btn=submit_text_btn_component,
        first_actions_after_trigger=user_text_box.first_actions_after_trigger)


    components: list[ComponentWrapper] = [start_btn, yes_btn, no_btn, files, upload_btn, submit_btn, clear_btn, user_text_box, submit_text_btn]
    internal_components: list[Component] = [component.component for component in components]

    proceed_to_next_step = lambda user_interaction_type: user_interaction_type not in [UserInteractionType.UPLOAD, UserInteractionType.CLEAR]
    is_visible_in_step = lambda component, step: component.user_interaction_type in CHATBOT_STEPS[step].user_interaction_types if 0 <= step < len(CHATBOT_STEPS) else False
    is_application_question_step = lambda step: 0 <= step < len(CHATBOT_STEPS) and CHATBOT_STEPS[step].context_key == ContextKeys.APPLICATION_QUESTION

    for component in components:
        if component.trigger_to_proceed is None:
            continue

        user_interaction_type = gr.State(component.user_interaction_type)
        component.chain_first_actions_after_trigger(
            # make components invisible if we proceed to next step
            fn=lambda user_interaction_type: [gr.update(visible=False) if proceed_to_next_step(user_interaction_type) else gr.skip()] * len(components),
            inputs=[user_interaction_type],
            outputs=internal_components
        ).then(
            # update chatbot step if needed and stream chatbot message for next step if needed
            fn=lambda user_interaction_type, chatbot, step: stream_next_step_chatbot_message(chatbot, step) if proceed_to_next_step(user_interaction_type) else [gr.skip()] * 2,
            inputs=[user_interaction_type, chatbot, step_var],
            outputs= [chatbot, step_var]
        ).then(
            # update visibility of components based on current chatbot step and user interaction type of component
            fn=lambda step: [gr.update(visible=is_visible_in_step(component, step)) for component in components],
            inputs=step_var,
            outputs=internal_components
        ).then(
            fn=lambda step: gr.update(visible=is_application_question_step(step)),
            inputs=step_var,
            outputs=examples_row # type: ignore
        )


if __name__ == '__main__':
    demo.launch()
