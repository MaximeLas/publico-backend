import importlib.metadata

import gradio as gr
from gradio.components import Component

# initialize langchain llm cache
from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()

from settings import CHATBOT_STEPS
from chatbot_step import YesNoStep, FilesStep, TextStep
from constants import ContextKeys, GRANT_APPLICATION_QUESTIONS_EXAMPLES
from component_logic import (
    handle_yes_no_clicked,
    handle_files_uploaded,
    handle_files_submitted,
    handle_text_submitted,
    stream_next_step_chatbot_message
)
from component_wrapper import (
    ComponentWrapper,
    StartWrapper,
    YesNoWrapper,
    FilesWrapper,
    UploadWrapper,
    SubmitWrapper,
    ClearWrapper,
    TextWrapper,
    SubmitTextButtonWrapper
)
from message_generator_llm import (
    generate_answer_to_question_stream,
    check_for_comprehensiveness,
    generate_answers_for_implicit_questions_stream,
    generate_final_answer_stream
)
from message_generator_publico import generate_validation_message_following_files_upload



# define chatbot steps and their properties
CHATBOT_STEPS += [
    YesNoStep( # 0
        message="Have you applied for this grant before?",
        context_key=ContextKeys.HAS_APPLIED_FOR_THIS_GRANT_BEFORE,
        next_step_if_no=lambda step: step + 1),
    FilesStep( # 1
        message="That's very useful! Please upload your {kind_of_document}.",
        context_key=ContextKeys.PRIOR_GRANT_APPLICATIONS,
        kind_of_document="prior grant application(s)",
        generate_message_fns=[generate_validation_message_following_files_upload]),
    TextStep( # 2
        message="Please type the grant application question, or copy and paste it from the application portal.",
        context_key=ContextKeys.APPLICATION_QUESTION),
    TextStep( # 3
        message="What is the word limit?",
        context_key=ContextKeys.WORD_LIMIT,
        generate_message_fns=[generate_answer_to_question_stream]),
    YesNoStep( # 4
        message="Do you want to check the comprehensiveness of the generated answer?",
        context_key=ContextKeys.CHECK_COMPREHENSIVENESS,
        generate_message_fns=[
            check_for_comprehensiveness,
            generate_answers_for_implicit_questions_stream,
            generate_final_answer_stream]),
    YesNoStep( # 5
        message="Do you want to generate an answer for another question?",
        context_key=ContextKeys.TRY_WITH_ANOTHER_QUESTION,
        next_step_if_yes=lambda step: 1)]



with gr.Blocks() as demo:
    with gr.Row():
        # create chatbot component
        if importlib.metadata.version('gradio') < '3.34.0':
            chatbot = gr.Chatbot(label='AI Grant Writing Coach', show_share_button=True).style(height=620)
        else:
            chatbot = gr.Chatbot(label='AI Grant Writing Coach', show_share_button=True, height=620)

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
    if type(clear_btn_component) is gr.Button:
        # current workaround until hg supports gradio>=3.35.0 in which ClearButton was introduced
        import json
        clear_btn_component.click(None, [], [files_component], _js=f"() => {json.dumps([files_component.postprocess(None)])}")
    else:
        assert type(clear_btn_component) is gr.ClearButton
        clear_btn_component.add(files_component) # make the clear button clear the files


    # define dict to store context variables from chatbot steps
    context: dict[ContextKeys, str | list[str] | None] = {}
    # create state variable to keep track of current context as well as the current chatbot step
    context_var = gr.State(context)
    step_var = gr.State(-1)

    # create wrappers for each component in the chatbot UI and define their first actions after trigger (i.e. after user interaction)
    create_yes_no_btn = lambda yes_no_btn_component: YesNoWrapper(
        yes_no_btn=yes_no_btn_component,
        first_actions_after_trigger=[{
            'fn': handle_yes_no_clicked,
            'inputs': [yes_no_btn_component, chatbot, step_var, context_var],
            'outputs': [chatbot, step_var, context_var]}])
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
            'inputs': [files_component, step_var, context_var],
            'outputs': context_var}])

    clear_btn = ClearWrapper(
        clear_btn=clear_btn_component,
        first_actions_after_trigger=[{
            'fn': lambda: [gr.update(interactive=False), gr.update(interactive=False)],
            'outputs': [submit_btn_component, clear_btn_component]}])

    user_text_box = TextWrapper(
        text_box=user_text_box_component,
        first_actions_after_trigger=[{
            'fn': handle_text_submitted,
            'inputs': [user_text_box_component, chatbot, step_var, context_var],
            'outputs': [user_text_box_component, chatbot, context_var]}])

    submit_text_btn = SubmitTextButtonWrapper(
        submit_text_btn=submit_text_btn_component,
        first_actions_after_trigger=user_text_box.first_actions_after_trigger)


    components: list[ComponentWrapper] = [start_btn, yes_btn, no_btn, files, upload_btn, submit_btn, clear_btn, user_text_box, submit_text_btn]
    internal_components: list[Component] = [component.component for component in components]

    is_visible_in_step = lambda component, step: component.user_interaction_type in CHATBOT_STEPS[step].user_interaction_types if 0 <= step < len(CHATBOT_STEPS) else False
    is_application_question_step = lambda step: 0 <= step < len(CHATBOT_STEPS) and CHATBOT_STEPS[step].context_key == ContextKeys.APPLICATION_QUESTION

    for component in components:
        if component.user_action is None:
            continue

        component_var = gr.State(component)
        component.chain_first_actions_after_trigger(
            # make components invisible if we proceed to next step
            fn=lambda component: [gr.update(visible=False) if component.proceed_to_next_step else gr.skip()] * len(components),
            inputs=component_var,
            outputs=internal_components
        ).then(
            # generate chatbot message for current step by streaming the messages from a generator function, if any
            fn=lambda component, chat_history, step, context: (
                (yield from CHATBOT_STEPS[step].generate_chatbot_message(chat_history, context))
                    if component.proceed_to_next_step and step > 0 and CHATBOT_STEPS[step].generate_message_fns != [] else
                (yield chat_history)
            ),
            inputs=[component_var, chatbot, step_var, context_var],
            outputs=chatbot
        ).then(
            # update chatbot step if needed and stream chatbot message for next step if needed
            fn=lambda component, chat_history, step: (
                stream_next_step_chatbot_message(chat_history, step)
                    if component.proceed_to_next_step else
                [gr.skip()] * 3
            ),
            inputs=[component_var, chatbot, step_var],
            outputs= [chatbot, step_var, context_var]
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
    demo.queue().launch()
