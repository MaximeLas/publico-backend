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
    handle_text_submitted
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


application_question_step = lambda: next(
    (step for step, chatbot_step_object in enumerate(CHATBOT_STEPS) if chatbot_step_object.context_key == ContextKeys.APPLICATION_QUESTION))


# define chatbot steps and their properties
CHATBOT_STEPS += [
    YesNoStep( # 0
        initial_message="Have you applied for this grant before?",
        context_key=ContextKeys.HAS_APPLIED_FOR_THIS_GRANT_BEFORE,
        next_step_if_no=lambda step: step + 1),
    FilesStep( # 1
        initial_message="That's very useful! Please upload your {kind_of_document}.",
        context_key=ContextKeys.PRIOR_GRANT_APPLICATIONS,
        kind_of_document="prior grant application(s)",
        generate_message_fns=[generate_validation_message_following_files_upload]),
    TextStep( # 2
        initial_message="Please type the grant application question, or copy and paste it from the application portal.",
        context_key=ContextKeys.APPLICATION_QUESTION),
    TextStep( # 3
        initial_message="What is the word limit?",
        context_key=ContextKeys.WORD_LIMIT,
        generate_message_fns=[generate_answer_to_question_stream]),
    YesNoStep( # 4
        initial_message="Do you want to check the comprehensiveness of the generated answer?",
        context_key=ContextKeys.CHECK_COMPREHENSIVENESS,
        generate_message_fns=[
            check_for_comprehensiveness,
            generate_answers_for_implicit_questions_stream,
            generate_final_answer_stream]),
    YesNoStep( # 5
        initial_message="Do you want to generate an answer for another question?",
        context_key=ContextKeys.TRY_WITH_ANOTHER_QUESTION,
        next_step_if_yes=lambda _: application_question_step())]



with gr.Blocks() as demo:
    with gr.Row():
        # create chatbot component
        if importlib.metadata.version('gradio') < '3.34.0':
            chatbot = gr.Chatbot(label='AI Grant Writing Coach', show_share_button=True).style(height=650)
        else:
            chatbot = gr.Chatbot(label='AI Grant Writing Coach', show_share_button=True, height=650)

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
            'outputs': [chatbot, context_var]}])
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


    for component in components:
        if component.user_action is None:
            continue

        chain = component.chain_first_actions_after_trigger(
            # if we proceed then we make all components invisible until we show the next step's initial message
            fn=lambda component_wrapper: (
                [gr.update(visible=False) if component_wrapper.proceed_to_next_step else gr.skip()] * len(components)
            ),
            inputs=gr.State(component),
            outputs=internal_components
        )

        if not component.proceed_to_next_step:
             continue

        chain.then(
            # generate chatbot messages for current step by streaming the messages from a generator function, if any
            fn=lambda chat_history, step, context: (
                (yield from CHATBOT_STEPS[step].generate_chatbot_message(chat_history, context))
                    if step > 0 else
                (yield chat_history)
            ),
            inputs=[chatbot, step_var, context_var],
            outputs=chatbot
        ).then(
            # update chatbot step by incrementing the step counter or by going to a specific step
            fn=lambda component, step: (
                # if the component is not a yes/no button, then simply increment the step counter
                step + 1
                    if step == -1 or type(CHATBOT_STEPS[step]) is not YesNoStep else
                # if the component is a yes/no button, then go to the next step according to the logic of the yes/no button for this step
                CHATBOT_STEPS[step].go_to_step(component, step)
            ),
            inputs=[component.component, step_var],
            outputs=step_var
        ).then(
            # if we proceed then we show next step's initial msg, if we're done then show a final message + reset step counter & context
            fn=lambda chat_history, step: (
                # if there are more steps, then stream the next step's message
                [chat_history + [[CHATBOT_STEPS[step].initial_message, None]], gr.skip(), gr.skip()]
                    if step < len(CHATBOT_STEPS) else
                # if there are no more steps, then end the chat and reset the step counter and context
                [chat_history + [['End of demo, thanks for participating!', None]], -1, {}]
            ),
            inputs=[chatbot, step_var],
            outputs= [chatbot, step_var, context_var]
        ).then(
            # update visibility of components based on current chatbot step and user interaction type of component
            fn=lambda step: (
                [gr.update(visible=(
                    0 <= step < len(CHATBOT_STEPS) and
                    component.user_interaction_type in CHATBOT_STEPS[step].user_interaction_types)
                ) for component in components]
            ),
            inputs=step_var,
            outputs=internal_components
        ).then(
            # update visibility of the examples (via its parent row) based on whether we're at the grant application question step
            fn=lambda step: gr.update(visible=step==application_question_step()),
            inputs=step_var,
            outputs=examples_row # type: ignore
        )


if __name__ == '__main__':
    demo.queue().launch()
