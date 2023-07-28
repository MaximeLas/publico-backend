from collections import defaultdict
import importlib.metadata
from typing import Any

import gradio as gr
from gradio.components import Component, IOComponent
from gradio.blocks import Block

# initialize langchain llm cache
from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()

from settings import CHATBOT_STEPS
from chatbot_step import ChatbotStep
from constants import StepID, GRANT_APPLICATION_QUESTIONS_EXAMPLES
from context import (
    UserContext,
    set_do_check_for_comprehensiveness,
    set_grant_application_question,
    set_prior_grant_applications,
    set_word_limit,
    set_current_step_id
)
from component_logic import (
    handle_number_submitted,
    handle_yes_no_clicked,
    handle_files_uploaded,
    handle_files_submitted,
    handle_text_submitted
)
from component_wrapper import (
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
from message_generator_llm import (
    generate_answer_to_question_stream,
    check_for_comprehensiveness,
    generate_answers_for_implicit_questions_stream,
    generate_final_answer_stream
)
from message_generator_publico import generate_chatbot_messages, generate_validation_message_following_files_upload



def get_current_step(context: UserContext) -> ChatbotStep:
    '''Get the current step based on the current step id in the context.'''

    return CHATBOT_STEPS[context.current_step_id]


def store_event_value_in_context(
    context: UserContext,
    default_value: Any,
    components: list[IOComponent],
    *components_values: Any,
) -> UserContext:
    '''Store the value of an event in the context, if a store_in_context_fn is defined for the current step.'''

    step = get_current_step(context)

    if step.store_in_context_fn is None:
        return context

    fn, component_name = step.store_in_context_fn

    # get index of element in components that has the same name as component_name, return None if not found
    index_component = next(
        (i for i, c in enumerate(components) if (c.label or c.value) == component_name),
        None)

    print(f'index_component: {index_component}\ncomponent_name: {component_name}\n' +
          f'components_values: {[f"{i}={c}" for i, c in enumerate(components_values)]}\n' +
          f'component_names={[f"{i}={c.label or c.value}" for i, c in enumerate(components)]}\n' +
          f'and will use {components_values[index_component] if index_component is not None else default_value}')
    return (
        fn(context, components_values[index_component])
            if (component_name and index_component is not None)
            else
        fn(context, default_value))


# define chatbot steps and their properties
CHATBOT_STEPS.update({
    StepID.START: ChatbotStep(
        initial_message="Hello there, please hit 'Start' when you're ready.",
        next_step=StepID.HAVE_YOU_APPLIED_BEFORE
    ),
    StepID.HAVE_YOU_APPLIED_BEFORE: ChatbotStep(
        initial_message="Have you applied for this grant before?",
        next_step=dict(
            Yes=StepID.UPLOAD_PRIOR_GRANT_APPLICATIONS,
            No=StepID.ENTER_QUESTION)
    ),
    StepID.UPLOAD_PRIOR_GRANT_APPLICATIONS: ChatbotStep(
        initial_message="That's very useful! Please upload your prior grant application(s).",
        next_step=StepID.ENTER_QUESTION,
        store_in_context_fn=(set_prior_grant_applications, 'Documents'),
        generate_chatbot_messages_fns=[generate_validation_message_following_files_upload]
    ),
    StepID.ENTER_QUESTION: ChatbotStep(
        initial_message="Please type the grant application question, or copy and paste it from the application portal.",
        next_step=StepID.ENTER_WORD_LIMIT,
        store_in_context_fn=(set_grant_application_question, 'User')
    ),
    StepID.ENTER_WORD_LIMIT: ChatbotStep(
        initial_message="What is the word limit?",
        next_step=StepID.DO_COMPREHENSIVENESS_CHECK,
        store_in_context_fn=(set_word_limit, 'Number'),
        generate_chatbot_messages_fns=[generate_answer_to_question_stream]
    ),
    StepID.DO_COMPREHENSIVENESS_CHECK: ChatbotStep(
        initial_message="Do you want to check the comprehensiveness of the generated answer?",
        next_step=dict(Yes=StepID.DO_ANOTHER_QUESTION, No=StepID.DO_ANOTHER_QUESTION),
        store_in_context_fn=(set_do_check_for_comprehensiveness, None),
        generate_chatbot_messages_fns=defaultdict(list,
            Yes=[check_for_comprehensiveness, generate_answers_for_implicit_questions_stream, generate_final_answer_stream])
    ),
    StepID.DO_ANOTHER_QUESTION: ChatbotStep(
        initial_message="Do you want to generate an answer for another question?",
        next_step=dict(
            Yes=StepID.ENTER_QUESTION,
            No=StepID.END
        )
    ),
    StepID.END: ChatbotStep(
        initial_message='End of demo, thanks for participating!',
        next_step=StepID.END
    )
})



with gr.Blocks() as demo:
    with gr.Row():
        # create chatbot component
        chatbot = (gr.Chatbot(label='AI Grant Writing Coach', show_share_button=True, height=650))

    with gr.Row():
        with gr.Column():
            # user text box component
            user_text_box_component = gr.Textbox(label='User', visible=False, interactive=True, lines=3, show_copy_button=True, placeholder='Type your message here')
            # submit text button component
            submit_text_btn_component = gr.Button(value='Submit', variant='primary', visible=False)
            # number component
            number_component = gr.Number(value=30, precision=0, label='Number', visible=False, interactive=True)
            # submit number component
            submit_number_btn_component = gr.Button(value='Submit', variant='primary', visible=False)

        
        with gr.Row(visible=False) as examples_row:
            # examples component (not a true gradio 'Component' techincally)
            examples=gr.Examples(
                examples=GRANT_APPLICATION_QUESTIONS_EXAMPLES,
                inputs=user_text_box_component,
                label='Examples of grant application questions')

    with gr.Row():
        # start button component
        start_btn_component = gr.Button(value='Start', variant='primary', visible=True)
        # yes/no button component
        yes_btn_component = gr.Button(value='Yes', variant='primary', visible=False)
        no_btn_component = gr.Button(value='No', variant='stop', visible=False)
        # upload button component
        upload_btn_component = gr.UploadButton(label='Upload', variant='primary', visible=False, file_types=['.docx', '.txt'], file_count='multiple')
        # clear button component
        clear_btn_component=gr.ClearButton(value='Clear', variant='stop', visible=False, interactive=False)
        # submit button component
        submit_btn_component=gr.Button(value='Submit', variant='primary', visible=False, interactive=False)

    with gr.Row() as row:
        # files component
        files_component = gr.Files(label='Documents', visible=False, interactive=False, file_types=['.docx', '.txt'])

    # specify that the clear button should clear the files component
    clear_btn_component.add(files_component) # make the clear button clear the files

    CHATBOT_STEPS[StepID.START].components = [start_btn_component]
    CHATBOT_STEPS[StepID.HAVE_YOU_APPLIED_BEFORE].components = [yes_btn_component, no_btn_component]
    CHATBOT_STEPS[StepID.UPLOAD_PRIOR_GRANT_APPLICATIONS].components = [upload_btn_component, files_component, submit_btn_component, clear_btn_component]
    CHATBOT_STEPS[StepID.ENTER_QUESTION].components = [user_text_box_component, submit_text_btn_component, examples_row]
    CHATBOT_STEPS[StepID.ENTER_WORD_LIMIT].components = [number_component, submit_number_btn_component]
    CHATBOT_STEPS[StepID.DO_COMPREHENSIVENESS_CHECK].components = [yes_btn_component, no_btn_component]
    CHATBOT_STEPS[StepID.DO_ANOTHER_QUESTION].components = [yes_btn_component, no_btn_component]


    # create wrappers for each component and define the actions to be executed after being triggered, if any
    start_btn = StartWrapper(component=start_btn_component)

    create_yes_no_btn = lambda yes_no_btn_component: YesNoWrapper(
        component=yes_no_btn_component,
        handle_user_action={
            'fn': handle_yes_no_clicked,
            'inputs': [yes_no_btn_component, chatbot],
            'outputs': [chatbot]})
    yes_btn = create_yes_no_btn(yes_btn_component)
    no_btn = create_yes_no_btn(no_btn_component)

    files = FilesWrapper(component=files_component)

    upload_btn = UploadWrapper(
        component=upload_btn_component,
        handle_user_action={
            'fn': handle_files_uploaded,
            'inputs': [upload_btn_component, files_component],
            'outputs': [files_component, submit_btn_component, clear_btn_component]
        })

    submit_btn = SubmitWrapper(
        component=submit_btn_component,
        handle_user_action={
            'fn': handle_files_submitted,
            'inputs': [files_component]
        })

    clear_btn = ClearWrapper(
        component=clear_btn_component,
        handle_user_action={
            'fn': lambda: [gr.update(interactive=False), gr.update(interactive=False)],
            'outputs': [submit_btn_component, clear_btn_component]})

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

    # create state variable to keep track of current context as well as the current chatbot step
    context = gr.State(UserContext())

    components: list[ComponentWrapper] = [
        start_btn, yes_btn, no_btn, files, upload_btn, submit_btn, clear_btn, user_text_box, submit_text_btn, number, submit_number_btn]

    internal_components: list[Component] = [component.component for component in components]
    internal_components_with_row: list[Component | gr.Row] = internal_components + [examples_row]


    for c in components:
        if c.user_action is None:
            continue

        chain = c.get_initial_chain_following_trigger(
        ).then(
            # update visibility of components based on current chatbot step and user interaction type of component
            fn=lambda proceed: [gr.update(visible=False) if proceed else gr.skip() for _ in internal_components_with_row],
            inputs=gr.State(c.proceed_to_next_step),
            outputs=internal_components_with_row # type: ignore
        )

        if c.proceed_to_next_step:
            # if we proceed then we store the value of the relevant component in the context, if defined
            # for the current step (e.g. store the user's reply to a question)
            chain = chain.then(
                fn=store_event_value_in_context,
                inputs=[context, c.component, gr.State(internal_components), *internal_components],
                outputs=context,
            )

        if c.handle_user_action is not None:
            # handle user action as defined by the component wrapper (e.g. upload files, submit text)
            chain = chain.then(**c.handle_user_action)

        if not c.proceed_to_next_step:
            continue

        chain.then(
            # generate any chatbot messages for current step (e.g. validation message following files upload, llm answer to question)
            fn=lambda component, chat_history, context: (
                yield from generate_chatbot_messages(
                    fns=get_current_step(context).determine_generate_chatbot_messages_fns(component),
                    chat_history=chat_history,
                    context=context
                )
            ),
            inputs=[c.component, chatbot, context],
            outputs=chatbot
        ).then(
            # update chatbot step according to the next step id determined by the current step and the user action (e.g. click yes/no)
            fn=lambda component, context: set_current_step_id(context, get_current_step(context).determine_next_step(component)),
            inputs=[c.component, context],
            outputs=context
        ).then(
            # show the initial (chatbot) message of the next step
            fn=lambda chat_history, context: chat_history + [[get_current_step(context).initial_message, None]],
            inputs=[chatbot, context],
            outputs= chatbot
        ).then(
            # update visibility of components based on whether they are defined for the next step (e.g. show yes/no buttons if defined for next step)
            fn=lambda context: [gr.update(visible=(c in get_current_step(context).components)) for c in internal_components_with_row],
            inputs=context,
            outputs=internal_components_with_row # type: ignore
        )


if __name__ == '__main__':
    demo.queue().launch()
