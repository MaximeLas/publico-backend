from collections import defaultdict
from collections.abc import Iterator
from typing import Callable

import gradio as gr
from gradio.blocks import Block

from workflow.app_context import AppContext
from workflow.chatbot_step import ChatbotStep, InitialChatbotMessage
from workflow.step_decider import FixedStepDecider, ConditionalStepDecider, MultiConditionalStepDecider
from message_generation.msg_gen_llm import (
    check_for_comprehensiveness,
    generate_answer_for_implicit_question_stream,
    generate_answer_to_question_stream,
    generate_final_answer_stream,
    generate_improved_answer_following_user_guidance_prompt
)
from message_generation.msg_gen_publico import (
    generate_chatbot_messages,
    generate_validation_message_following_files_upload
)
from configurations.constants import (
    CHATBOT_HEIGHT,
    CHATBOT_LAYOUT,
    DEFAULT_NUM_OF_DOC_CHUNKS,
    DEFAULT_NUM_OF_TOKENS,
    DEFAULT_WORD_LIMIT,
    IS_DEV_MODE,
    ComponentID,
    ComponentLabel,
    StepID,
    SYSTEM_PROMPT_FOR_ANSWERING_ORIGINAL_QUESTION,
    SYSTEM_PROMPT_FOR_ANSWERING_IMPLICIT_QUESTION,
)



class WorkflowState():
    '''
    Represents the state of the chatbot's workflow at a particular point in time.

    The WorkflowState is used as an intermediate structure that can hold and manage the workflow
    context and the current step of the workflow when the application's state is updated
    by Gradio. This is necessary because Gradio flattens complex objects into dictionaries,
    causing issues when trying to directly manipulate instances of classes like WorkflowManager.

    Attributes:
        current_step_id (StepID): The ID of the current step of the chatbot workflow
        current_step (ChatbotStep): The current step of the chatbot workflow
        context (AppContext): The context of the user interacting with the chatbot
    '''

    def __init__(self, start_step: ChatbotStep):
        self.context = AppContext()
        self.current_step_id = StepID.START
        self.current_step = start_step


class WorkflowManager:
    def __init__(self):
        self.components: dict[str, Block] = self.initialize_components()
        self.steps: dict[StepID, ChatbotStep] = self.initialize_steps()


    def get_components_for_step(self, step_id: StepID) ->  dict[ComponentID, Block]:
        return {component_id: self.components[component_id] for component_id in self.steps[step_id].components.keys()}


    def get_component(self, component_id: ComponentID) -> Block:
        return self.components[component_id]


    def get_step(self, step_id: StepID) -> ChatbotStep:
        return self.steps[step_id]


    def initialize_components(self) -> dict[str, Block]:
        return {
            ComponentID.CHATBOT: gr.Chatbot(
                value=[[None, "Welcome! 👋\n" +
                    "I'm Publico, your personal grant writing coach.\n" +
                    "Are you ready to start writing together?"]],
                label=ComponentLabel.CHATBOT,
                height=CHATBOT_HEIGHT,
                bubble_full_width=False,
                avatar_images=(None, 'assets/logo192.png'),
                layout=CHATBOT_LAYOUT
            ),
            ComponentID.USER_TEXT_BOX: gr.Textbox(
                label=ComponentLabel.USER,
                visible=False,
                interactive=True,
                lines=3,
                show_copy_button=True,
                placeholder='Type your message here'
            ),
            ComponentID.NUMBER_1: gr.Number(
                value=DEFAULT_WORD_LIMIT,
                precision=0,
                label=ComponentLabel.WORD_LIMIT,
                visible=False,
                interactive=True
            ),
            ComponentID.NUMBER_2: gr.Number(
                value=0,
                precision=0,
                label=ComponentLabel.NUM_OF_DOCS,
                visible=False,
                interactive=True
            ),
            ComponentID.SUBMIT_USER_INPUT_BTN: gr.Button(
                value=ComponentLabel.SUBMIT,
                variant='primary',
                visible=False
            ),
            ComponentID.BTN_1: gr.Button(
                value=ComponentLabel.START,
                variant='primary'
            ),
            ComponentID.BTN_2: gr.Button(
                visible=False
            ),
            ComponentID.FILES: gr.Files(
                label=ComponentLabel.FILES,
                visible=False,
                interactive=False,
                file_types=['.docx', '.txt']
            ),
            ComponentID.UPLOAD_FILES_BTN: gr.UploadButton(
                label=ComponentLabel.UPLOAD,
                variant='primary',
                visible=False,
                file_types=['.docx', '.txt'],
                file_count='multiple'
            ),
            ComponentID.CLEAR_FILES_BTN: gr.ClearButton(
                value=ComponentLabel.CLEAR,
                variant='stop',
                visible=False,
                interactive=False
            ),
            ComponentID.SUBMIT_FILES_BTN: gr.Button(
                value=ComponentLabel.SUBMIT,
                variant='primary',
                visible=False,
                interactive=False
            ),
            ComponentID.ANSWER_MARKUP: gr.Markdown(
                label=ComponentLabel.ANSWERS,
                line_breaks=True,
                visible=True
            )
        }

    def initialize_steps(self) -> dict[StepID, ChatbotStep]:
        yes_btn_props = dict(value=ComponentLabel.YES, variant='primary')
        no_btn_props = dict(value=ComponentLabel.NO, variant='stop')
        user_props = dict(value='', label=ComponentLabel.USER)
        num_tokens_props = dict(value=DEFAULT_NUM_OF_TOKENS, label=ComponentLabel.NUM_OF_TOKENS)
        num_docs_props = dict(value=DEFAULT_NUM_OF_DOC_CHUNKS, label=ComponentLabel.NUM_OF_DOCS)

        return {
            StepID.START: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Welcome! 👋\n" +
                    "I'm Publico, your personal grant writing coach.\n" +
                    "Are you ready to start writing together?"),
                next_step_decider=FixedStepDecider(StepID.HAVE_MATERIALS_TO_SHARE),
                components={ComponentID.BTN_1: dict(value=ComponentLabel.START, variant='primary')}
            ),
            StepID.HAVE_MATERIALS_TO_SHARE: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Do you have any existing materials you'd like to share (past grants, reports, etc.)?"),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.UPLOAD_FILES),
                    ComponentLabel.NO: FixedStepDecider(StepID.ENTER_QUESTION)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props}
            ),
            StepID.UPLOAD_FILES: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "That's very useful! Please upload your documents below.\n" +
                    "Supported file types: **.docx** & **.txt**"),
                next_step_decider=FixedStepDecider(StepID.ENTER_QUESTION),
                components={
                    ComponentID.FILES: {},
                    ComponentID.UPLOAD_FILES_BTN: {},
                    ComponentID.CLEAR_FILES_BTN: {},
                    ComponentID.SUBMIT_FILES_BTN: {}},
                save_event_outcome_fn=AppContext.set_uploaded_files,
                generate_chatbot_messages_fns=[generate_validation_message_following_files_upload]
            ),
            StepID.ENTER_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Please type the grant application question."),
                next_step_decider=FixedStepDecider(StepID.ENTER_WORD_LIMIT),
                components={
                    ComponentID.USER_TEXT_BOX: user_props,
                    ComponentID.SUBMIT_USER_INPUT_BTN:{}},
                initialize_step_func=AppContext.add_new_question,
                save_event_outcome_fn=AppContext.set_grant_application_question
            ),
            StepID.ENTER_WORD_LIMIT: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "What is the word limit? 🛑"),
                next_step_decider=FixedStepDecider(
                    StepID.GO_OVER_IMPLICIT_QUESTIONS
                        if not IS_DEV_MODE else
                    StepID.ENTER_RAG_CONFIG_ORIGINAL_QUESTION),
                components={
                    ComponentID.NUMBER_1: dict(value=DEFAULT_WORD_LIMIT, label=ComponentLabel.WORD_LIMIT),
                    ComponentID.SUBMIT_USER_INPUT_BTN: {}},
                save_event_outcome_fn=AppContext.set_word_limit,
                generate_chatbot_messages_fns=[
                    generate_answer_to_question_stream, check_for_comprehensiveness]
                        if not IS_DEV_MODE else
                    []
            ),
            StepID.ENTER_RAG_CONFIG_ORIGINAL_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "1) What system prompt should be used to generate the answer?\n" +
                    "2) How many tokens at most should be included in a single document chunk?\n" +
                    "3) How many chunks sould be selected in the similarity check step?"),
                next_step_decider=FixedStepDecider(StepID.GO_BACK_TO_CONFIG_STEP_ORIGINAL_QUESTION),
                components={
                    ComponentID.USER_TEXT_BOX: dict(
                        value=SYSTEM_PROMPT_FOR_ANSWERING_ORIGINAL_QUESTION, label=ComponentLabel.SYSTEM_PROMPT),
                    ComponentID.NUMBER_1: num_tokens_props, ComponentID.NUMBER_2: num_docs_props,
                    ComponentID.SUBMIT_USER_INPUT_BTN: {}},
                save_event_outcome_fn=AppContext.set_test_config_params,
                generate_chatbot_messages_fns=[
                    generate_answer_to_question_stream]
            ),
            StepID.GO_BACK_TO_CONFIG_STEP_ORIGINAL_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Do you want to go back to the previous step (RAG Config)?"),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.ENTER_RAG_CONFIG_ORIGINAL_QUESTION),
                    ComponentLabel.NO: FixedStepDecider(StepID.GO_OVER_IMPLICIT_QUESTIONS)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props},
                generate_chatbot_messages_fns={
                    ComponentLabel.NO: [check_for_comprehensiveness]}
            ),
            StepID.GO_OVER_IMPLICIT_QUESTIONS: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    'Would you like to answer the questions one by one?'),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                    ComponentLabel.NO: FixedStepDecider(StepID.DO_ANOTHER_QUESTION)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props}
            ),
            StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    message="(**{index}**) **{question}**\n\n" +
                        "Does this question address a topic or information that should be incorporated into the final answer?",
                    extract_formatting_variables_func=lambda context: (yield {
                        'question': context.get_next_implicit_question(),
                        'index': context.get_index_of_implicit_question_being_answered()})),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(
                        StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT
                            if not IS_DEV_MODE else
                        StepID.ENTER_RAG_CONFIG_IMPLICIT_QUESTION),
                    ComponentLabel.NO: MultiConditionalStepDecider(
                        conditional_steps=[
                            (AppContext.has_more_implcit_questions_to_answer, StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                            (AppContext.exists_answer_to_any_implicit_question, StepID.READY_TO_GENERATE_FINAL_ANSWER)
                        ],
                        default_next_step=StepID.DO_ANOTHER_QUESTION)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props},
                generate_chatbot_messages_fns=defaultdict(list, {
                    ComponentLabel.YES: [generate_answer_for_implicit_question_stream] if not IS_DEV_MODE else [],
                    ComponentLabel.NO: [lambda context: (
                        "Okay, let's skip this one."
                            if AppContext.has_more_implcit_questions_to_answer(context) or
                                AppContext.exists_answer_to_any_implicit_question(context)
                            else
                        "None of the implicit questions were answered.")]})
            ),
            StepID.ENTER_RAG_CONFIG_IMPLICIT_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "1) What system prompt should be used to generate the answer?\n" +
                    "2) How many tokens at most should be included in a single document chunk?\n" +
                    "3) How many chunks sould be selected in the similarity check step?"),
                next_step_decider=FixedStepDecider(StepID.GO_BACK_TO_CONFIG_STEP_IMPLICIT_QUESTION),
                components={
                    ComponentID.USER_TEXT_BOX: dict(
                        value=SYSTEM_PROMPT_FOR_ANSWERING_IMPLICIT_QUESTION, label=ComponentLabel.SYSTEM_PROMPT),
                    ComponentID.NUMBER_1: num_tokens_props, ComponentID.NUMBER_2: num_docs_props,
                    ComponentID.SUBMIT_USER_INPUT_BTN: {}},
                save_event_outcome_fn=AppContext.set_test_config_params,
                generate_chatbot_messages_fns=[generate_answer_for_implicit_question_stream]
            ),
            StepID.GO_BACK_TO_CONFIG_STEP_IMPLICIT_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Do you want to go back to the previous step (RAG Config)?"),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.ENTER_RAG_CONFIG_IMPLICIT_QUESTION),
                    ComponentLabel.NO: FixedStepDecider(StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props}
            ),
            StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    message="{response}",
                    extract_formatting_variables_func=lambda context: (yield (
                        'Is this helpful?'
                            if context.exists_answer_to_current_implicit_question() else
                        'Would you like to answer it yourself?'))),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.PROMPT_USER_TO_SUBMIT_ANSWER),
                    ComponentLabel.EDIT_IT: FixedStepDecider(StepID.PROMPT_USER_TO_SUBMIT_ANSWER),
                    ComponentLabel.NO: MultiConditionalStepDecider(
                        conditional_steps=[
                            (AppContext.has_more_implcit_questions_to_answer, StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                            (AppContext.exists_answer_to_any_implicit_question, StepID.READY_TO_GENERATE_FINAL_ANSWER)
                        ],
                        default_next_step=StepID.DO_ANOTHER_QUESTION),
                    ComponentLabel.GOOD_AS_IS: ConditionalStepDecider(
                            condition=AppContext.has_more_implcit_questions_to_answer,
                            if_true_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                            if_false_step=StepID.READY_TO_GENERATE_FINAL_ANSWER)},
                components={
                    ComponentID.BTN_1: (lambda context:
                        yes_btn_props
                            if not AppContext.exists_answer_to_current_implicit_question(context)
                            else
                        dict(value=ComponentLabel.GOOD_AS_IS, variant='primary')),
                    ComponentID.BTN_2: (lambda context:
                        no_btn_props
                            if not AppContext.exists_answer_to_current_implicit_question(context)
                            else
                        dict(value=ComponentLabel.EDIT_IT, variant='secondary'))},
                generate_chatbot_messages_fns=defaultdict(list, {
                    ComponentLabel.NO: [lambda context: (
                        "Okay, let's skip this one."
                            if AppContext.has_more_implcit_questions_to_answer(context) or
                                AppContext.exists_answer_to_any_implicit_question(context)
                            else
                        "None of the implicit questions were answered.")],
                    ComponentLabel.GOOD_AS_IS: [lambda _: "Great! We'll use this answer."]})
            ),
            StepID.PROMPT_USER_TO_SUBMIT_ANSWER: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Okay, go ahead and write an answer to the question."),
                next_step_decider=ConditionalStepDecider(
                    condition=AppContext.has_more_implcit_questions_to_answer,
                    if_true_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                    if_false_step=StepID.READY_TO_GENERATE_FINAL_ANSWER),
                components={
                    ComponentID.USER_TEXT_BOX: user_props,
                    ComponentID.SUBMIT_USER_INPUT_BTN: {}},
                save_event_outcome_fn=AppContext.set_answer_to_current_implicit_question,
                generate_chatbot_messages_fns=[lambda context: (
                    'Great! Now that we\'ve answered that question, let\'s move on to the next.'
                        if context.has_more_implcit_questions_to_answer()
                        else 
                    None)]
            ),
            StepID.READY_TO_GENERATE_FINAL_ANSWER: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "We're done with the implicit questions! 🏁\n" +
                    "Are you ready to have your final answer generated?"),
                next_step_decider=FixedStepDecider(StepID.ASK_USER_IF_GUIDANCE_NEEDED),
                components={ComponentID.BTN_1: dict(value=ComponentLabel.OF_COURSE, variant='primary')},
                generate_chatbot_messages_fns=[generate_final_answer_stream]
            ),
            StepID.ASK_USER_IF_GUIDANCE_NEEDED: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "If you'd like, you can give me guidance to improve the answer."),
                next_step_decider={
                    ComponentLabel.GOOD_AS_IS: FixedStepDecider(StepID.DO_ANOTHER_QUESTION),
                    ComponentLabel.ADD_GUIDANCE: FixedStepDecider(StepID.USER_GUIDANCE_PROMPT)},
                components={
                    ComponentID.BTN_1: dict(value=ComponentLabel.GOOD_AS_IS, variant='primary'),
                    ComponentID.BTN_2: dict(value=ComponentLabel.ADD_GUIDANCE, variant='secondary')}
            ),
            StepID.USER_GUIDANCE_PROMPT: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "What guidance do you want to provide to improve the answer? " +
                    "(ie. *Make it more formal* or *Mention that we helped 150 clients last year*)"),
                next_step_decider=ConditionalStepDecider(
                    condition=AppContext.is_allowed_to_add_more_guidance,
                    if_true_step=StepID.ASK_USER_IF_GUIDANCE_NEEDED,
                    if_false_step=StepID.DO_ANOTHER_QUESTION),
                components={
                    ComponentID.USER_TEXT_BOX: user_props,
                    ComponentID.SUBMIT_USER_INPUT_BTN: {}},
                save_event_outcome_fn=AppContext.set_user_guidance_prompt,
                generate_chatbot_messages_fns=[generate_improved_answer_following_user_guidance_prompt]
            ),
            StepID.DO_ANOTHER_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Do you want to generate an answer for another question? 🔄"),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.ENTER_QUESTION),
                    ComponentLabel.NO: FixedStepDecider(StepID.END)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props}
            ),
            StepID.END: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "End of demo, thanks for participating! 👏"),
                next_step_decider=FixedStepDecider(StepID.END)
            )
        }


def update_visibility_of_components_in_current_step(
    all_components: dict[ComponentID, Block],
    workflow_state: WorkflowState,
    is_visible: bool
) -> dict[Block, dict]:
    '''Update visibility of components based on current step'''

    return {
        all_components[component_id]: gr.update(visible=is_visible, **(
            properties
                if not isinstance(properties, Callable)
                else
            properties(workflow_state.context))
        ) for component_id, properties in workflow_state.current_step.components.items()}


def update_workflow_step(
    steps: dict[StepID, ChatbotStep],
    workflow_state: WorkflowState,
    component_name: str
):
    '''Update the workflow step based on the component that triggered the event'''

    next_step_id = workflow_state.current_step.determine_next_step(component_name, workflow_state.context)
    workflow_state.current_step_id = next_step_id
    workflow_state.current_step = steps[next_step_id]


def modify_context(workflow_state: WorkflowState):
    '''Update the workflow context based on the current step'''

    current_step = workflow_state.current_step
    current_step.initialize_step_func(workflow_state.context)


def get_initial_chatbot_message(workflow_state: WorkflowState) -> Iterator[list[tuple[str, None]]]:
    '''Append the initial message of the current step to the chat history'''

    for chatbot_message in workflow_state.current_step.initial_chatbot_message.get_formatted_message(workflow_state.context):
        yield [[None, chatbot_message]]


def generate_chatbot_messages_from_trigger(
    workflow_state: WorkflowState,
    component_name: str
) -> Iterator[list[tuple[str, None]]]:
    '''Generate chatbot messages based on component trigger'''

    if (fns := workflow_state.current_step.get_generate_chatbot_messages_fns_for_trigger(component_name)):
        yield from generate_chatbot_messages(fns=fns, context=workflow_state.context)
