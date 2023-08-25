from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
import time
from typing import Any

import gradio as gr
from gradio.blocks import Block
from gradio.components import IOComponent

from chatbot_step import ChatbotStep, EventOutcomeSaver, InitialChatbotMessage
from constants import DEFAULT_NUMBER, ComponentID, ComponentLabel, StepID
from context import UserContext
from message_generator_llm import (
    check_for_comprehensiveness,
    generate_answer_for_implicit_question_stream,
    generate_answer_to_question_stream,
    generate_final_answer_stream
)
from message_generator_publico import (
    generate_chatbot_messages,
    generate_validation_message_following_files_upload
)
from step_decider import FixedStepDecider, ConditionalStepDecider, MultiConditionalStepDecider



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
        context (UserContext): The context of the user interacting with the chatbot
    '''

    def __init__(self, start_step: ChatbotStep):
        self.context = UserContext()
        self.current_step_id = StepID.START
        self.current_step = start_step


class WorkflowManager:
    def __init__(self):
        self.components: dict[str, Block] = self.initialize_components()
        self.steps: dict[StepID, ChatbotStep] = self.initialize_steps()


    def get_component(self, component_id: ComponentID) -> IOComponent:
        return self.components[component_id]


    def get_step(self, step_id: StepID) -> ChatbotStep:
        return self.steps[step_id]


    def initialize_components(self) -> dict[str, Block]:
        return {
            ComponentID.CHATBOT: gr.Chatbot(
                value=[["Hello there, please hit **Start** when you're ready.", None]],
                label=ComponentLabel.CHATBOT,
                show_share_button=True,
                show_copy_button=True,
                height=600
            ),
            ComponentID.USER_TEXT_BOX: gr.Textbox(
                label=ComponentLabel.USER,
                visible=False,
                interactive=True,
                lines=3,
                show_copy_button=True,
                placeholder='Type your message here'
            ),
            ComponentID.NUMBER: gr.Number(
                value=DEFAULT_NUMBER,
                precision=0,
                label=ComponentLabel.NUMBER,
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
            ComponentID.BTN_3: gr.Button(
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
            )
        }

    def initialize_steps(self) -> dict[StepID, ChatbotStep]:
        yes_btn_props = dict(value=ComponentLabel.YES, variant='primary')
        no_btn_props = dict(value=ComponentLabel.NO, variant='stop')
        return {
            StepID.START: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Hello there, please hit **Start** when you're ready."),
                next_step_decider=FixedStepDecider(StepID.HAVE_YOU_APPLIED_BEFORE),
                components={ComponentID.BTN_1: dict(value=ComponentLabel.START, variant='primary')}
            ),
            StepID.HAVE_YOU_APPLIED_BEFORE: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Have you applied for this grant before?"),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.UPLOAD_FILES),
                    ComponentLabel.NO: FixedStepDecider(StepID.ENTER_QUESTION)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props}
            ),
            StepID.UPLOAD_FILES: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "That's very useful! Please upload your documents.\n" +
                    "Supported file types: **.docx** & **.txt**"),
                next_step_decider=FixedStepDecider(StepID.ENTER_QUESTION),
                components={
                    ComponentID.FILES: {},
                    ComponentID.UPLOAD_FILES_BTN: {},
                    ComponentID.CLEAR_FILES_BTN: {},
                    ComponentID.SUBMIT_FILES_BTN: {}},
                save_event_outcome=EventOutcomeSaver(UserContext.set_uploaded_files, ComponentLabel.FILES),
                generate_chatbot_messages_fns=[generate_validation_message_following_files_upload]
            ),
            StepID.ENTER_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Please type the grant application question."),
                next_step_decider=FixedStepDecider(StepID.ENTER_WORD_LIMIT),
                components={
                    ComponentID.USER_TEXT_BOX: {},
                    ComponentID.SUBMIT_USER_INPUT_BTN:{}},
                initialize_step_func=UserContext.add_new_question,
                save_event_outcome=EventOutcomeSaver(UserContext.set_grant_application_question, ComponentLabel.USER)
            ),
            StepID.ENTER_WORD_LIMIT: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "What is the word limit?"),
                next_step_decider=FixedStepDecider(StepID.DO_COMPREHENSIVENESS_CHECK),
                components={ComponentID.NUMBER: {}, ComponentID.SUBMIT_USER_INPUT_BTN: {}},
                save_event_outcome=EventOutcomeSaver(UserContext.set_word_limit, ComponentLabel.NUMBER),
                generate_chatbot_messages_fns=[
                    generate_answer_to_question_stream]
            ),
            StepID.DO_COMPREHENSIVENESS_CHECK: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Would you like me to review the answer to make sure it includes everything needed?"),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                    ComponentLabel.NO: FixedStepDecider(StepID.DO_ANOTHER_QUESTION)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props},
                save_event_outcome=EventOutcomeSaver(UserContext.set_do_check_for_comprehensiveness, None),
                generate_chatbot_messages_fns=defaultdict(list, {
                    ComponentLabel.YES: [check_for_comprehensiveness]})
            ),
            StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    message="(**{index}**) **{question}**\n\n" +
                        "Does this question address a topic or information that should be included?",
                    extract_formatting_variables_func=lambda context: (yield {
                        'question': context.get_next_implicit_question(),
                        'index': context.get_index_of_implicit_question_being_answered()})),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT),
                    ComponentLabel.NO: MultiConditionalStepDecider(
                        conditional_steps=[
                            (UserContext.has_more_implcit_questions_to_answer, StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                            (UserContext.exists_answer_to_any_implicit_question, StepID.READY_TO_GENERATE_FINAL_ANSWER)
                        ],
                        default_next_step=StepID.DO_ANOTHER_QUESTION)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props},
                generate_chatbot_messages_fns=defaultdict(list, {
                    ComponentLabel.NO: [lambda context: (
                        "Okay, let's skip this one."
                            if UserContext.has_more_implcit_questions_to_answer(context) or
                                UserContext.exists_answer_to_any_implicit_question(context)
                            else
                        "None of the implicit questions were answered.")]})
            ),
            StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    message="{response}",
                    extract_formatting_variables_func=generate_answer_for_implicit_question_stream),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.PROMPT_USER_TO_SUBMIT_ANSWER),
                    ComponentLabel.EDIT_IT: FixedStepDecider(StepID.PROMPT_USER_TO_SUBMIT_ANSWER),
                    ComponentLabel.NO: MultiConditionalStepDecider(
                        conditional_steps=[
                            (UserContext.has_more_implcit_questions_to_answer, StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                            (UserContext.exists_answer_to_any_implicit_question, StepID.READY_TO_GENERATE_FINAL_ANSWER)
                        ],
                        default_next_step=StepID.DO_ANOTHER_QUESTION),
                    ComponentLabel.GOOD_AS_IS: ConditionalStepDecider(
                            condition=UserContext.has_more_implcit_questions_to_answer,
                            if_true_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                            if_false_step=StepID.READY_TO_GENERATE_FINAL_ANSWER)},
                components={
                    ComponentID.BTN_1: (lambda context:
                        yes_btn_props
                            if not UserContext.exists_answer_to_current_implicit_question(context)
                            else
                        dict(value=ComponentLabel.GOOD_AS_IS, variant='primary')),
                    ComponentID.BTN_2: (lambda context:
                        no_btn_props
                            if not UserContext.exists_answer_to_current_implicit_question(context)
                            else
                        dict(value=ComponentLabel.EDIT_IT, variant='primary'))},
                generate_chatbot_messages_fns=defaultdict(list, {
                    ComponentLabel.NO: [lambda context: (
                        "Okay, let's skip this one."
                            if UserContext.has_more_implcit_questions_to_answer(context) or
                                UserContext.exists_answer_to_any_implicit_question(context)
                            else
                        "None of the implicit questions were answered.")],
                    ComponentLabel.GOOD_AS_IS: [lambda _: "Great! We'll use this answer."]})
            ),
            StepID.PROMPT_USER_TO_SUBMIT_ANSWER: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Okay, go ahead and write an answer to the question."),
                next_step_decider=ConditionalStepDecider(
                    condition=UserContext.has_more_implcit_questions_to_answer,
                    if_true_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                    if_false_step=StepID.READY_TO_GENERATE_FINAL_ANSWER),
                components={ComponentID.USER_TEXT_BOX: {}, ComponentID.SUBMIT_USER_INPUT_BTN: {}},
                save_event_outcome=EventOutcomeSaver(UserContext.set_answer_to_current_implicit_question, ComponentLabel.USER),
                generate_chatbot_messages_fns=[lambda context: (
                    'Great! Now that we\'ve answered that question, let\'s move on to the next.'
                        if context.has_more_implcit_questions_to_answer()
                        else 
                    None)]
            ),
            StepID.READY_TO_GENERATE_FINAL_ANSWER: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "We're done with the implicit questions!\n" +
                    "Are you ready to have your final answer generated?"),
                next_step_decider=FixedStepDecider(StepID.DO_ANOTHER_QUESTION),
                components={ComponentID.BTN_1: dict(value=ComponentLabel.OF_COURSE, variant='primary')},
                generate_chatbot_messages_fns=[generate_final_answer_stream]
            ),
            StepID.DO_ANOTHER_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Do you want to generate an answer for another question?"),
                next_step_decider={
                    ComponentLabel.YES: FixedStepDecider(StepID.ENTER_QUESTION),
                    ComponentLabel.NO: FixedStepDecider(StepID.END)},
                components={ComponentID.BTN_1: yes_btn_props, ComponentID.BTN_2: no_btn_props}
            ),
            StepID.END: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "End of demo, thanks for participating! 🏆"),
                next_step_decider=FixedStepDecider(StepID.END)
            )
        }



def update_workflow_step(steps: dict[StepID, ChatbotStep], workflow_state: WorkflowState, component_name: str) -> WorkflowManager:
    '''Update the workflow step based on the component that triggered the event'''

    next_step_id = workflow_state.current_step.determine_next_step(component_name, workflow_state.context)
    workflow_state.current_step_id = next_step_id
    workflow_state.current_step = steps[next_step_id]

    return workflow_state


def modify_context(
    workflow_state: WorkflowState
) -> WorkflowState:
    '''Update the workflow context based on the current step'''

    current_step = workflow_state.current_step
    current_step.initialize_step_func(workflow_state.context)

    return workflow_state


def show_initial_chatbot_message(
    workflow_state: WorkflowState,
    chat_history: list[list]
) -> tuple[WorkflowState, Iterator[list[tuple[str, None]]]]:
    '''Append the initial message of the current step to the chat history'''

    time.sleep(0.5)

    current_step = workflow_state.current_step
    
    for chatbot_message in current_step.initial_chatbot_message.get_formatted_message(workflow_state.context):
        yield workflow_state, chat_history + [[chatbot_message, None]]


def generate_chatbot_messages_from_trigger(
    workflow_state: WorkflowState,
    component_name: str,
    chat_history: list[list]
) -> Iterator[list[tuple[str, None]]]:
    '''Generate chatbot messages based on component trigger'''

    fns = workflow_state.current_step.get_generate_chatbot_messages_fns_for_trigger(component_name)
    yield from generate_chatbot_messages(fns=fns, chat_history=chat_history, context=workflow_state.context)


def find_matching_component_value_or_default(
    components: list[IOComponent], 
    component_name: str, 
    default_value: Any, 
    *components_values: Any
) -> Any:
    '''Find the value to save based on component name or use default value'''

    return next(
        (components_values[i] for i, c in enumerate(components) if (c.label or c.value) == component_name),
        default_value
    )

def find_and_store_event_value(
    workflow_state: WorkflowState, 
    default_value: Any, 
    components: list[IOComponent], 
    *components_values: Any
) -> WorkflowManager:
    '''Find the value to save and store it in the context'''

    outcome_saver = workflow_state.current_step.save_event_outcome

    if outcome_saver is not None:
        value_to_save = find_matching_component_value_or_default(
            components, 
            outcome_saver.component_name, 
            default_value, 
            *components_values
        )
        outcome_saver.save_fn(workflow_state.context, value_to_save)
        print(f"Saved value to context on step {workflow_state.current_step_id}: '{value_to_save}'")

    return workflow_state
