from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from gradio.components import IOComponent

from chatbot_step import ChatbotStep, ConditionalNextStepDecider, EventOutcomeSaver, InitialChatbotMessage, NextStepDecider
from constants import ComponentLabel, StepID
from context import UserContext
from message_generator_llm import check_for_comprehensiveness, generate_answer_for_implicit_question_stream, generate_answer_to_question_stream, generate_final_answer_stream
from message_generator_publico import generate_chatbot_messages, generate_validation_message_following_files_upload



@dataclass
class WorkflowState():
    '''
    Represents the state of the chatbot's workflow at a particular point in time.

    The WorkflowState is used as an intermediate structure that can hold and manage the workflow
    context and the current step of the workflow when the application's state is updated
    by Gradio. This is necessary because Gradio flattens complex objects into dictionaries,
    causing issues when trying to directly manipulate instances of classes like WorkflowManager.

    Attributes:
        current_step_id (StepID): The current step of the chatbot workflow
        context (UserContext): The context of the user interacting with the chatbot
    '''

    current_step_id: StepID
    context: UserContext


class WorkflowManager:
    def __init__(self):
        self.steps: dict[StepID, ChatbotStep] = self.initialize_steps()
        self._current_step_id = StepID.START
        self.context: UserContext = UserContext()

    @property
    def current_step_id(self) -> StepID:
        return self._current_step_id

    @current_step_id.setter
    def current_step_id(self, step_id: StepID) -> None:
        if step_id not in self.steps:
            raise ValueError(f'Invalid step ID: {step_id}')

        self._current_step_id = step_id


    def get_current_step(self) -> ChatbotStep:
        return self.steps[self.current_step_id]


    def get_step(self, step_id: StepID) -> ChatbotStep:
        return self.steps[step_id]


    @staticmethod
    def initialize_steps() -> dict[StepID, ChatbotStep]:
        return {
            StepID.START: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Hello there, please hit **Start** when you're ready."),
                next_step_decider=NextStepDecider(StepID.HAVE_YOU_APPLIED_BEFORE)
            ),
            StepID.HAVE_YOU_APPLIED_BEFORE: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Have you applied for this grant before?"),
                next_step_decider={
                    ComponentLabel.YES: NextStepDecider(StepID.UPLOAD_FILES),
                    ComponentLabel.NO: NextStepDecider(StepID.ENTER_QUESTION)}
            ),
            StepID.UPLOAD_FILES: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "That's very useful! Please upload your documents.\n" +
                    "Supported file types: **.docx** & **.txt**"),
                next_step_decider=NextStepDecider(StepID.ENTER_QUESTION),
                save_event_outcome=EventOutcomeSaver(UserContext.set_uploaded_files, ComponentLabel.FILES),
                generate_chatbot_messages_fns=[
                    generate_validation_message_following_files_upload]
            ),
            StepID.ENTER_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Please type the grant application question."),
                next_step_decider=NextStepDecider(StepID.ENTER_WORD_LIMIT),
                initialize_step_func=UserContext.add_new_question,
                save_event_outcome=EventOutcomeSaver(UserContext.set_grant_application_question, ComponentLabel.USER)
            ),
            StepID.ENTER_WORD_LIMIT: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "What is the word limit?"),
                next_step_decider=NextStepDecider(StepID.DO_COMPREHENSIVENESS_CHECK),
                save_event_outcome=EventOutcomeSaver(UserContext.set_word_limit, ComponentLabel.NUMBER),
                generate_chatbot_messages_fns=[
                    generate_answer_to_question_stream]
            ),
            StepID.DO_COMPREHENSIVENESS_CHECK: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Would you like me to review the answer to make sure it includes everything needed?"),
                next_step_decider={
                    ComponentLabel.YES: NextStepDecider(StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                    ComponentLabel.NO: NextStepDecider(StepID.DO_ANOTHER_QUESTION)},
                save_event_outcome=EventOutcomeSaver(UserContext.set_do_check_for_comprehensiveness, None),
                generate_chatbot_messages_fns=defaultdict(list, {
                    ComponentLabel.YES: [check_for_comprehensiveness]})
            ),
            StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    message="(**{index}**) **{question}**\n\n" +
                        "Does this question address a topic or information that should be included?",
                    extract_formatting_variables_func=lambda context: (yield {
                        'question': context.get_next_implicit_question_to_be_answered(),
                        'index': context.get_index_of_implicit_question_being_answered()})),
                next_step_decider={
                    ComponentLabel.YES: NextStepDecider(StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT),
                    ComponentLabel.NO: ConditionalNextStepDecider(
                        next_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                        alternative_step=StepID.READY_TO_GENERATE_FINAL_ANSWER,
                        condition=UserContext.has_more_implcit_questions_to_answer)},
                generate_chatbot_messages_fns=defaultdict(list, {
                    ComponentLabel.NO: [lambda _: "Okay, let's skip this one."]})
            ),
            StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    message="{response}",
                    extract_formatting_variables_func=generate_answer_for_implicit_question_stream),
                next_step_decider={
                    button: (
                        ConditionalNextStepDecider(
                            next_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                            alternative_step=StepID.READY_TO_GENERATE_FINAL_ANSWER,
                            condition=UserContext.has_more_implcit_questions_to_answer
                        ) if button in [ComponentLabel.GOOD_AS_IS, ComponentLabel.NO] else
                        NextStepDecider(StepID.PROMPT_USER_TO_SUBMIT_ANSWER)
                    ) for button in [
                        ComponentLabel.GOOD_AS_IS, ComponentLabel.EDIT_IT,
                        ComponentLabel.YES, ComponentLabel.NO]},
                generate_chatbot_messages_fns=defaultdict(list, {
                    ComponentLabel.NO: [lambda _: "Okay, let's skip this one."],
                    ComponentLabel.GOOD_AS_IS: [lambda _: "Great! We'll use this answer."]})
                
            ),
            StepID.PROMPT_USER_TO_SUBMIT_ANSWER: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Okay, go ahead and write an answer to the question."),
                next_step_decider=ConditionalNextStepDecider(
                    next_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                    alternative_step=StepID.READY_TO_GENERATE_FINAL_ANSWER,
                    condition=UserContext.has_more_implcit_questions_to_answer),
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
                next_step_decider=NextStepDecider(StepID.DO_ANOTHER_QUESTION),
                generate_chatbot_messages_fns=[generate_final_answer_stream]
            ),
            StepID.DO_ANOTHER_QUESTION: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "Do you want to generate an answer for another question?"),
                next_step_decider={
                    ComponentLabel.YES: NextStepDecider(StepID.ENTER_QUESTION),
                    ComponentLabel.NO: NextStepDecider(StepID.END)}
            ),
            StepID.END: ChatbotStep(
                initial_chatbot_message=InitialChatbotMessage(
                    "End of demo, thanks for participating! 🏆"),
                next_step_decider=NextStepDecider(StepID.END)
            )
        }

            

def get_current_chatbot_step(workflow_manager: WorkflowManager, workflow_state: dict):
    current_step_id = workflow_state["current_step_id"]
    return workflow_manager.steps[current_step_id]


def update_workflow_step(steps: dict[StepID, ChatbotStep], workflow_state: WorkflowState, component_name: str) -> WorkflowManager:
    '''Update the workflow step based on the component that triggered the event'''

    next_step = steps[workflow_state.current_step_id].determine_next_step(component_name, workflow_state.context)
    workflow_state.current_step_id = next_step
    return workflow_state


def modify_context(
    steps: dict[StepID, ChatbotStep],
    workflow_state: WorkflowState
) -> WorkflowState:
    '''Update the workflow context based on the current step'''

    current_step = steps[workflow_state.current_step_id]
    current_step.initialize_step_func(workflow_state.context)

    return workflow_state


def show_initial_chatbot_message(
    steps: dict[StepID, ChatbotStep],
    workflow_state: WorkflowState,
    chat_history: list[list]
) -> tuple[WorkflowState, Iterator[list[tuple[str, None]]]]:
    '''Append the initial message of the current step to the chat history'''

    current_step = steps[workflow_state.current_step_id]
    
    for chatbot_message in current_step.initial_chatbot_message.get_formatted_message(workflow_state.context):
        yield workflow_state, chat_history + [[chatbot_message, None]]


def generate_chatbot_messages_from_trigger(
    steps: dict[StepID, ChatbotStep],
    workflow_state: WorkflowState,
    component_name: str,
    chat_history: list[list]
) -> Iterator[list[tuple[str, None]]]:
    '''Generate chatbot messages based on component trigger'''

    fns = steps[workflow_state.current_step_id].get_generate_chatbot_messages_fns_for_trigger(component_name)
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
    steps: dict[StepID, ChatbotStep],
    workflow_state: WorkflowState, 
    default_value: Any, 
    components: list[IOComponent], 
    *components_values: Any
) -> WorkflowManager:
    '''Find the value to save and store it in the context'''

    outcome_saver = steps[workflow_state.current_step_id].save_event_outcome

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
