

# define chatbot steps and their properties
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from devtools import debug

from gradio.components import IOComponent

from chatbot_step import ChatbotStep, ConditionalStepDecider, EventOutcomeSaver, StepDecider
from constants import StepID
from context import UserContext
from message_generator_llm import check_for_comprehensiveness, generate_answer_for_implicit_question_stream, generate_answer_to_question_stream, generate_answers_for_implicit_questions_stream, generate_final_answer_stream
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
                initial_chatbot_message="Hello there, please hit **Start** when you're ready.",
                step_decider=StepDecider(StepID.HAVE_YOU_APPLIED_BEFORE)
            ),
            StepID.HAVE_YOU_APPLIED_BEFORE: ChatbotStep(
                initial_chatbot_message="Have you applied for this grant before?",
                step_decider=dict(
                    Yes=StepDecider(StepID.UPLOAD_PRIOR_GRANT_APPLICATIONS),
                    No=StepDecider(StepID.ENTER_QUESTION))
            ),
            StepID.UPLOAD_PRIOR_GRANT_APPLICATIONS: ChatbotStep(
                initial_chatbot_message="That's very useful! Please upload your prior grant application(s).",
                step_decider=StepDecider(StepID.ENTER_QUESTION),
                save_event_outcome=EventOutcomeSaver(UserContext.set_prior_grant_applications, 'Documents'),
                generate_chatbot_messages_fns=[generate_validation_message_following_files_upload]
            ),
            StepID.ENTER_QUESTION: ChatbotStep(
                initial_chatbot_message="Please type the grant application question.",
                step_decider=StepDecider(StepID.ENTER_WORD_LIMIT),
                save_event_outcome=EventOutcomeSaver(UserContext.set_grant_application_question, 'User')
            ),
            StepID.ENTER_WORD_LIMIT: ChatbotStep(
                initial_chatbot_message="What is the word limit?",
                step_decider=StepDecider(StepID.DO_COMPREHENSIVENESS_CHECK),
                save_event_outcome=EventOutcomeSaver(UserContext.set_word_limit, 'Number'),
                generate_chatbot_messages_fns=[generate_answer_to_question_stream]
            ),
            StepID.DO_COMPREHENSIVENESS_CHECK: ChatbotStep(
                initial_chatbot_message="Do you want to check the comprehensiveness of the generated answer?",
                step_decider=dict(
                    Yes=StepDecider(StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                    No=StepDecider(StepID.DO_ANOTHER_QUESTION)),
                save_event_outcome=EventOutcomeSaver(UserContext.set_do_check_for_comprehensiveness, None),
                generate_chatbot_messages_fns=defaultdict(list,
                    Yes=[check_for_comprehensiveness])
                    #Yes=[check_for_comprehensiveness, generate_answers_for_implicit_questions_stream, generate_final_answer_stream])
            ),
            # for each implicit question:
                StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION: ChatbotStep(
                    initial_chatbot_message="(**{index}**) **{question}**\n\nDo you want to answer this question in the revised answer?",
                    retrieve_relevant_vars_func=lambda context: {
                        'question': context.get_next_implicit_question_to_be_answered(),
                        'index': context.get_index_of_implicit_question_being_answered()},
                    step_decider=dict(
                        Yes=StepDecider(StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT),
                        No=ConditionalStepDecider(
                            next_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                            alternative_step=StepID.DO_ANOTHER_QUESTION,
                            condition=UserContext.has_more_implcit_questions_to_answer)),
                    generate_chatbot_messages_fns=defaultdict(list,
                        No=[lambda _: "Okay, let's skip this one."])
                ),
                StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT: ChatbotStep(
                    initial_chatbot_message=generate_answer_for_implicit_question_stream,
                    step_decider=dict({
                        'Good as is!': ConditionalStepDecider(
                            next_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                            alternative_step=StepID.READY_TO_GENERATE_FINAL_ANSWER,
                            condition=UserContext.has_more_implcit_questions_to_answer),
                        'Let me edit it': StepDecider(StepID.PROMPT_USER_TO_SUBMIT_ANSWER),
                        "I'll write one myself": StepDecider(StepID.PROMPT_USER_TO_SUBMIT_ANSWER)})
                ),
                StepID.PROMPT_USER_TO_SUBMIT_ANSWER: ChatbotStep(
                    initial_chatbot_message="Okay, let's edit the proposed answer.\nYou can edit the text in the box below, or type over it to replace it with any information you think better answers the question.",
                    step_decider=ConditionalStepDecider(
                        next_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                        alternative_step=StepID.READY_TO_GENERATE_FINAL_ANSWER,
                        condition=UserContext.has_more_implcit_questions_to_answer)
                ),
                StepID.READY_TO_GENERATE_FINAL_ANSWER: ChatbotStep(
                    initial_chatbot_message="We're done with the implicit questions!\nAre you ready to have your final answer generated?",
                    step_decider=StepDecider(StepID.DO_ANOTHER_QUESTION),
                    generate_chatbot_messages_fns=[generate_final_answer_stream]
                ),
                
            # end for
            StepID.DO_ANOTHER_QUESTION: ChatbotStep(
                initial_chatbot_message="Do you want to generate an answer for another question?",
                step_decider=dict(
                    Yes=StepDecider(StepID.ENTER_QUESTION),
                    No=StepDecider(StepID.END))
            ),
            StepID.END: ChatbotStep(
                initial_chatbot_message='End of demo, thanks for participating! 🏆',
                step_decider=StepDecider(StepID.END)
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


def show_initial_chatbot_message(
    steps: dict[StepID, ChatbotStep],
    workflow_state: WorkflowState,
    chat_history: list[list]
) -> Iterator[list[tuple[str, None]]]:
    '''Append the initial message of the current step to the chat history'''
    debug(workflow_state.current_step_id)
    current_step = steps[workflow_state.current_step_id]
    debug(current_step)
    initial_chatbot_message = current_step.initial_chatbot_message
    debug(initial_chatbot_message)
    chatbot_message = ''
    vars_for_step = current_step.retrieve_relevant_vars_func(workflow_state.context)
    if isinstance(initial_chatbot_message, str):
        chatbot_message = initial_chatbot_message.format(**vars_for_step)
        debug(chatbot_message)
        yield chat_history + [[chatbot_message, None]]
    else:
        response_generator = initial_chatbot_message(context=workflow_state.context)
        for response in response_generator:
            yield chat_history + [[response, None]]
        debug(response)


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

    return workflow_state