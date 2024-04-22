


from collections import defaultdict

from configurations.constants import (
    IS_DEV_MODE,
    Component,
    StepID
)
from message_generation.msg_gen import (
    check_for_comprehensiveness,
    generate_answer_for_implicit_question_stream,
    generate_answer_to_question_stream,
    generate_final_answer_stream,
    generate_improved_answer_following_user_guidance_prompt
)
from message_generation.msg_gen import generate_validation_message_following_files_upload
from workflow.chatbot_step import ChatbotStep, EditorContentType
from workflow.session_state import SessionState
from workflow.step_decider import ConditionalStepDecider, FixedStepDecider, MultiConditionalStepDecider

dnl = '\n&nbsp;\n&nbsp;'

def get_initial_chatbot_message_for_generating_answer_to_implicit_question(state: SessionState) -> str:
    question, index = state.get_next_implicit_question_and_index()

    return (
        f'(**{index+1}**) **{question}**{dnl}' +
        'Does this question address a topic or information that should be incorporated into the final answer?')


STEPS: dict[StepID, ChatbotStep] = {
    StepID.START: ChatbotStep(
        initial_chatbot_message=(
            f"Welcome! 👋{dnl}" +
            f"I'm Publico, your personal grant writing coach.{dnl}" +
            "Are you ready to start writing together?"),
        components={Component.START},
        next_step_decider=FixedStepDecider(StepID.HAVE_MATERIALS_TO_SHARE)
    ),
    StepID.HAVE_MATERIALS_TO_SHARE: ChatbotStep(
        initial_chatbot_message=(
            "Do you have any existing materials you'd like to share (past grants, reports, etc.)?"),
        components={Component.YES, Component.NO},
        next_step_decider={
            Component.YES: FixedStepDecider(StepID.UPLOAD_FILES),
            Component.NO: FixedStepDecider(StepID.ENTER_QUESTION)}
    ),
    StepID.UPLOAD_FILES: ChatbotStep(
        initial_chatbot_message=(
            f"That's very useful! Please upload your documents below.{dnl}" +
            "Supported file types: **.docx** & **.txt**"),
        components={Component.FILES},
        save_event_outcome_fn=SessionState.set_uploaded_files,
        next_step_decider=FixedStepDecider(StepID.ENTER_QUESTION),
        generate_chatbot_messages_fns=[generate_validation_message_following_files_upload]
    ),
    StepID.ENTER_QUESTION: ChatbotStep(
        initial_chatbot_message=(
            "Please type the grant application question."),
        components={Component.CHATBOT},
        save_event_outcome_fn=SessionState.set_grant_application_question,
        next_step_decider=FixedStepDecider(StepID.ENTER_WORD_LIMIT),
        initialize_step_func=SessionState.add_new_question,
        updated_editor_contents={EditorContentType.QUESTION}
    ),
    StepID.ENTER_WORD_LIMIT: ChatbotStep(
        initial_chatbot_message=("What is the word limit? 🛑"),
        components={Component.WORD_LIMIT},
        save_event_outcome_fn=SessionState.set_word_limit,
        next_step_decider=FixedStepDecider(
            StepID.GO_OVER_IMPLICIT_QUESTIONS
                if not IS_DEV_MODE else
            StepID.ENTER_RAG_CONFIG_ORIGINAL_QUESTION),
        generate_chatbot_messages_fns=[
            generate_answer_to_question_stream, check_for_comprehensiveness]
                if not IS_DEV_MODE else
            [],
        updated_editor_contents={EditorContentType.WORD_LIMIT, EditorContentType.ANSWER}
    ),
    StepID.ENTER_RAG_CONFIG_ORIGINAL_QUESTION: ChatbotStep(
        initial_chatbot_message=(
            f"1) What system prompt should be used to generate the answer?{dnl}" +
            f"2) How many tokens at most should be included in a single document chunk?{dnl}" +
            "3) How many chunks sould be selected in the similarity check step?"),
        components={Component.NUM_OF_TOKENS, Component.NUM_OF_DOCS},
        save_event_outcome_fn=SessionState.set_test_config_params,
        next_step_decider=FixedStepDecider(StepID.GO_BACK_TO_CONFIG_STEP_ORIGINAL_QUESTION),
        generate_chatbot_messages_fns=[generate_answer_to_question_stream]
    ),
    StepID.GO_BACK_TO_CONFIG_STEP_ORIGINAL_QUESTION: ChatbotStep(
        initial_chatbot_message=("Do you want to go back to the previous step (RAG Config)?"),
        components={Component.YES, Component.NO},
        next_step_decider={
            Component.YES: FixedStepDecider(StepID.ENTER_RAG_CONFIG_ORIGINAL_QUESTION),
            Component.NO: FixedStepDecider(StepID.GO_OVER_IMPLICIT_QUESTIONS)},
        generate_chatbot_messages_fns={Component.NO: [check_for_comprehensiveness]}
    ),
    StepID.GO_OVER_IMPLICIT_QUESTIONS: ChatbotStep(
        initial_chatbot_message=("Would you like to answer the questions one by one?"),
        components={Component.YES, Component.NO},
        next_step_decider={
            Component.YES: FixedStepDecider(StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
            Component.NO: FixedStepDecider(StepID.ASK_USER_IF_GUIDANCE_NEEDED)}
    ),
    StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION: ChatbotStep(
        initial_chatbot_message=get_initial_chatbot_message_for_generating_answer_to_implicit_question,
        components={Component.YES, Component.NO},
        next_step_decider={
            Component.YES: FixedStepDecider(
                StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT
                    if not IS_DEV_MODE else
                StepID.ENTER_RAG_CONFIG_IMPLICIT_QUESTION),
            Component.NO: MultiConditionalStepDecider(
                conditional_steps=[
                    (SessionState.has_more_implcit_questions_to_answer, StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                    (SessionState.exists_answer_to_any_implicit_question, StepID.READY_TO_GENERATE_FINAL_ANSWER)
                ],
                default_next_step=StepID.DO_ANOTHER_QUESTION)},
        generate_chatbot_messages_fns=defaultdict(list, {
            Component.YES: [generate_answer_for_implicit_question_stream] if not IS_DEV_MODE else [],
            Component.NO: [lambda state, queue: (
                queue.put_nowait("Okay, let's skip this one.")
                    if SessionState.has_more_implcit_questions_to_answer(state) or
                        SessionState.exists_answer_to_any_implicit_question(state)
                    else
                queue.put_nowait("None of the implicit questions were answered."))]})
    ),
    StepID.ENTER_RAG_CONFIG_IMPLICIT_QUESTION: ChatbotStep(
        initial_chatbot_message=(
            f"1) What system prompt should be used to generate the answer?{dnl}" +
            f"2) How many tokens at most should be included in a single document chunk?{dnl}" +
            "3) How many chunks sould be selected in the similarity check step?"),
        components={Component.NUM_OF_TOKENS, Component.NUM_OF_DOCS},
        save_event_outcome_fn=SessionState.set_test_config_params,
        next_step_decider=FixedStepDecider(StepID.GO_BACK_TO_CONFIG_STEP_IMPLICIT_QUESTION),
        generate_chatbot_messages_fns=[generate_answer_for_implicit_question_stream]
    ),
    StepID.GO_BACK_TO_CONFIG_STEP_IMPLICIT_QUESTION: ChatbotStep(
        initial_chatbot_message=("Do you want to go back to the previous step (RAG Config)?"),
        components={Component.YES, Component.NO},
        next_step_decider={
            Component.YES: FixedStepDecider(StepID.ENTER_RAG_CONFIG_IMPLICIT_QUESTION),
            Component.NO: FixedStepDecider(StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT)}
    ),
    StepID.SELECT_WHAT_TO_DO_WITH_ANSWER_GENERATED_FROM_CONTEXT: ChatbotStep(
        initial_chatbot_message=(lambda state: (
            "Is this helpful?" if state.exists_answer_to_current_implicit_question()
                else
            "Would you like to answer it yourself?")),
        components=lambda state:
            {Component.YES, Component.NO}
                if not SessionState.exists_answer_to_current_implicit_question(state)
                else
            {Component.GOOD_AS_IS, Component.EDIT_IT},
        next_step_decider={
            Component.YES: FixedStepDecider(StepID.PROMPT_USER_TO_SUBMIT_ANSWER),
            Component.EDIT_IT: FixedStepDecider(StepID.PROMPT_USER_TO_SUBMIT_ANSWER),
            Component.NO: MultiConditionalStepDecider(
                conditional_steps=[
                    (SessionState.has_more_implcit_questions_to_answer, StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION),
                    (SessionState.exists_answer_to_any_implicit_question, StepID.READY_TO_GENERATE_FINAL_ANSWER)
                ],
                default_next_step=StepID.DO_ANOTHER_QUESTION),
            Component.GOOD_AS_IS: ConditionalStepDecider(
                condition=SessionState.has_more_implcit_questions_to_answer,
                if_true_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
                if_false_step=StepID.READY_TO_GENERATE_FINAL_ANSWER)},
        generate_chatbot_messages_fns=defaultdict(list, {
            Component.NO: [lambda state, queue: (
                queue.put_nowait("Okay, let's skip this one.")
                    if SessionState.has_more_implcit_questions_to_answer(state) or
                        SessionState.exists_answer_to_any_implicit_question(state)
                    else
                queue.put_nowait("None of the implicit questions were answered."))],
            Component.GOOD_AS_IS: [lambda _, queue: queue.put_nowait("Great! We'll use this answer.")]})
    ),
    StepID.PROMPT_USER_TO_SUBMIT_ANSWER: ChatbotStep(
        initial_chatbot_message=("Okay, go ahead and write an answer to the question."),
        components={Component.CHATBOT},
        save_event_outcome_fn=SessionState.set_answer_to_current_implicit_question,
        next_step_decider=ConditionalStepDecider(
            condition=SessionState.has_more_implcit_questions_to_answer,
            if_true_step=StepID.DO_PROCEED_WITH_IMPLICIT_QUESTION,
            if_false_step=StepID.READY_TO_GENERATE_FINAL_ANSWER),
        generate_chatbot_messages_fns=[lambda state, queue: (
            queue.put_nowait("Great! Now that we've answered that question, let's move on to the next.")
                if state.has_more_implcit_questions_to_answer()
                else 
            queue.put_nowait(''))]
    ),
    StepID.READY_TO_GENERATE_FINAL_ANSWER: ChatbotStep(
        initial_chatbot_message=(
            f"We're done with the implicit questions! 🏁{dnl}" +
            "Are you ready to have your final answer generated?"),
        components={Component.OF_COURSE},
        next_step_decider=FixedStepDecider(StepID.ASK_USER_IF_GUIDANCE_NEEDED),
        generate_chatbot_messages_fns=[generate_final_answer_stream],
        updated_editor_contents={EditorContentType.ANSWER}
    ),
    StepID.ASK_USER_IF_GUIDANCE_NEEDED: ChatbotStep(
        initial_chatbot_message=("If you'd like, you can give me guidance to improve the answer."),
        components={Component.GOOD_AS_IS, Component.ADD_GUIDANCE},
        next_step_decider={
            Component.GOOD_AS_IS: FixedStepDecider(StepID.DO_ANOTHER_QUESTION),
            Component.ADD_GUIDANCE: FixedStepDecider(StepID.USER_GUIDANCE_PROMPT)}
    ),
    StepID.USER_GUIDANCE_PROMPT: ChatbotStep(
        initial_chatbot_message=(
            "What guidance do you want to provide to improve the answer? " +
            "(ie. *Make it more formal* or *Mention that we helped 150 clients last year*)"),
        components={Component.CHATBOT},
        save_event_outcome_fn=SessionState.set_user_guidance_prompt,
        next_step_decider=ConditionalStepDecider(
            condition=SessionState.is_allowed_to_add_more_guidance,
            if_true_step=StepID.ASK_USER_IF_GUIDANCE_NEEDED,
            if_false_step=StepID.DO_ANOTHER_QUESTION),
        generate_chatbot_messages_fns=[generate_improved_answer_following_user_guidance_prompt],
        updated_editor_contents={EditorContentType.ANSWER}
    ),
    StepID.DO_ANOTHER_QUESTION: ChatbotStep(
        initial_chatbot_message=("Do you want to generate an answer for another question? 🔄"),
        components={Component.YES, Component.NO},
        next_step_decider={
            Component.YES: FixedStepDecider(StepID.ENTER_QUESTION),
            Component.NO: FixedStepDecider(StepID.END)}
    ),
    StepID.END: ChatbotStep(
        initial_chatbot_message=("Thanks for participating! 👏"),
        components={}
    )
}

def get_chatbot_step(step_id: StepID) -> ChatbotStep:
    return STEPS[step_id]
