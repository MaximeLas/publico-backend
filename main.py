from enum import IntEnum, auto
import uuid
import logging
from threading import Thread
from asyncio import Queue, sleep

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import UUID4, BaseModel

from configurations.constants import Component
from firestore import update_session_state_in_firestore
from workflow.chatbot_step import EditorContentType
from workflow.session_state import SessionState
from workflow.steps import get_chatbot_step

logging.basicConfig(level=logging.INFO)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


# create a dict of sessions to session state
sessions: dict[UUID4, SessionState] = {}


class InputType(IntEnum):
    Chatbot = auto()
    Button = auto()
    NumberInput = auto()
    Files = auto()

class UserInput(BaseModel):
    input_type: InputType
    input_value: str | Component | int | list[str]


class ChatRequest(BaseModel):
    model_config = {
        'extra': 'forbid'
    }
    session_id: UUID4
    user_input: UserInput


JOB_DONE = object()  # Define a unique sentinel value for job completion

async def async_queue_generator(queue: Queue):
    while True:
        try:
            item = await queue.get()
            if item is JOB_DONE:
                break
            else:
                yield item
        except:
            logging.info("Error in async_queue_generator")
            await sleep(0.01)

class NewSessionResponse(BaseModel):
    session_id: UUID4
    initial_message: str
    components: set[Component]

class UpdatedEditorContent(BaseModel):
    question_index: int
    question: str | None = None
    word_limit: int | None = None
    answer: str | None = None

class AfterChatResponse(BaseModel):
    initial_message: str
    components: set[Component]
    updated_content: UpdatedEditorContent | None

class AfterChatRequest(BaseModel):
    session_id: UUID4

class EditAnswerRequest(BaseModel):
    session_id: UUID4
    question_index: int
    answer: str


def get_updated_content(state: SessionState) -> UpdatedEditorContent | None:
    updated_content = None

    if content_types := get_chatbot_step(state.current_step_id).updated_editor_contents:
        updated_content = UpdatedEditorContent(question_index=state.get_index_of_last_question())
        last_question_context = state.get_last_question_context()
        for content_type in content_types:
            if content_type == EditorContentType.QUESTION:
                updated_content.question = last_question_context.question
            elif content_type == EditorContentType.WORD_LIMIT:
                updated_content.word_limit = last_question_context.word_limit
            elif content_type == EditorContentType.ANSWER:
                updated_content.answer = (
                    last_question_context.get_last_improved_answer() or
                    last_question_context.get_revised_answer() or
                    last_question_context.get_original_answer()
                )
            else:
                raise ValueError(f'Unknown content type: {content_type}')

    return updated_content


def handle_chat_request(request: ChatRequest, queue: Queue):
    user_input = request.user_input.input_value
    state = sessions[request.session_id]
    chatbot_step = get_chatbot_step(state.current_step_id)

    if save_fn := chatbot_step.save_event_outcome_fn:
        save_fn(state, user_input)

    for fn in chatbot_step.get_generate_chatbot_messages_fns_for_trigger(trigger=state.last_user_input):
        fn(state, queue)

    queue.put_nowait(JOB_DONE)


'''API Endpoints'''

@app.post('/new_session')
async def new_session() -> NewSessionResponse:
    print(f'New session request')
    session_id = uuid.uuid4()
    state = SessionState(session_id=str(session_id))
    sessions[session_id] = state

    chatbot_step = get_chatbot_step(state.current_step_id)

    initial_message = chatbot_step.get_initial_chatbot_message(state)
    components=chatbot_step.get_components(state)

    print(f'New session id: {session_id}')
    return NewSessionResponse(session_id=session_id, initial_message=initial_message, components=components)


@app.post("/chat/")
async def chat(request: ChatRequest) -> StreamingResponse:
    print(f'Chat request: {request}')
    sessions[request.session_id].last_user_input = (
        Component(request.user_input.input_value)
            if request.user_input.input_type == InputType.Button
            else
        None)

    queue = Queue()
    Thread(target=handle_chat_request, kwargs=dict(request=request, queue=queue)).start()

    return StreamingResponse(content=async_queue_generator(queue=queue), media_type="text/event-stream")


@app.post("/after_chat")
async def after_chat(request: AfterChatRequest) -> AfterChatResponse:
    print(f'After chat request: {request}')
    state = sessions[request.session_id]
    
    chatbot_step = get_chatbot_step(state.current_step_id)

    updated_content = get_updated_content(state)
    
    state.current_step_id = chatbot_step.determine_next_step(state)
    chatbot_step = get_chatbot_step(state.current_step_id)
    chatbot_step.initialize_step_func(state)

    response = AfterChatResponse(
        initial_message=chatbot_step.get_initial_chatbot_message(state),
        components=chatbot_step.get_components(state),
        updated_content=updated_content
    )
    if updated_content:
        response.updated_content = updated_content

    update_session_state_in_firestore(str(request.session_id), state)  # Update the session state in Fi
    return response


@app.post("/edit")
async def edit(request: EditAnswerRequest) -> None:
    state = sessions[request.session_id]
    state.edit_last_question(request.question_index, request.answer)
    print(f"Edited answer for question {request.question_index} to: {request.answer}\n")
