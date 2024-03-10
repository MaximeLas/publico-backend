from enum import Enum, auto
import logging
from threading import Thread
from asyncio import Queue, sleep
import uuid

from fastapi.responses import StreamingResponse

from workflow.steps import get_chatbot_step

from fastapi import FastAPI
from pydantic import UUID4, BaseModel
from configurations.constants import Component
from workflow.session_state import SessionState

from workflow.chatbot_step import EditorContentType

logging.basicConfig(level=logging.INFO)


app = FastAPI()

# create a dict of sessions to session state
sessions: dict[UUID4, SessionState] = {}


class InputType(Enum):
    Chatbot = auto()
    NumberInput = auto()
    Button = auto()

class UserInput(BaseModel):
    input_type: InputType
    input_value: str | Component | int


class ChatRequest(BaseModel):
    model_config = {
        'extra': 'forbid'
    }
    session_id: UUID4
    user_input: UserInput


job_done = object()  # Define a unique sentinel value for job completion
async def async_queue_generator(queue: Queue):
    logging.info("Starting async generator")
    while True:
        try:
            item = await queue.get()
            if item is job_done:
                logging.info("Received job_done, stopping generator")
                break
            else:
                logging.info(f"Yielding item: {item}")
                yield item
        except:
            logging.info("Error in async_queue_generator")
            await sleep(0.01)

class NewSessionResponse(BaseModel):
    session_id: UUID4
    initial_message: str

class UpdatedEditorContent(BaseModel):
    question_index: int
    question: str | None = None
    word_limit: int | None = None
    answer: str | None = None

class AfterChatResponse(BaseModel):
    initial_message: str
    components: set[Component]
    updated_content: UpdatedEditorContent | None = None

class AfterChatRequest(BaseModel):
    session_id: UUID4


def get_updated_content(state: SessionState):
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
                    last_question_context.get_last_improved_answer(False) or
                    last_question_context.get_revised_answer(False) or
                    last_question_context.get_original_answer(False)
                )
            else:
                raise ValueError(f'Unknown content type: {content_type}')

    return updated_content


def handle_chat_request(request: ChatRequest, queue: Queue):
    user_input = request.user_input.input_value
    state = sessions[request.session_id] #state.chat_history += [[f'**{user_input}**', None]]
    chatbot_step = get_chatbot_step(state.current_step_id)

    if save_fn := chatbot_step.save_event_outcome_fn:
        save_fn(state, user_input)

    for fn in chatbot_step.get_generate_chatbot_messages_fns_for_trigger(user_input):
        fn(state, queue)

    queue.put_nowait(job_done)


'''API Endpoints'''

@app.post('/new_session')
async def new_session() -> NewSessionResponse:
    session_id = uuid.uuid4()

    state = SessionState()

    sessions[session_id] = state

    chatbot_step = get_chatbot_step(state.current_step_id)
    initial_message = chatbot_step.initial_chatbot_message.get_formatted_message(state)

    return NewSessionResponse(session_id=session_id, initial_message=initial_message)


@app.post("/chat/")
async def chat(request: ChatRequest) -> StreamingResponse:
    if request.session_id not in sessions:
        sessions[request.session_id] = SessionState()

    user_input = request.user_input.input_value
    sessions[request.session_id].last_user_input = user_input

    queue = Queue()
    Thread(target=handle_chat_request, kwargs=dict(request=request, queue=queue)).start()

    return StreamingResponse(content=async_queue_generator(queue=queue), media_type="text/event-stream")


@app.post("/after_chat")
async def after_chat(request: AfterChatRequest) -> AfterChatResponse:
    state = sessions[request.session_id]
    
    chatbot_step = get_chatbot_step(state.current_step_id)

    updated_content = get_updated_content(state)
    
    state.current_step_id = chatbot_step.determine_next_step(state.last_user_input, state)
    chatbot_step = get_chatbot_step(state.current_step_id)
    chatbot_step.initialize_step_func(state)

    initial_message = chatbot_step.initial_chatbot_message.get_formatted_message(state)

    response = AfterChatResponse(
        initial_message=initial_message,
        components=chatbot_step.get_components(state)
    )
    if updated_content:
        response.updated_content = updated_content

    return response
