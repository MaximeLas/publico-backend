from collections.abc import Iterator
from queue import Queue
import time

import gradio as gr

from workflow.chatbot_step import GenerateMessageFunc
from workflow.session_state import SessionState


def generate_validation_message_following_files_upload(state: SessionState, queue: Queue) -> list[str]:
    '''Generate a validation message following a file upload.'''

    files = state.uploaded_files.files
    file_or_files = 'file' if len(files) == 1 else 'files'

    # time.sleep(0.15)
    queue.put_nowait(
        f'You successfully uploaded **{len(files)}** {file_or_files}! 🎉\n\n' +
        'Now, on to your first grant application question!')


def create_new_chatbot_messages_from_response(response: str | list[str]) -> list[tuple[str, None]]:
    '''Create new chatbot messages from a response.'''

    new_chatbot_messages = (
        [(None, response)]
            if isinstance(response, str)
            else
        [(None, message) for message in response]
    )

    return new_chatbot_messages


def generate_chatbot_messages(
    fns: list[GenerateMessageFunc],
    context: SessionState
) -> Iterator[list[tuple[str, None]]]:
    '''Generate chatbot messages from a list of functions, and yield the chat history with the new chatbot messages.'''

    if not fns:
        yield gr.skip()
        return

    all_new_chatbot_messages = []
    for fn in fns:
        if (response := fn(context)) is not None:
            new_chatbot_messages = []

            if isinstance(response, Iterator):
                for response_so_far in response:
                    new_chatbot_messages = create_new_chatbot_messages_from_response(response_so_far)
                    yield all_new_chatbot_messages + new_chatbot_messages
            else:
                new_chatbot_messages = create_new_chatbot_messages_from_response(response)
                yield all_new_chatbot_messages + new_chatbot_messages

            all_new_chatbot_messages += new_chatbot_messages

    yield all_new_chatbot_messages
