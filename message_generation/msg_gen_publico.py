from asyncio import Queue

from workflow.session_state import SessionState


def generate_validation_message_following_files_upload(state: SessionState, queue: Queue) -> list[str]:
    '''Generate a validation message following a file upload.'''

    files = state.uploaded_files.files
    file_or_files = 'file' if len(files) == 1 else 'files'

    queue.put_nowait(
        f'You successfully uploaded **{len(files)}** {file_or_files}! 🎉\n\n' +
        'Now, on to your first grant application question!')
