from constants import ContextKeys

def generate_validation_message_following_files_upload(context: dict) -> list[str]:
    if ContextKeys.APPLICATION_QUESTION in context:
        # if the user already answered a question, then this validation message has already been generated
        return []
    else:
        files = context[ContextKeys.PRIOR_GRANT_APPLICATIONS]
        file_or_files = 'file' if len(files) == 1 else 'files'

        return [f'You successfully uploaded {len(files)} {file_or_files}! 🎉', 'Now, on to the first question!']
