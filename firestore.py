import logging
from dataclasses import asdict, fields, is_dataclass
from enum import Enum, StrEnum

import firebase_admin
from firebase_admin import credentials, firestore, storage

from workflow.session_state import SessionState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Firebase Configuration
CREDENTIAL_PATH = "./publico-ai-firebase-adminsdk-rnl7e-43eac58a0e.json"
STORAGE_BUCKET = 'publico-ai.appspot.com'
MY_ID = 'YcsbK8htCbUYO5egX1L428LiYwi2'

# Firebase Initialization
cred = credentials.Certificate(CREDENTIAL_PATH)
firebase_admin.initialize_app(cred, {'storageBucket': STORAGE_BUCKET})
db = firestore.client()

def fetch_document(collection, document_id):
    doc_ref = db.collection(collection).document(document_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        logger.info(f"No such document with ID: {document_id}")
        return None

def get_files_for_user(user_id: str, file_names: list[str] = ['PBRC.txt']) -> list[str]:
    user_folder = f'chat_documents/{user_id}/'
    bucket = storage.bucket()

    file_contents = []
    for file_name in file_names:
        file_path = f"{user_folder}{file_name}"
        blob = bucket.blob(file_path)
        if blob.exists():
            try:
                content = blob.download_as_text()
                file_contents.append(content)
            except Exception as e:
                logger.error(f"Error reading content from {file_path}: {e}")

    return file_contents

def handle_list_conversion(element_type, list_values):
    return [convert_value(element_type, item) for item in list_values]

def handle_enum_conversion(type_hint, value):
    if issubclass(type_hint, StrEnum):
        return type_hint[value.upper()]
    return type_hint(value)

def convert_value(type_hint, value):
    if not isinstance(type_hint, type):
        return [convert_value(dict, item) for item in value] if isinstance(value, list) else value

    try:
        if is_dataclass(type_hint):
            return deserialize_to_dataclass(type_hint, value)
        if isinstance(value, list):
            element_type = type_hint.__args__[0] if hasattr(type_hint, '__args__') else dict
            return handle_list_conversion(element_type, value)
        if issubclass(type_hint, Enum):
            return handle_enum_conversion(type_hint, value)
    except Exception as e:
        logger.error(f"Failed to convert value '{value}' with type_hint '{type_hint}': {e}")
        return None
    return value

def deserialize_list(field_type, value):
    element_type = field_type.__args__[0] if hasattr(field_type, '__args__') else dict
    if is_dataclass(element_type):
        return [deserialize_to_dataclass(element_type, item) for item in value]
    else:
        return [convert_value(element_type, item) for item in value]

def deserialize_to_dataclass(cls, data):
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in fields(cls)}
    converted_data = {}

    for key, value in data.items():
        type_hint = field_types.get(key)
        if not type_hint:
            logger.debug(f"Field '{key}' not defined in dataclass '{cls.__name__}'. Skipping...")
            continue

        try:
            converted_data[key] = deserialize_list(type_hint, value) if isinstance(value, list) else convert_value(type_hint, value)
        except Exception as e:
            logger.error(f"Error processing field '{key}' in dataclass '{cls.__name__}': {e}", exc_info=True)
            converted_data[key] = None

    return cls(**converted_data)

def serialize_for_firestore(obj):
    if is_dataclass(obj):
        return {k: serialize_for_firestore(v) for k, v in asdict(obj).items() if v is not None}
    return serialize_iterable_for_firestore(obj) if isinstance(obj, (list, tuple, dict)) else obj

def serialize_iterable_for_firestore(obj):
    if isinstance(obj, dict):
        return {k: serialize_for_firestore(v) for k, v in obj.items() if v is not None}
    else:  # it's a list or tuple
        return [serialize_for_firestore(v) for v in obj if v is not None]

def retrieve_session_state_from_firestore(session_id: str) -> SessionState:
    server_data = fetch_document('chat_sessions', session_id)
    if server_data and 'server_state' in server_data:
        return deserialize_to_dataclass(SessionState, server_data['server_state'])
    else:
        raise ValueError(f"No valid server state found with ID: {session_id}")

def update_chat_session_in_firestore(session_id: str, session_state: SessionState):
    session_state_serialized = serialize_for_firestore(session_state)
    db.collection('chat_sessions').document(session_id).set({'server_state': session_state_serialized}, merge=True)
