from dataclasses import asdict, is_dataclass
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud import exceptions

cred = credentials.Certificate("./publico-ai-firebase-adminsdk-rnl7e-43eac58a0e.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'publico-ai.appspot.com'
})

db = firestore.client()

def get_chat_session(session_id: str = '05b5253c-903c-41e9-8a5d-605a0161db16') -> dict:
    doc_ref = db.collection('chat_sessions').document(session_id)
    try:
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return None
    except exceptions.NotFound:
        print(f'No such document with ID: {session_id}')
        return None


def get_files_for_user(user_id: str, file_names: list[str] = ['PBRC.txt']) -> list[str]:
    user_folder = f'chat_documents/{user_id}/'
    bucket = storage.bucket()

    # This list will store the contents of the found files
    file_contents = []

    for file_name in file_names:
        file_path = f"{user_folder}{file_name}"
        blob = bucket.blob(file_path)
        
        # Check if the file exists in the user's folder
        if blob.exists():
            # Read the content of the file
            try:
                content = blob.download_as_text()
                file_contents.append(content)
            except Exception as e:
                print(f"Error reading content from {file_path}: {e}")
    
    return file_contents


from firebase_admin import auth

MY_ID = 'YcsbK8htCbUYO5egX1L428LiYwi2'
user = auth.get_user(MY_ID)
print(f'Fetched user data: {user.email}')

def serialize_for_firestore(obj):
    if is_dataclass(obj):
        return {k: serialize_for_firestore(v) for k, v in asdict(obj).items() if v is not None}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_firestore(v) for v in obj if v is not None]
    elif isinstance(obj, dict):
        return {str(k): serialize_for_firestore(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, datetime.datetime):
        return obj.replace(tzinfo=datetime.timezone.utc) if obj.tzinfo is None else obj
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        raise TypeError(f"Unsupported type for Firestore serialization: {type(obj)}")


def update_session_state_in_firestore(session_id: str, session_state: any):
    # Use this enhanced function to serialize your session state before sending to Firestore
    session_state_serialized = serialize_for_firestore(session_state)
    #print(f'\nsession_state_serialized: {session_state_serialized}')  # Add this right before the Firestore update call
    db.collection('chat_sessions').document(session_id).update(session_state_serialized)
