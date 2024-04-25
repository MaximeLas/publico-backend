from devtools import debug

import tiktoken

from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.vectorstores.base import VectorStore

from configurations.constants import IS_DEV_MODE, GPT_MODEL
from firestore import get_files_for_user
from workflow.session_state import SessionState


VECTOR_STORE: Chroma = Chroma(embedding_function=OpenAIEmbeddings(client=None, model='text-embedding-3-large', dimensions=1024))

def print_pretty_index(index: int):
    '''
    Print pretty index for document chunks with index+1 surrounded by underscores
    
        Parameters:
            index (int): index to print pretty
    '''
    print(f"{'_'*35}\n|{'_'*15} {index+1} {'_'*15}|\n")
    

def get_token_count_in_text(text: str, model: str = GPT_MODEL) -> int:
    '''
    Get token count in text
    
        Parameters:
            text (str): text to get token count in
            model (str): name of the model to use for tokenization (default: 'GPT_MODEL')
        
            Returns:
                int: token count in text
    '''
    return len(tiktoken.encoding_for_model(model).encode(text))


def get_token_count_in_documents(documents: list[Document], model: str = GPT_MODEL) -> list[int]:
    '''
    Get token count in documents
    
        Parameters:
            documents (list[Document]): list of documents to get token count in
            model (str): name of the model to use for tokenization (default: GPT_MODEL)
        
        Returns:
            list[int]: list of token counts in documents
    '''
    return [get_token_count_in_text(doc.page_content, model) for doc in documents]


def add_index_and_current_token_count_to_metadata_in_documents(
    documents: list[Document],
    model: str = GPT_MODEL
) -> None:
    '''
    Add index and current token count to metadata in documents
    
        Parameters:
            documents (list[Document]): list of documents to add index and current token count to metadata
            model (str): name of the model to use for tokenization (default: GPT_MODEL)
        
    '''
    for i, doc in enumerate(documents):
        doc.metadata['current_token_count'] = get_token_count_in_text(doc.page_content, model)
        doc.metadata['index'] = i + 1


def create_document(file: dict[str, str]) -> Document:
    '''
    Create document from file and return it

        Parameters:
            file (dict[str, str]): dictionary containing file name and content
        
        Returns:
            Document: document created from file
    '''

    document = Document(
        page_content=file['content'],
        metadata={'source': file['file_name'], 'original_token_count': get_token_count_in_text(file['content'])}
    )

    # print document created from txt file and its token count in metadata
    print(f'• Document containing {document.metadata["original_token_count"]} tokens created from txt file \'{document.metadata["source"].split("/")[-1]}\'')
    
    return document


def create_documents_from_files(file_names: list[str], user_id: str) -> list[Document]:
    '''
    Create list of (type) Documents from files of different types and return it
    
        Parameters:
            files (list[str]): list of paths to files
        
        Returns:
            list[Document]: list of documents created
    '''

    documents: list[Document] = []
    files = get_files_for_user(file_names, user_id)
    for file in files:
        # create document from txt file and add it to list of documents
        if file['file_name'].endswith('.txt'):
            documents.append(create_document(file))
        elif file['file_name'].endswith('.docx'):
            loader = UnstructuredFileLoader(file['file_name']) # needs fixing
            text = loader.load()[0].page_content

            file_path_txt = file['file_name'].replace('.docx', '.txt')
            with open(file_path_txt, 'w') as file:
                file.write(text)

            documents.append(create_document(file))

    print(f'\n{len(documents)} Documents created from given list of files')

    return documents


def get_documents_chunks_from_documents(
    documents: list[Document],
    model=GPT_MODEL,
    chunk_size=4000,
    chunk_overlap=400,
    separators=["\n\n", "\n"]
) -> list[Document]:
    '''
    Get list of (type) Documents containing chunks of documents with given model name, chunk size, chunk overlap and separators
    
        Parameters:
            documents (list[Document]): list of documents to split into chunks
            model (str): name of the model to use for tokenization (default: GPT_MODEL)
            chunk_size (int): max token size of chunks (default: 4000)
            chunk_overlap (int): max overlap of chunks (default: 400)
            separators (list[str]): list of separators to use for splitting documents into chunks (default: ["\n\n", "\n"])
        
        Returns:
            list[Document]: list of documents containing chunks of documents
    '''

    # create text splitter for splitting documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model,
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    print(f'\nSplitting original Documents using separators {separators} into chunks ' +
        f'with max token size of {chunk_size} and max overlap of {chunk_overlap}\n')

    # split documents into chunks
    documents_chunks = text_splitter.split_documents(documents)

    print(f'{len(documents_chunks)} Documents created after split:')

    # add current token count and index to metadata for each document
    add_index_and_current_token_count_to_metadata_in_documents(documents_chunks)

    # print summary of metadata for each document
    for doc in documents_chunks:
        print(f'• {doc.metadata["current_token_count"]} tokens in documents chunks from source \'{doc.metadata["source"].rsplit("/", 1)[-1]}\'')
    print()

    return documents_chunks
    
    
def get_documents_chunks_for_files(
    files: list[str],
    user_id: str,
    model=GPT_MODEL,
    chunk_size=4000,
    chunk_overlap=400,
    separators=["\n\n", "\n"]
) -> list[Document]:
    '''
    Split documents into chunks with max token size of chunk_size and
    max overlap of chunk_overlap using separators and return list of documents chunks
    created from them with metadata containing current token count and index

        Parameters:
            files (list[str]): list of paths to files to create documents from
            model (str): name of the model to use for tokenization (default: GPT_MODEL)
            chunk_size (int): max token size of each chunk (default: 4000)
            chunk_overlap (int): max overlap of each chunk (default: 400)
            separators (list[str]): list of separators to use for splitting text into chunks (default: ["\n\n", "\n"])

        Returns:
            list[Document]: list of documents chunks created from files
    '''

    documents = create_documents_from_files(files, user_id)

    return get_documents_chunks_from_documents(documents, model, chunk_size, chunk_overlap, separators)


def print_summary_of_relevant_documents_and_scored(docs: list[tuple[Document, float]]):

    debug(**{'Similarities (distance)': [f'{score:.3f}' for _, score in docs]})

    # print summary of each relevant document and its similarity score (distance) to question
    '''for i, (doc, score) in enumerate(docs):
        print_pretty_index(i)
        words = doc.page_content.split()
        debug(**{
            'Similarity score (distance)': f'{score:.3f}',
            'Index': doc.metadata['index'],
            'Character length of chunk': len(doc.page_content),
            'Token count of chunk': doc.metadata['current_token_count'],
            'Source': doc.metadata['source'].rsplit("/", 1)[-1],
            'Session ID': doc.metadata['session_id'],
            'Content (Preview of first and last 15 words)': ' '.join(words[:15]) + ' ... ' + ' '.join(words[-15:]),
        })
    '''

    # print total token count of relevant documents
    debug(**{'Total token count of relevant documents': sum([doc.metadata["current_token_count"] for doc, _ in docs])})


def add_files_to_vector_store(state: SessionState):
    '''
    Get vector store for files uploaded by user and return it
        Parameters:
            session_id (str): session ID of the user
            files (list[str]): list of paths to files uploaded by user
            tokens_per_doc_chunk (int): number of tokens per document chunk (default: 1000)

        Returns:
            VectorStore: vector store for files uploaded by user
    '''

    # get the files in the vector store
    files_for_session = set(md['source'].rsplit('.', 1)[0] for md in VECTOR_STORE.get(where={'session_id': state.session_id})['metadatas'])

    # get the files uploaded by the user
    files_uploaded = set(file.rsplit('.', 1)[0] for file in state.uploaded_files)

    # check if the uploaded files are different from the files in the vector store
    if files_uploaded != files_for_session or IS_DEV_MODE:
        # if so, get the documents chunks for the uploaded files
        documents_chunks = get_documents_chunks_for_files(
            files=state.uploaded_files,
            user_id=state.user_id,
            chunk_size=state.get_num_of_tokens_per_doc_chunk(),
            chunk_overlap=150)

        # delete the current embeddings in the vector store
        if files_for_session:
            VECTOR_STORE.delete(ids=VECTOR_STORE.get(where={'session_id': state.session_id})['ids'])

        # add the documents chunks to the vector store as embeddings
        VECTOR_STORE.add_texts(
            texts=[doc.page_content for doc in documents_chunks],
            metadatas=[doc.metadata | {"session_id": state.session_id} for doc in documents_chunks])

def get_most_relevant_docs_in_vector_store_for_answering_question(
    session_id: str,
    question: str,
    n_results: int = 3
) -> list[Document]:
    '''
    Perform a similarity search in vector store for question and return the n_results most relevant documents
        Parameters:
            session_id (str): session ID of the user
            question (str): question to perform similarity search for
            n_results (int): number of most relevant documents to return (default: 3)
        
        Returns:
            list[Document]: list of the n_results most relevant documents for question
    '''

    # perform similarity search in vector store for question and return the n_results most relevant documents
    relevant_docs_and_scores = VECTOR_STORE.similarity_search_with_score(query=question, k=n_results, filter={'session_id': session_id})

    print(f'Retrieved {n_results} most relevant Documents by performing a similarity search for question "{question}"')
    print_summary_of_relevant_documents_and_scored(relevant_docs_and_scores)

    # return list of the relevant documents without their similarity scores
    relevant_docs = [doc for doc, _ in relevant_docs_and_scores]

    return relevant_docs
