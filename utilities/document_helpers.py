import os

from devtools import debug

import tiktoken

from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore

from configurations.constants import IS_DEV_MODE, GPT_MODEL


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


def create_document(file_path: str) -> Document:
    '''
    Create (type) Document from txt file and return it with metadata containing original token count and source file name
    
        Parameters:
            file_path (str): path to txt file
        
        Returns:
            Document: document created
    '''

    # read file and create document from it with metadata containing original token count and source file name
    with open(file_path) as f:
        text = f.read()
        tokens_count = get_token_count_in_text(text)
        document = Document(
            page_content=text,
            metadata={'source': file_path, 'original_token_count': tokens_count}
        )
    
    # print document created from txt file and its token count in metadata
    print(f'• Document containing {document.metadata["original_token_count"]} tokens created from txt file \'{document.metadata["source"].split("/")[-1]}\'')
    
    return document


def create_documents_from_files(files: list[str]) -> list[Document]:
    '''
    Create list of (type) Documents from files of different types and return it
    
        Parameters:
            files (list[str]): list of paths to files
        
        Returns:
            list[Document]: list of documents created
    '''

    documents: list[Document] = []

    for file_path in files:
        # create document from txt file and add it to list of documents
        if file_path.endswith('.txt'):
            documents.append(create_document(file_path))
        elif file_path.endswith('.docx'):
            loader = UnstructuredFileLoader(file_path)
            text = loader.load()[0].page_content

            file_path_txt = file_path.replace('.docx', '.txt')
            with open(file_path_txt, 'w') as file:
                file.write(text)

            documents.append(create_document(file_path_txt))

    print(f'\n{len(documents)} Documents created from given list of files')

    return documents


def create_documents_from_txt_files_in_dir(dir: str) -> list[Document]:
    '''
    Create list of (type) Documents from txt files in directory and return it
    
        Parameters:
            dir (str): path to directory containing txt files
        
        Returns:
            list[Document]: list of documents created
    '''

    documents: list[Document] = []

    for file in os.listdir(dir):
        # create document from txt file and add it to list of documents
        if file.endswith(".txt"):
            file_path = os.path.join(dir, file)
            documents.append(create_document(file_path))

    print(f'\n{len(documents)} Documents created from txt files in directory \'{os.path.abspath(dir)}\'')

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

    documents = create_documents_from_files(files)

    return get_documents_chunks_from_documents(documents, model, chunk_size, chunk_overlap, separators)


def get_documents_chunks_for_txt_files_in_dir(
    dir: str,
    model=GPT_MODEL,
    chunk_size=4000,
    chunk_overlap=400,
    separators=["\n\n", "\n"]
) -> list[Document]:
    '''
    Split documents in dir into chunks with max token size of chunk_size and
    max overlap of chunk_overlap using separators and return list of documents chunks
    created from them with metadata containing current token count and index

        Parameters:
            dir (str): path to directory containing txt files
            model (str): name of the model to use for tokenization (default: GPT_MODEL)
            chunk_size (int): max token size of each chunk (default: 4000)
            chunk_overlap (int): max overlap of each chunk (default: 400)
            separators (list[str]): list of separators to use for splitting text into chunks (default: ["\n\n", "\n"])

        Returns:
            list[Document]: list of documents chunks created from txt files in dir
    '''

    documents = create_documents_from_txt_files_in_dir(dir)

    return get_documents_chunks_from_documents(documents, model, chunk_size, chunk_overlap, separators)


def print_summary_of_relevant_documents_and_scored(docs: list[tuple[Document, float]]):

    debug(**{'Similarities (distance)': [f'{score:.3f}' for _, score in docs]})

    # print summary of each relevant document and its similarity score (distance) to question
    for i, (doc, score) in enumerate(docs):
        print_pretty_index(i)
        words = doc.page_content.split()
        debug(**{
            'Similarity score (distance)': f'{score:.3f}',
            'Index': doc.metadata['index'],
            'Character length of chunk': len(doc.page_content),
            'Token count of chunk': doc.metadata['current_token_count'],
            'Source': doc.metadata["source"].rsplit("/", 1)[-1],
            'Content (Preview of first and last 15 words)': ' '.join(words[:15]) + ' ... ' + ' '.join(words[-15:]),
        })

    # print total token count of relevant documents
    debug(**{'Total token count of relevant documents': sum([doc.metadata["current_token_count"] for doc, _ in docs])})


def get_vector_store_for_files(files: list[str], tokens_per_doc_chunk=1000) -> VectorStore:
    '''
    Get vector store for files uploaded by user and return it
        Parameters:
            files (list[str]): list of paths to files uploaded by user

        Returns:
            VectorStore: vector store for files uploaded by user
    '''

    # create vector store with OpenAIEmbeddings
    vector_store = Chroma(embedding_function=OpenAIEmbeddings(client=None, model='text-embedding-3-large', dimensions=1024))

    # get the files in the vector store
    files_in_vector_store = set(metadata['source'].rsplit('.', 1)[0] for metadata in vector_store._collection.get()['metadatas'])
    # get the files uploaded by the user
    files_uploaded = set(file.rsplit('.', 1)[0] for file in files)

    # check if the uploaded files are different from the files in the vector store
    if IS_DEV_MODE or files_uploaded != files_in_vector_store:
        # if so, get the documents chunks for the uploaded files
        documents_chunks = get_documents_chunks_for_files(
            files=files,
            chunk_size=tokens_per_doc_chunk,
            chunk_overlap=150)

        # delete the current embeddings in the vector store
        if files_in_vector_store:
            vector_store.delete(ids=vector_store._collection.get()['ids'])

        # add the documents chunks to the vector store as embeddings
        vector_store.add_texts(
            texts=[doc.page_content for doc in documents_chunks],
            metadatas=[doc.metadata for doc in documents_chunks])

    return vector_store


def get_most_relevant_docs_in_vector_store_for_answering_question(
    vector_store: VectorStore,
    question: str,
    n_results: int = 3
) -> list[Document]:
    '''
    Perform a similarity search in vector store for question and return the n_results most relevant documents
        Parameters:
            vector_store (VectorStore): vector store to perform similarity search in
            question (str): question to perform similarity search for
            n_results (int): number of most relevant documents to return (default: 3)
        
        Returns:
            list[Document]: list of the n_results most relevant documents for question
    '''

    # perform similarity search in vector store for question and return the n_results most relevant documents
    relevant_docs_and_scores = vector_store.similarity_search_with_score(query=question, k=n_results)

    print(f'Retrieved {n_results} most relevant Documents by performing a similarity search for question "{question}"')
    print_summary_of_relevant_documents_and_scored(relevant_docs_and_scores)

    # return list of the relevant documents without their similarity scores
    relevant_docs = [doc for doc, _ in relevant_docs_and_scores]

    return relevant_docs
