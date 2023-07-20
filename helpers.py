from typing import List
import os

import tiktoken

from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore

from prompts import get_prompt_template_for_generating_original_answer


def print_pretty_index(index: int):
    '''
    Print pretty index for document chunks with index+1 surrounded by underscores
    
        Parameters:
            index (int): index to print pretty
    '''
    print(f"{'_'*35}\n|{'_'*15} {index+1} {'_'*15}|\n")
    

def get_token_count_in_text(text: str, model_name: str = 'gpt-4') -> int:
    '''
    Get token count in text
    
        Parameters:
            text (str): text to get token count in
            model_name (str): name of the model to use for tokenization (default: 'gpt-4')
        
            Returns:
                int: token count in text
    '''
    return len(tiktoken.encoding_for_model(model_name).encode(text))


def get_token_count_in_documents(documents: List[Document], model_name: str = 'gpt-4') -> List[int]:
    '''
    Get token count in documents
    
        Parameters:
            documents (List[Document]): list of documents to get token count in
            model_name (str): name of the model to use for tokenization (default: 'gpt-4')
        
        Returns:
            List[int]: list of token counts in documents
    '''
    return [get_token_count_in_text(doc.page_content, model_name) for doc in documents]


def add_index_and_current_token_count_to_metadata_in_documents(
    documents: List[Document],
    model_name: str = 'gpt-4'
) -> None:
    '''
    Add index and current token count to metadata in documents
    
        Parameters:
            documents (List[Document]): list of documents to add index and current token count to metadata
            model_name (str): name of the model to use for tokenization (default: 'gpt-4')
        
    '''
    for i, doc in enumerate(documents):
        doc.metadata['current_token_count'] = get_token_count_in_text(doc.page_content, model_name)
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


def create_documents_from_files(files: list[str]) -> List[Document]:
    '''
    Create list of (type) Documents from files of different types and return it
    
        Parameters:
            files (list[str]): list of paths to files
        
        Returns:
            List[Document]: list of documents created
    '''

    documents: List[Document] = []

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


def create_documents_from_txt_files_in_dir(dir: str) -> List[Document]:
    '''
    Create list of (type) Documents from txt files in directory and return it
    
        Parameters:
            dir (str): path to directory containing txt files
        
        Returns:
            List[Document]: list of documents created
    '''

    documents: List[Document] = []

    for file in os.listdir(dir):
        # create document from txt file and add it to list of documents
        if file.endswith(".txt"):
            file_path = os.path.join(dir, file)
            documents.append(create_document(file_path))

    print(f'\n{len(documents)} Documents created from txt files in directory \'{os.path.abspath(dir)}\'')

    return documents


def get_documents_chunks_from_documents(
    documents: list[Document],
    model_name='gpt-4',
    chunk_size=4000,
    chunk_overlap=400,
    separators=["\n\n", "\n"]
) -> List[Document]:
    '''
    Get list of (type) Documents containing chunks of documents with given model name, chunk size, chunk overlap and separators
    
        Parameters:
            documents (list[Document]): list of documents to split into chunks
            model_name (str): name of the model to use for tokenization (default: 'gpt-4')
            chunk_size (int): max token size of chunks (default: 4000)
            chunk_overlap (int): max overlap of chunks (default: 400)
            separators (list[str]): list of separators to use for splitting documents into chunks (default: ["\n\n", "\n"])
        
        Returns:
            List[Document]: list of documents containing chunks of documents
    '''

    # create text splitter for splitting documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
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
    model_name='gpt-4',
    chunk_size=4000,
    chunk_overlap=400,
    separators=["\n\n", "\n"]
) -> List[Document]:
    '''
    Split documents into chunks with max token size of chunk_size and
    max overlap of chunk_overlap using separators and return list of documents chunks
    created from them with metadata containing current token count and index

        Parameters:
            files (List[str]): list of paths to files to create documents from
            model_name (str): name of the model to use for tokenization (default: 'gpt-4')
            chunk_size (int): max token size of each chunk (default: 4000)
            chunk_overlap (int): max overlap of each chunk (default: 400)
            separators (List[str]): list of separators to use for splitting text into chunks (default: ["\n\n", "\n"])

        Returns:
            List[Document]: list of documents chunks created from files
    '''

    documents = create_documents_from_files(files)

    return get_documents_chunks_from_documents(documents, model_name, chunk_size, chunk_overlap, separators)


def get_documents_chunks_for_txt_files_in_dir(
    dir: str,
    model_name='gpt-4',
    chunk_size=4000,
    chunk_overlap=400,
    separators=["\n\n", "\n"]
) -> List[Document]:
    '''
    Split documents in dir into chunks with max token size of chunk_size and
    max overlap of chunk_overlap using separators and return list of documents chunks
    created from them with metadata containing current token count and index

        Parameters:
            dir (str): path to directory containing txt files
            model_name (str): name of the model to use for tokenization (default: 'gpt-4')
            chunk_size (int): max token size of each chunk (default: 4000)
            chunk_overlap (int): max overlap of each chunk (default: 400)
            separators (List[str]): list of separators to use for splitting text into chunks (default: ["\n\n", "\n"])

        Returns:
            List[Document]: list of documents chunks created from txt files in dir
    '''

    documents = create_documents_from_txt_files_in_dir(dir)

    return get_documents_chunks_from_documents(documents, model_name, chunk_size, chunk_overlap, separators)


def get_most_relevant_docs_in_vector_store_for_answering_question(
    vector_store: VectorStore,
    question: str,
    n_results: int = 3
) -> List[Document]:
    '''
    Perform a similarity search in vector store for question and return the n_results most relevant documents
        Parameters:
            vector_store (VectorStore): vector store to perform similarity search in
            question (str): question to perform similarity search for
            n_results (int): number of most relevant documents to return (default: 3)
        
        Returns:
            List[Document]: list of the n_results most relevant documents for question
    '''

    # perform similarity search in vector store for question and return the n_results most relevant documents
    relevant_docs_and_scores = vector_store.similarity_search_with_score(query=question, k=n_results)

    print(f'Retrieved {n_results} most relevant Documents by performing a similarity search for question "{question}"')
    print(f'Similarities (distance): {[f"{score:.3f}" for _, score in relevant_docs_and_scores]}\n')

    # print summary of each relevant document and its similarity score (distance) to question
    for i, (doc, score) in enumerate(relevant_docs_and_scores):
        print_pretty_index(i)
        print(f'Similarity score (distance): {score:.3f}')
        print(f'Index: {doc.metadata["index"]}')
        print(f'Character length of chunk: {len(doc.page_content)}')
        print(f'Token count of chunk: {doc.metadata["current_token_count"]}')
        print(f'Source: \'{doc.metadata["source"].rsplit("/", 1)[-1]}\'\n')
        print(f'Content: (Preview of first 100 characters)\n{doc.page_content[:100]}.....\n')

    # return list of the relevant documents without their similarity scores
    relevant_docs = [doc for doc, _ in relevant_docs_and_scores]

    print(f'Total token count of most relevant documents: {sum([doc.metadata["current_token_count"] for doc in relevant_docs])}\n')

    return relevant_docs


def generate_answers_from_documents_for_question(
    documents: List[Document],
    chat_openai: ChatOpenAI,
    question: str,
    step: int = 1
) -> List[str]:
    '''
    Generate answers from documents for question using chat_model and print them
        Parameters:
            documents (List[Document]): list of documents to generate answers from
            chat_model (BaseChatModel): chat model to use for generating answers
            question (str): question to generate answers for
            step (int): see https://docs.google.com/document/d/1oxQJ0xYPkySs1C0t92md--t_AwD-smjhHkMd9dYqGyY/edit?pli=1 (default: 1)
        
        Returns:
            List[str]: list of answers generated from documents for question
    '''

    answers = []
    for i, doc in enumerate(documents):
        with get_openai_callback() as cb:
            chain = load_qa_chain(
                llm=chat_openai,
                chain_type='stuff',
                verbose=False,
                prompt=get_prompt_template_for_generating_original_answer() if step == 1 else None
            )

            answer = chain.run(input_documents=[doc], question=question)

            print_pretty_index(i)
            print(f'Generated answer for "{question}":\n\n{answer}\n')
            print(f'Summary info OpenAI callback:\n{cb}\n')

            answers.append(answer)
    
    print(f'Finished generating answer(s) for question "{question}" from {len(documents)} documents\n\n')

    return answers
