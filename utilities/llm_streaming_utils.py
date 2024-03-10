from collections.abc import Iterator
from queue import Queue, Empty
#from asyncio import Queue
from threading import Thread
from typing import Literal
import re


from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate

#from configurations.constants import GPT_MODEL

import logging
logging.basicConfig(level=logging.INFO)

class QueueCallback(BaseCallbackHandler):
    '''Callback handler for streaming LLM generated tokens to a queue, used to create a generator.'''

    def __init__(self, q: Queue):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        logging.info(f"Putting new token into queue: {token}")
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        logging.info("LLM end signaled")
        return self.q.empty()


def stream_from_llm_generation(
    prompt: ChatPromptTemplate,
    queue: Queue,
    on_llm_end: callable = lambda _: print('LLM end signaled'),
    chain_type: Literal['llm_chain', 'qa_chain'] = 'llm_chain',
    model: str = 'gpt-3.5-turbo',
    temperature: float = 0,
    verbose = False,
    docs: list[Document] | None = None,
    **input_variables
) -> Iterator[tuple[str, str, str]]:
    '''
    This function will return a generator that will yield the next token from the LLM.
    
        Parameters:
            prompt: the prompt to use for the LLM
            chain_type: the type of chain to use for the LLM (defaults to llm_chain)
            model: the model to use for the LLM (defaults to GPT_MODEL)
            temperature: the temperature to use for the LLM (defaults to 0)
            verbose: whether to print out the LLM's output (defaults to False)
            docs: the documents to use for the qa_chain, only used if chain_type is qa_chain (defaults to None)
            input_variables: the input variables included in the prompt
        
        Returns:
            a generator that will yield the next token from the LLM, the current answer, and the current answer formatted for display
    '''

    if chain_type not in ['llm_chain', 'qa_chain']:
        print(f'chain_type: {chain_type} not recognized\n')
        return

    print('\n-------------------------------------------------------------')
    print('---------------------- Input variables ----------------------')
    #debug(**input_variables)
    print('-------------------------------------------------------------\n')

    # Create a Queue
    #q = Queue()
    job_done = object()

    q = Queue()
    # Initialize the LLM we'll be using
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=True, 
        callbacks=[QueueCallback(q)]
    )

    logging.info("Starting LLM streaming")
    # Create a funciton to call - this will run in a thread
    def task():
        logging.info("LLM task started")
        chain_constructor = LLMChain if chain_type == 'llm_chain' else load_qa_chain
        chain = chain_constructor(
            llm=llm,
            prompt= prompt,
            verbose=verbose)

        # print out the length of the documents if we're using a qa_chain
        if chain_type == 'qa_chain':
            if docs is not None:
                logging.info(f'length of documents provided: {sum([len(doc.page_content) for doc in docs])}')
                #print(**{'length of documents provided': sum([len(doc.page_content) for doc in docs])})
            else:
                raise ValueError('No documents were provided, this should never happen!')

        # add the documents to the kwargs if we're using a qa_chain and run the chain
        kwargs = input_variables if chain_type == 'llm_chain' else {'input_documents': docs, **input_variables}
        chain.invoke(input=kwargs)

        logging.info("LLM streaming finished, adding job_done to queue")
        # put the job_done object in the queue to signal that we're done
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()
    logging.info("Task started")

    answer = ''
    answer_formatted = '*'
    num_tokens = 0
    # Get each new token from the queue and yield for our generator
    while True:
        try:
            # get the next token from the queue
            if (next_token := q.get(True, 1)) is not job_done:
                num_tokens += 1
                answer += next_token
                if '\n\n' in next_token:
                    # to display italics correctly remove any whitespaces before the \n\n
                    # and add an asterix before and after \n\n
                    answer_formatted += re.sub(r'\s*\n\n', '*\n\n*', next_token)
                else:
                    answer_formatted += next_token

                queue.put_nowait(next_token)
                # TODO: add an asterix on the client '*'
            else:
                print_end_of_stream(answer, num_tokens)
                if on_llm_end is not None:
                    on_llm_end(answer, answer_formatted + '*')

                return
        except Empty:
            print('Queue is empty')
            continue

def print_end_of_stream(answer: str, num_tokens: int):
    print('\n-------------------------------------------------------------')
    print('----------------------- End of stream -----------------------')
    print(
        f'Number of tokens generated: {num_tokens}\n' +
        f'Number of words generated: {len(answer.split())}\n' +
        f'Final answer from LLM: {answer}'
    )
    print('-------------------------------------------------------------\n')
