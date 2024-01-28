from collections.abc import Iterator
from queue import Queue, Empty
from threading import Thread
from typing import Literal
import re

from devtools import debug

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate

from configurations.constants import GPT_MODEL



class QueueCallback(BaseCallbackHandler):
    '''Callback handler for streaming LLM generated tokens to a queue, used to create a generator.'''

    def __init__(self, q: Queue):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        return self.q.empty()


def stream_from_llm_generation(
    prompt: ChatPromptTemplate,
    chain_type: Literal['llm_chain', 'qa_chain'] = 'llm_chain',
    model: str = GPT_MODEL,
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
    debug(**input_variables)
    print('-------------------------------------------------------------\n')

    # Create a Queue
    q = Queue()
    job_done = object()

    # Initialize the LLM we'll be using
    llm = ChatOpenAI(
        client=None,
        model=model,
        temperature=temperature,
        streaming=True, 
        callbacks=[QueueCallback(q)]
    )

    # Create a funciton to call - this will run in a thread
    def task():
        chain_constructor = LLMChain if chain_type == 'llm_chain' else load_qa_chain
        chain = chain_constructor(
            llm=llm,
            prompt= prompt,
            verbose=verbose)

        # print out the length of the documents if we're using a qa_chain
        if chain_type == 'qa_chain':
            if docs is not None:
                debug(**{'length of documents provided': sum([len(doc.page_content) for doc in docs])})
            else:
                raise ValueError('No documents were provided, this should never happen!')

        # add the documents to the kwargs if we're using a qa_chain and run the chain
        kwargs = input_variables if chain_type == 'llm_chain' else {'input_documents': docs, **input_variables}
        chain.run(**kwargs)

        # put the job_done object in the queue to signal that we're done
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    answer = ''
    answer_formatted = '*'
    num_tokens = 0
    # Get each new token from the queue and yield for our generator
    while True:
        try:
            # get the next token from the queue
            if (next_token := q.get(True, timeout=1)) is not job_done:
                num_tokens += 1
                answer += next_token
                if '\n\n' in next_token:
                    # to display italics correctly remove any whitespaces before the \n\n
                    # and add an asterix before and after \n\n
                    answer_formatted += re.sub(r'\s*\n\n', '*\n\n*', next_token)
                else:
                    answer_formatted += next_token

                yield next_token, answer, answer_formatted + '*'
            else:
                print('\n-------------------------------------------------------------')
                print('----------------------- End of stream -----------------------')
                debug(**{
                    'Number of tokens generated': num_tokens,
                    'Number of words generated': len(answer.split()),
                    'Final answer from LLM': answer
                })
                print('-------------------------------------------------------------\n')
                break
        except Empty:
            continue
