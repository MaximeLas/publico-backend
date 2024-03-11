from collections.abc import Iterator
from typing import Callable, Literal
from asyncio import Queue as AsyncQueue
from queue import Queue, Empty
from devtools import debug
import logging
from threading import Thread
import re

from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate

from configurations.constants import GPT_MODEL


logging.basicConfig(level=logging.INFO)


class QueueCallback(BaseCallbackHandler):
    '''Callback handler for streaming LLM generated tokens to a queue, used to create a generator.'''

    def __init__(self, q: Queue):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        logging.info("LLM end signaled")
        return self.q.empty()


def stream_from_llm_generation(
    prompt: ChatPromptTemplate,
    queue: AsyncQueue,
    on_llm_end: Callable[[str, str], None] | None = None,
    chain_type: Literal['llm_chain', 'qa_chain'] = 'llm_chain',
    model: str = GPT_MODEL,
    temperature: float = 0,
    verbose: bool = False,
    docs: list[Document] | None = None,
    **input_variables
) -> Iterator[tuple[str, str, str]]:
    '''
    This function streams tokens from the LLM to a queue, and yields the tokens as they come in.

    Args:
        prompt: the prompt to use for the LLM
        queue: the queue to stream the tokens to
        on_llm_end: a function to call when the LLM has finished generating tokens
        chain_type: the type of chain to use, either 'llm_chain' or 'qa_chain'
        model: the model to use for the LLM (default is GPT_MODEL, which is set to 'gpt-3.5-turbo' if not specified)
        temperature: the temperature to use for the LLM (default is 0)
        verbose: whether to print out the LLM's output (defaults to False)
        docs: the documents to use for the QA chain, only used if chain_type is 'qa_chain' (defaults to None)
        input_variables: the input variables included in the prompt
    '''

    if chain_type not in ['llm_chain', 'qa_chain']:
        print(f'chain_type: {chain_type} not recognized\n')
        return

    print('\n-------------------------------------------------------------')
    print('---------------------- Input variables ----------------------')
    debug(**input_variables)
    print('-------------------------------------------------------------\n')

    q = Queue()
    job_done = object()

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=True, 
        callbacks=[QueueCallback(q)]
    )

    def task():
        chain_constructor = LLMChain if chain_type == 'llm_chain' else load_qa_chain
        chain = chain_constructor(
            llm=llm,
            prompt= prompt,
            verbose=verbose)

        if chain_type == 'qa_chain':
            if docs is not None:
                logging.info(f'length of documents provided: {sum([len(doc.page_content) for doc in docs])}')
            else:
                raise ValueError('No documents were provided, this should never happen!')

        # add the documents to the kwargs if we're using a qa_chain and run the chain
        kwargs = input_variables if chain_type == 'llm_chain' else {'input_documents': docs, **input_variables}
        chain.invoke(input=kwargs)

        # put the job_done object in the queue to signal that we're done
        logging.info("LLM streaming finished, adding job_done to queue")
        q.put(job_done)

    # Create a thread and start the function
    Thread(target=task).start()
    logging.info("Task started")

    answer = ''
    answer_formatted = '*'
    num_tokens = 0

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            # get the next token from the queue
            if (next_token := q.get(block=True, timeout=1)) is not job_done:
                num_tokens += 1
                answer += next_token
                answer_formatted += (
                    # to display italics correctly remove any whitespaces before
                    # the \n\n and add an asterix before and after \n\n
                    re.sub(r'\s*\n\n', '*\n\n*', next_token)
                        if '\n\n' in next_token
                        else
                    next_token
                )

                queue.put_nowait(next_token)
            else:
                print_end_of_stream(answer, num_tokens)
                if on_llm_end is not None:
                    on_llm_end(answer, answer_formatted + '*')
                return
        except Empty:
            logging.info('Queue is empty after 1 second of waiting')
            continue

def print_end_of_stream(answer: str, num_tokens: int):
    logging.info('----------------------- End of stream -----------------------')
    logging.info(
        f'Number of tokens generated: {num_tokens}\n' +
        f'Number of words generated: {len(answer.split())}\n' +
        f'Final answer from LLM: {answer}'
    )
    logging.info('-------------------------------------------------------------\n')
