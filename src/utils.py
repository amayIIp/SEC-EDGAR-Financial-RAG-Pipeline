# src/utils.py
# This module provides utility functions for performance profiling, logging, and token counting.
# In production RAG systems, monitoring execution time and API cost (tokens) is crucial.

import time # Standard library module to measure time and system clocks.
import logging # Standard library module to print structured log messages.
import functools # Standard library containing higher-order functions for wrapping decorators.
import tiktoken # OpenAI's library to count how many tokens (sub-word chunks) a text contains.

# Set up the logging configuration. Logging is a structured way of tracking events in a running program.
# We set the level to INFO, meaning it will print informational messages, warnings, and errors.
# The format specifies that every log print will show the timestamp, the severity level, and the message text.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sec_rag_utils") # Create a logger object specifically for this utility module.

# =========================================================================================
# Advanced Concept: RAG Profiling and Latency Measurement
# Retrieval-Augmented Generation (RAG) involves multiple distinct steps: fetching data,
# parsing HTML, splitting text into chunks, embedding chunks using deep learning models,
# inserting them into databases, querying databases, reranking results, and calling an LLM.
# To identify bottlenecks (which part is slow), we measure the execution time of functions
# using a decorator. A decorator in Python is a function that wraps another function to
# extend its behavior without editing the original function's source code.
# =========================================================================================

def profile_time(func):
    """
    A decorator that measures the time it takes for a function to execute and logs it.
    This helps us identify which parts of our RAG pipeline are slow (e.g., database queries vs. embeddings).
    """
    # functools.wraps is used to copy the original function's name and metadata (docstrings) to the wrapper.
    # Without this, the decorated function would appear to have the name 'wrapper' instead of its actual name.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record the high-precision start time using the computer's CPU performance counter.
        start_time = time.perf_counter()
        
        # Execute the original function and save its return value.
        result = func(*args, **kwargs)
        
        # Record the high-precision end time right after the function finishes.
        end_time = time.perf_counter()
        
        # Calculate the elapsed time in seconds by subtracting the start time from the end time.
        duration = end_time - start_time
        
        # Log the function name and how long it took to execute.
        logger.info(f"Function '{func.__name__}' completed execution in {duration:.4f} seconds.")
        
        # Return the output of the original function so the program flow continues normally.
        return result
    
    # Return the wrapper function, replacing the original function with this profiled version.
    return wrapper


class ProfilerContext:
    """
    A Context Manager that allows profiling of specific blocks of code using the 'with' statement.
    This is useful for profiling a small subset of lines inside a large function.
    """
    def __init__(self, block_name: str):
        # Store the name of the code block so we can print it in the logs.
        self.block_name = block_name
        # Initialize the start time variable to None.
        self.start_time = None

    def __enter__(self):
        # This method is called when we enter the 'with' block.
        # We record the start time right as we enter the block.
        self.start_time = time.perf_counter()
        # Return self so the block runs.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # This method is called automatically when we exit the 'with' block, even if an error occurs.
        # Record the end time.
        end_time = time.perf_counter()
        # Calculate the total duration the block took to execute.
        duration = end_time - self.start_time
        # Log the block's name and the elapsed time.
        logger.info(f"Block '{self.block_name}' took {duration:.4f} seconds to run.")


# =========================================================================================
# Advanced Concept: Tokenization and Token Counting
# Large Language Models (LLMs) do not read text character-by-character or word-by-word.
# Instead, they break text down into small pieces of words (sometimes whole words, sometimes
# prefixes, suffixes, or syllables) called "tokens".
# Different models use different tokenizers (the algorithm that translates text into token IDs).
# To manage context window limits (the maximum amount of text a model can read at once) and
# to estimate cost (since APIs charge per token), we must count the number of tokens in our text.
# =========================================================================================

def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    Counts the number of tokens in a given text using tiktoken.
    Defaults to the gpt-4o-mini tokenizer.
    """
    try:
        # Load the encoding/tokenizer specific to the model we want to use.
        # This fetches the tokenizer rules (like cl100k_base or o200k_base) for the model.
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # If the model name is unrecognized, fall back to the standard cl100k_base tokenizer rules,
        # which is used by most modern OpenAI models (like gpt-4, gpt-3.5-turbo).
        logger.warning(f"Model '{model_name}' not recognized by tiktoken. Falling back to 'cl100k_base' encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Encode the text string into a list of token integers.
    token_list = encoding.encode(text)
    
    # Return the length of the list, which tells us how many tokens are in the text.
    return len(token_list)
