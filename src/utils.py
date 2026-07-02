import time 
import logging 
import functools 
import tiktoken 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sec_rag_utils") 
def profile_time(func):
    """
    A decorator that measures the time it takes for a function to execute and logs it.
    This helps us identify which parts of our RAG pipeline are slow (e.g., database queries vs. embeddings).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"Function '{func.__name__}' completed execution in {duration:.4f} seconds.")
        return result
    return wrapper
class ProfilerContext:
    """
    A Context Manager that allows profiling of specific blocks of code using the 'with' statement.
    This is useful for profiling a small subset of lines inside a large function.
    """
    def __init__(self, block_name: str):
        self.block_name = block_name
        self.start_time = None
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        logger.info(f"Block '{self.block_name}' took {duration:.4f} seconds to run.")
def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    Counts the number of tokens in a given text using tiktoken.
    Defaults to the gpt-4o-mini tokenizer.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Model '{model_name}' not recognized by tiktoken. Falling back to 'cl100k_base' encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    token_list = encoding.encode(text)
    return len(token_list)
