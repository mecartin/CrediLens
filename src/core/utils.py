import time
from functools import wraps
from typing import Callable, Any
from .logger import logger

def timer(func: Callable) -> Callable:
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if it doesn't."""
    import os
    os.makedirs(path, exist_ok=True)
    return path
