import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logger.info("function:%r took: %2.5f sec", f.__name__, end - start)
        return result

    return wrapper
