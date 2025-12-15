"""Retry helper with exponential backoff for external API calls."""

import asyncio
import logging
import random
from typing import Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0


class RetryableError(Exception):
    """Error that can be retried."""
    
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


async def with_retry(
    func: Callable,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    operation_name: str = "operation",
    **kwargs,
) -> T:
    """Execute an async function with exponential backoff retry.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay between retries
        operation_name: Name of operation for logging
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of successful function execution
        
    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            
            if status not in RETRYABLE_STATUS_CODES:
                logger.warning(f"{operation_name}: Non-retryable HTTP {status}")
                raise
            
            last_exception = e
            
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_exception = e
            
        except RetryableError as e:
            last_exception = e
            
        except Exception as e:
            logger.warning(f"{operation_name}: Non-retryable error: {type(e).__name__}")
            raise
        
        if attempt < max_retries - 1:
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.3)
            total_delay = delay + jitter
            
            logger.info(
                f"{operation_name}: Attempt {attempt + 1}/{max_retries} failed, "
                f"retrying in {total_delay:.1f}s"
            )
            await asyncio.sleep(total_delay)
    
    logger.error(f"{operation_name}: All {max_retries} attempts exhausted")
    if last_exception:
        raise last_exception
    raise RuntimeError(f"{operation_name}: Failed after {max_retries} attempts")


def get_retry_client(timeout: float = 20.0) -> httpx.AsyncClient:
    """Get an httpx AsyncClient configured with appropriate timeout.
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        Configured httpx.AsyncClient
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
    )
