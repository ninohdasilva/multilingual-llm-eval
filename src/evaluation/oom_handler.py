"""Out-of-memory (OOM) handling with automatic batch size reduction."""

import logging
from typing import Callable, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_oom_retry(
    func: Callable[[int], T],
    initial_batch_size: int,
    min_batch_size: int = 1,
    max_retries: int = 3,
) -> tuple[T, int]:
    """
    Execute a function with automatic batch size reduction on OOM.

    Args:
        func: Function that takes batch_size as argument and may raise OOM
        initial_batch_size: Starting batch size
        min_batch_size: Minimum batch size to try (default: 1)
        max_retries: Maximum number of retries (default: 3)

    Returns:
        Tuple of (result, final_batch_size_used)

    Raises:
        RuntimeError: If OOM persists after all retries
    """
    batch_size = initial_batch_size
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Clear CUDA cache before attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            result = func(batch_size)
            if attempt > 0:
                logger.info(
                    f"Successfully executed with batch_size={batch_size} after {attempt} retries"
                )
            return result, batch_size

        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                last_error = e
                if batch_size <= min_batch_size:
                    logger.error(
                        f"OOM error persists even with batch_size={batch_size}. Cannot reduce further."
                    )
                    raise RuntimeError(
                        f"Out of memory error persists after {attempt + 1} attempts. "
                        f"Minimum batch size {min_batch_size} reached."
                    ) from e

                # Halve the batch size
                new_batch_size = max(min_batch_size, batch_size // 2)
                logger.warning(
                    f"OOM error on attempt {attempt + 1} with batch_size={batch_size}. "
                    f"Retrying with batch_size={new_batch_size}"
                )
                batch_size = new_batch_size

                # Clear cache before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Not an OOM error, re-raise
                raise

    # If we get here, all retries failed
    raise RuntimeError(
        f"Failed after {max_retries + 1} attempts with OOM errors. "
        f"Last batch_size tried: {batch_size}"
    ) from last_error

