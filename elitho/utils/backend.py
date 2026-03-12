import os
import numpy as np


def get_array_module(arr):
    """Get appropriate array module from array (Array-based Dispatch pattern).

    Args:
        arr: numpy or cupy array

    Returns:
        numpy or cupy module
    """
    try:
        import cupy as cp
        return cp.get_array_module(arr)
    except ImportError:
        return np


def setup_backend(use_gpu=None):
    """Setup backend and return appropriate module.

    Args:
        use_gpu: Explicitly specify True/False, or None to read USE_GPU env var

    Returns:
        numpy or cupy module
    """
    if use_gpu is None:
        use_gpu = os.getenv('USE_GPU', '0') == '1'

    if use_gpu:
        try:
            import cupy as cp
            return cp
        except ImportError:
            return np
    else:
        return np
