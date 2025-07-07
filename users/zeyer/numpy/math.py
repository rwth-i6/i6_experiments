import numpy as np
from typing import Optional


def log_softmax(x: np.ndarray, *, axis: Optional[int]) -> np.ndarray:
    max_score = np.max(x, axis=axis, keepdims=True)
    x = x - max_score
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    # log_sigmoid(x) = -log(1 + exp(-x)) = -log1p(exp(-x))
    return np.log1p(np.exp(-x))
