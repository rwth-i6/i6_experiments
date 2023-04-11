__all__ = ["OutputLayerType"]

from enum import Enum


class OutputLayerType(Enum):
    SOFTMAX = "softmax"
    LOG_SOFTMAX = "log_softmax"
