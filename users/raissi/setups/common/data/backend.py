__all__ = [
    "Backend",
    "BackendInfo",
]

from enum import Enum
from dataclasses import dataclass


class Backend(Enum):
    """includes different choices for both train and decode"""
    TF = "tensorflow"
    TORCH = "torch"
    ONNX = "onnx"

@dataclass(eq=True, frozen=True)
class BackendInfo:
    train: Backend
    decode: Backend

    @classmethod
    def default(cls) -> "BackendInfo":
        return BackendInfo(
            train=Backend.TF,
            decode=Backend.TF
        )
