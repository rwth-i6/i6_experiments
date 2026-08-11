from typing import Protocol
from abc import ABC, abstractmethod

import numpy as np

class UpdaterBase_(ABC):
    @abstractmethod
    def update(self, features: np.ndarray, idxs: np.ndarray) -> None:
        ...

    @abstractmethod
    def get_model(self):
        ...