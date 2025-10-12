__all__ = ["CtcModel"]

from abc import abstractmethod
from typing import List, Protocol, Tuple, Union

from returnn.datasets.util.vocabulary import Vocabulary
from returnn.tensor import TensorDict, batch_dim
from torch import Tensor


class CtcModel(Protocol):
    @abstractmethod
    def forward_ctc(self, features: Tensor, features_len: Tensor) -> Tuple[Union[List[Tensor], Tensor], Tensor]:
        raise NotImplementedError
