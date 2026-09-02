from typing import Union, Iterable, Iterator, Optional, Dict, Any, Callable
import os

import numpy as np

from returnn.tensor import TensorDict


class DuplicatePhonemeSequence:
    """ """

    preserves_num_seqs = True

    def __init__(
        self,
        data_key: str,
        min_dup: int,
        max_dup: int,
        **unused_kwargs,
    ):
        """ """

        self.data_key = data_key

        assert min_dup >= 1 and max_dup >= min_dup
        self.min_dup = min_dup
        self.max_dup = max_dup

    def __call__(self, seq_or_iterator: Union[TensorDict, Iterator[TensorDict]], *args, **kwargs):
        """ """

        if isinstance(seq_or_iterator, Iterator):
            return (self.duplicate(seq) for seq in seq_or_iterator)
        else:
            assert isinstance(seq_or_iterator, TensorDict)
            return self.duplicate(seq_or_iterator)

    def duplicate(
        self,
        tensor_dict: TensorDict,
    ):
        data: np.ndarray = tensor_dict.data[self.data_key].raw_tensor.astype(np.int32)

        seed = int.from_bytes(os.urandom(4), byteorder="little")
        rng = np.random.default_rng(seed)

        durations = rng.integers(
            low=self.min_dup,
            high=self.max_dup + 1,
            size=data.shape[0],
            dtype=np.int64,
        )

        expanded = np.repeat(data, durations)

        tensor = tensor_dict.data[self.data_key].copy_template(dtype="int32")
        tensor.raw_tensor = expanded
        tensor_dict.data[self.data_key] = tensor

        return tensor_dict
