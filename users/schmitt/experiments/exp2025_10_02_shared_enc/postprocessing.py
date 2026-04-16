from typing import Any, Dict, Iterator, Optional, Union
import io
import numpy as np

from returnn.tensor import Dim, Tensor, TensorDict


class AddLabelNoiseV1:
    """
    """

    preserves_num_seqs = True

    def __init__(
        self,
        data_key: str = "data",
        alpha: int = 5,
    ):

        # from returnn.datasets.util.feature_extraction import ExtractAudioFeatures
        #
        self.data_key = data_key
        self.alpha = alpha

    def __call__(self, seq_or_seq_iter: Union[TensorDict, Iterator[TensorDict]], *args, **kwargs):
        """
        """

        if isinstance(seq_or_seq_iter, TensorDict):
            return self._add_noise(seq_or_seq_iter, *args, **kwargs)
        elif isinstance(seq_or_seq_iter, Iterator):
            return (self._add_noise(seq, *args, **kwargs) for seq in seq_or_seq_iter)
        else:
            raise ValueError(
                f"invalid argument, must be {TensorDict.__class__.__name__} or iterator: {seq_or_seq_iter}"
            )

    def _add_noise(self, seq: TensorDict, *args, **kwargs) -> TensorDict:
        """
        Used here: https://arxiv.org/abs/1711.00043
        Args:
            seq:
            *args:
            **kwargs:

        Returns:

        """
        assert isinstance(seq, TensorDict)

        data: np.ndarray = seq.data[self.data_key]
        data_raw = data.raw_tensor

        x = np.arange(data_raw.shape[0])
        y = x + np.random.uniform(0, self.alpha, size=x.shape)
        idxs = np.argsort(y)
        noised_data_raw = data_raw[idxs]

        seq.data[self.data_key] = Tensor(
            name="noised_data",
            dims=[Dim(None, name="T")],
            sparse_dim=data.sparse_dim,
            dtype=data.dtype,
            raw_tensor=noised_data_raw,
        )

        seq.data[self.data_key] = Tensor(
            name="noised_data",
            dims=[Dim(None, name="T")],
            sparse_dim=data.sparse_dim,
            dtype=data.dtype,
            raw_tensor=noised_data_raw,
        )

        return seq
