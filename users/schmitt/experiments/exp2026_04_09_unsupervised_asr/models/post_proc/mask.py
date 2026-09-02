from typing import Union, Iterable, Iterator, Optional, Dict, Any, Callable
import os

import numpy as np

from returnn.tensor import TensorDict


class MaskPhonemeSequence:
    """
    Randomly mask spans in a phoneme sequence and replace each contiguous
    masked region with a single mask token.

    The input sequence is assumed to have no batch dimension and to have
    shape (T,) or (T, ...).
    """

    preserves_num_seqs = True

    def __init__(
        self,
        data_key: str,
        mask_prob: float,
        min_span: int,
        max_span: int,
        mask_value: Union[int, float, np.ndarray],
        **unused_kwargs,
    ):
        """
        Args:
            data_key:
                Key of the sequence in the TensorDict.
            mask_prob:
                Approximate fraction of sequence positions to mask.
            min_span:
                Minimum sampled mask-span length.
            max_span:
                Maximum sampled mask-span length.
            mask_value:
                Value used to replace each contiguous masked region. For data
                of shape (T, D, ...), this must be broadcastable to (D, ...).
        """
        self.data_key = data_key

        assert 0.0 <= mask_prob <= 1.0
        assert min_span >= 1
        assert max_span >= min_span

        self.mask_prob = mask_prob
        self.min_span = min_span
        self.max_span = max_span
        self.mask_value = mask_value

    def __call__(
        self,
        seq_or_iterator: Union[TensorDict, Iterator[TensorDict]],
        *args,
        **kwargs,
    ):
        """Apply masking to one sequence or lazily to an iterator."""
        if isinstance(seq_or_iterator, Iterator):
            return (self.mask(seq) for seq in seq_or_iterator)

        assert isinstance(seq_or_iterator, TensorDict)
        return self.mask(seq_or_iterator)

    def mask(self, tensor_dict: TensorDict) -> TensorDict:
        """
        Randomly mask the configured sequence and update the TensorDict
        in place.
        """
        source_tensor = tensor_dict.data[self.data_key]
        data = np.asarray(source_tensor.raw_tensor)

        if data.ndim < 1:
            raise ValueError(f"Expected {self.data_key!r} to have shape (T, ...), but got {data.shape}")

        seq_len = data.shape[0]

        if seq_len == 0:
            return tensor_dict

        seed = int.from_bytes(os.urandom(4), byteorder="little")
        rng = np.random.default_rng(seed)

        num_to_mask = int(np.ceil(seq_len * self.mask_prob))

        mask_lens = rng.integers(
            low=self.min_span,
            high=self.max_span + 1,
            size=seq_len,
            dtype=np.int64,
        )

        # Keep complete sampled spans only while their cumulative length
        # remains within the masking budget.
        mask_lens_cumsum = np.cumsum(mask_lens)
        mask_lens[mask_lens_cumsum > num_to_mask] = 0
        mask_lens = mask_lens[mask_lens > 0]

        # True means that the original position is retained.
        mask = np.ones(seq_len, dtype=bool)

        if mask_lens.size > 0:
            max_start = seq_len - int(mask_lens.max())

            mask_starts = rng.integers(
                low=0,
                high=max_start + 1,
                size=mask_lens.size,
                dtype=np.int64,
            )

            for start, span_len in zip(mask_starts, mask_lens):
                start = int(start)
                span_len = int(span_len)
                mask[start : start + span_len] = False

        result_elements = []
        position = 0

        mask_value = np.asarray(self.mask_value, dtype=data.dtype)
        if data.ndim > 1:
            try:
                mask_value = np.broadcast_to(mask_value, data.shape[1:])
            except ValueError as exc:
                raise ValueError(
                    f"mask_value with shape {mask_value.shape} cannot be broadcast to feature shape {data.shape[1:]}"
                ) from exc

        while position < seq_len:
            if mask[position]:
                result_elements.append(data[position])
                position += 1
            else:
                # Replace the complete contiguous masked region with one
                # mask value.
                result_elements.append(mask_value)

                while position < seq_len and not mask[position]:
                    position += 1

        masked_data = np.asarray(result_elements, dtype=data.dtype)

        tensor = source_tensor.copy_template(dtype=source_tensor.dtype)
        tensor.raw_tensor = masked_data
        tensor_dict.data[self.data_key] = tensor

        return tensor_dict
