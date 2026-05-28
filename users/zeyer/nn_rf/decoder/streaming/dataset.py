"""
Dataset wiring for streaming chunked-decoder training.

Combine the audio (an OggZip dataset) with the CTC forced-align HDF (per-frame
best-path labels) via a ``MetaDataset``, then post-process each seq with
:class:`returnn.datasets.postprocessing.PostprocessingDataset` to turn the
per-frame alignment into the EOC-augmented target stream ``aug_targets`` (the
single supervision stream the streaming decoder trains on; see
:mod:`.segmentation`). The chunking is done on the fly so ``chunk_size`` is a
config knob, not baked into a precomputed HDF.
"""

from __future__ import annotations

from typing import Any, Callable, Dict
import functools

import numpy as np
from returnn.tensor import Dim

from .segmentation import chunk_augmented_targets


def chunk_augment_map_seq(
    vocab_ext_dim: Dim,
    *,
    blank_idx: int,
    chunk_size: int,
    alignment_key: str = "alignment",
) -> Callable:
    """
    Build the ``map_seq`` for :class:`PostprocessingDataset`.

    :param vocab_ext_dim: EOC-extended target vocab (EOC at the last index).
    :param blank_idx: blank index in the alignment frames.
    :param chunk_size: chunk length in alignment/encoder frames.
    :param alignment_key: stream name of the per-frame alignment in the inner dataset.
    :return: ``(seq: TensorDict, *, rng, **kwargs) -> TensorDict`` mapping
        ``{data, <alignment_key>}`` -> ``{data, aug_targets}``.
    """
    return functools.partial(
        _chunk_augment_map_seq,
        vocab_ext_dim=vocab_ext_dim,
        blank_idx=blank_idx,
        chunk_size=chunk_size,
        alignment_key=alignment_key,
    )


def chunk_augment_map_outputs(audio_data: Dict[str, Any], vocab_ext_dim: Dim) -> Dict[str, Dict[str, Any]]:
    """``map_outputs`` for the post-processor: passthrough audio + the new sparse ``aug_targets``."""
    return {
        "data": audio_data,
        "aug_targets": {"dims": [Dim(None, name="aug_spatial")], "sparse_dim": vocab_ext_dim, "dtype": "int32"},
    }


# ``map_seq`` must accept ``**kwargs`` (PostprocessingDataset passes randomly named params for
# forward-compat); the bound params come via the functools.partial above.
def _chunk_augment_map_seq(seq, *, rng=None, vocab_ext_dim: Dim, blank_idx: int, chunk_size: int, alignment_key: str, **kwargs):
    from returnn.tensor import Tensor, TensorDict

    align = seq[alignment_key]
    frames = np.asarray(align.raw_tensor).reshape(-1)
    eoc_idx = vocab_ext_dim.dimension - 1
    aug, _ = chunk_augmented_targets(frames, blank_idx=blank_idx, chunk_size=chunk_size, eoc_idx=eoc_idx)
    aug = aug.astype("int32")

    out = TensorDict()
    out.data["data"] = seq["data"]
    spatial = Dim(int(aug.shape[0]), name="aug_spatial")
    out.data["aug_targets"] = Tensor(
        "aug_targets", dims=[spatial], dtype="int32", sparse_dim=vocab_ext_dim, raw_tensor=aug
    )
    return out
