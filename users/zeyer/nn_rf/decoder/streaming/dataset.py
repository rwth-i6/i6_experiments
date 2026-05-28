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

from typing import Any, Callable, Dict, Optional
import functools

import numpy as np
from sisyphus import tk, Job, Task
from returnn.tensor import Dim
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from .segmentation import chunk_augmented_targets


class ExtendVocabWithEocJob(Job):
    """Append an end-of-chunk symbol to a RETURNN ``{label: id}`` vocab file.

    The streaming decoder vocab is the spm vocab plus one EOC marker at the last
    index; ``train_v4`` requires the target sparse dim to carry a vocab, so we build
    the extended one from the spm vocab (e.g. ``ExtractSentencePieceVocabJob.out_vocab``).
    """

    def __init__(self, vocab_file: tk.Path, *, eoc_label: str = "<eoc>"):
        self.vocab_file = vocab_file
        self.eoc_label = eoc_label
        self.out_vocab = self.output_path("vocab_with_eoc.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import ast

        with open(self.vocab_file.get_path()) as f:
            d = ast.literal_eval(f.read())
        assert isinstance(d, dict) and self.eoc_label not in d, f"bad/duplicate vocab for {self.eoc_label!r}"
        d[self.eoc_label] = max(d.values()) + 1
        with open(self.out_vocab.get_path(), "w") as f:
            f.write("{\n")
            for label, idx in sorted(d.items(), key=lambda kv: kv[1]):
                f.write(f"{label!r}: {idx},\n")
            f.write("}\n")


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


class ChunkAlignDataset(DatasetConfig):
    """
    Audio (OggZip) + CTC forced-align HDF, post-processed on the fly into the
    EOC-augmented target stream ``aug_targets`` for streaming chunked-decoder training.

    Streams: ``data`` (raw audio, default input) and ``aug_targets`` (sparse over the
    EOC-extended vocab, default target). The raw spm labels + per-position chunk index
    are derived in-graph in the train step (see :func:`...chunkwise.chunkwise_train_forward`).
    """

    def __init__(
        self,
        *,
        oggzip,
        alignment_hdfs: Dict[str, tk.Path],
        vocab_ext_dim_int: int,
        blank_idx: int,
        chunk_size: int,
        train_main_key: str = "train",
        dev_main_key: str = "dev-other",
        eval_subset: Optional[int] = None,
        aug_vocab: Optional[Dict[str, Any]] = None,
    ):
        """
        :param oggzip: an audio-only ``LibrispeechOggZip`` (``vocab=None``); provides ``data``.
        :param alignment_hdfs: ``{main_key: forced-align out.hdf}`` (must cover train + dev keys).
        :param vocab_ext_dim_int: EOC-extended vocab size (spm vocab + 1).
        :param blank_idx: blank index in the alignment frames (== spm vocab size).
        :param chunk_size: chunk length in encoder frames (must match the encoder's chunk_size).
        """
        super().__init__()
        self.oggzip = oggzip
        self.alignment_hdfs = alignment_hdfs
        self.vocab_ext_dim_int = vocab_ext_dim_int
        self.blank_idx = blank_idx
        self.chunk_size = chunk_size
        self.train_main_key = train_main_key
        self.dev_main_key = dev_main_key
        self.eval_subset = eval_subset
        self.aug_vocab = aug_vocab  # RETURNN vocab opts for aug_targets (spm + EOC); train_v4 requires a vocab

        self._time_dim = Dim(None, name="time", kind=Dim.Types.Spatial)
        self._feature_dim = Dim(oggzip.audio_dim, name="audio", kind=Dim.Types.Feature)
        self._aug_spatial_dim = Dim(None, name="aug_spatial", kind=Dim.Types.Spatial)
        self._vocab_ext_dim = Dim(vocab_ext_dim_int, name="spm_ext", kind=Dim.Types.Feature)

    def get_default_input(self) -> str:
        return "data"

    def get_default_target(self) -> str:
        return "aug_targets"

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        from returnn.tensor import batch_dim

        aug_targets = {"dim_tags": [batch_dim, self._aug_spatial_dim], "sparse_dim": self._vocab_ext_dim}
        if self.aug_vocab is not None:
            aug_targets["vocab"] = self.aug_vocab
        return {
            "data": {"dim_tags": [batch_dim, self._time_dim, self._feature_dim]},
            "aug_targets": aug_targets,
        }

    def _wrap(self, main_key: str, *, training: bool, subset: Optional[int] = None) -> Dict[str, Any]:
        ogg = self.oggzip.get_dataset(main_key, training=training, subset=subset)
        hdf = {"class": "HDFDataset", "files": [self.alignment_hdfs[main_key]], "use_cache_manager": True}
        meta = {
            "class": "MetaDataset",
            "datasets": {"ogg_zip": ogg, "align": hdf},
            "data_map": {"data": ("ogg_zip", "data"), "alignment": ("align", "data")},
            "seq_order_control_dataset": "ogg_zip",
        }
        return {
            "class": "PostprocessingDataset",
            # Explicit "default" required: PostprocessingDataset rejects any non-default seq_ordering
            # on itself, and RETURNN would otherwise inject one (train: config "batching"; dev:
            # eval-default "sorted") into this top-level dataset. The actual order is controlled by
            # the inner ogg_zip (laplace / sorted_reverse) via seq_order_control_dataset.
            "seq_ordering": "default",
            "dataset": meta,
            "map_seq": chunk_augment_map_seq(self._vocab_ext_dim, blank_idx=self.blank_idx, chunk_size=self.chunk_size),
            "map_outputs": {
                "data": {"dims": [self._time_dim, self._feature_dim], "dtype": "float32"},
                "aug_targets": {"dims": [self._aug_spatial_dim], "sparse_dim": self._vocab_ext_dim, "dtype": "int32"},
            },
        }

    def get_train_dataset(self) -> Dict[str, Any]:
        return self._wrap(self.train_main_key, training=True)

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        return {"dev": self._wrap(self.dev_main_key, training=False, subset=self.eval_subset)}

    def get_main_name(self) -> str:
        return self.train_main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self._wrap(self.train_main_key, training=False)


# ``map_seq`` must accept ``**kwargs`` (PostprocessingDataset passes randomly named params for
# forward-compat); the bound params come via the functools.partial in chunk_augment_map_seq.
def _chunk_augment_map_seq(seq, *, rng=None, vocab_ext_dim: Dim, blank_idx: int, chunk_size: int, alignment_key: str, **kwargs):
    from returnn.tensor import Tensor, TensorDict

    align = seq[alignment_key]
    frames = np.asarray(align.raw_tensor).reshape(-1)
    eoc_idx = vocab_ext_dim.dimension - 1
    aug, _ = chunk_augmented_targets(frames, blank_idx=blank_idx, chunk_size=chunk_size, eoc_idx=eoc_idx)
    aug = aug.astype("int32")

    out = TensorDict()
    out.data["data"] = seq["data"]
    # Dynamic spatial dim (dimension=None): the per-seq length comes from raw_tensor;
    # PostprocessingDataset requires the mapped tensor's dim to be dynamic to match map_outputs.
    spatial = Dim(None, name="aug_spatial")
    out.data["aug_targets"] = Tensor(
        "aug_targets", dims=[spatial], dtype="int32", sparse_dim=vocab_ext_dim, raw_tensor=aug
    )
    return out
