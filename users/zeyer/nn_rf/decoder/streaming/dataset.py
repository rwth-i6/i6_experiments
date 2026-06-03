"""
Dataset wiring for streaming chunked-decoder training.

Combine the audio (an OggZip dataset) with the CTC forced-align HDF (per-frame
best-path labels) via a ``MetaDataset``, then post-process each seq with
:class:`returnn.datasets.postprocessing.PostprocessingDataset` to turn the
per-frame alignment into the per-seq target stream. Two ``target_mode``s:

- ``chunk_eoc``: the EOC-augmented per-chunk label sequence ``aug_targets`` (for the
  chunk-synchronous decoder; see :func:`...segmentation.chunk_augmented_targets`).
- ``rna_frame``: the per-frame RNA alignment ``rna_targets`` (one label/blank per frame,
  padded to the encoder chunk-multiple length; for frame-synchronous decoders, see
  :func:`...segmentation.rna_frame_targets`).

The target is derived on the fly so ``chunk_size`` is a config knob, not baked into HDFs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union
import functools

import numpy as np
from sisyphus import tk, Job, Task
from returnn.tensor import Dim
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from .segmentation import chunk_augmented_targets, rna_frame_targets


class ExtendVocabWithEocJob(Job):
    """Append an end-of-chunk symbol to a RETURNN ``{label: id}`` vocab file.

    The streaming decoder vocab is the spm vocab plus one extra symbol at the last
    index (EOC for chunk-sync, RNA-blank for frame-sync); ``train_v4`` requires the
    target sparse dim to carry a vocab, so we build the extended one from the spm vocab
    (e.g. ``ExtractSentencePieceVocabJob.out_vocab``). The extra symbol's name is cosmetic
    (it is stripped from recog output), so the same vocab serves both modes.
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
    Build the ``map_seq`` for :class:`PostprocessingDataset` (chunk-EOC target ``aug_targets``).

    :param vocab_ext_dim: EOC-extended target vocab (EOC at the last index).
    :param blank_idx: blank index in the alignment frames.
    :param chunk_size: chunk length in alignment/encoder frames.
    :param alignment_key: stream name of the per-frame alignment in the inner dataset.
    """
    return functools.partial(
        _chunk_augment_map_seq,
        vocab_ext_dim=vocab_ext_dim,
        blank_idx=blank_idx,
        chunk_size=chunk_size,
        alignment_key=alignment_key,
    )


def rna_frame_map_seq(
    vocab_ext_dim: Dim,
    *,
    blank_idx: int,
    chunk_size: int,
    alignment_key: str = "alignment",
) -> Callable:
    """
    Build the ``map_seq`` for :class:`PostprocessingDataset` (per-frame RNA target ``rna_targets``).

    The RNA target is padded to ``ceil(T_align/chunk_size)*chunk_size`` so it lines up
    frame-for-frame with the chunked encoder output (which pads to a full chunk).

    :param vocab_ext_dim: target vocab incl. the RNA blank at the last index.
    :param blank_idx: blank index in the alignment frames (== the last vocab index).
    :param chunk_size: chunk length in encoder frames (pad-to-multiple).
    :param alignment_key: stream name of the per-frame alignment in the inner dataset.
    """
    return functools.partial(
        _rna_frame_map_seq,
        vocab_ext_dim=vocab_ext_dim,
        blank_idx=blank_idx,
        chunk_size=chunk_size,
        alignment_key=alignment_key,
    )


class ChunkAlignDataset(DatasetConfig):
    """
    Audio (OggZip) + CTC forced-align HDF, post-processed on the fly into the per-seq
    target stream for streaming-decoder training.

    ``target_mode``:
    - ``chunk_eoc`` (default): target ``aug_targets`` (EOC-augmented per-chunk labels).
    - ``rna_frame``: target ``rna_targets`` (per-frame RNA alignment).

    Streams: ``data`` (raw audio, default input) and the target (default target, sparse
    over the extended vocab). Everything else is derived in-graph in the train step.
    """

    def __init__(
        self,
        *,
        oggzip,
        alignment_hdfs: Dict[str, Union[tk.Path, List[tk.Path]]],
        vocab_ext_dim_int: int,
        blank_idx: int,
        chunk_size: int,
        target_mode: str = "chunk_eoc",
        train_main_key: str = "train",
        dev_main_key: str = "dev-other",
        eval_subset: Optional[int] = None,
        aug_vocab: Optional[Dict[str, Any]] = None,
        train_mpd_num_workers: Optional[int] = None,
        mpd_buffer_size: int = 10,
        postproc_num_workers: int = 0,
        audio_data_key: str = "data",
        audio_has_feature_dim: bool = True,
    ):
        """
        :param oggzip: an audio-only ``LibrispeechOggZip`` (``vocab=None``); provides ``data``.
        :param alignment_hdfs: ``{main_key: forced-align out.hdf}`` (must cover train + dev keys);
            the value may be a single HDF path or a list of shard HDF paths (loaded as one HDFDataset).
        :param vocab_ext_dim_int: extended vocab size (spm vocab + 1).
        :param blank_idx: blank index in the alignment frames (== spm vocab size).
        :param chunk_size: chunk length in encoder frames (must match the encoder's chunk_size).
        :param target_mode: ``chunk_eoc`` or ``rna_frame``.
        """
        super().__init__()
        assert target_mode in ("chunk_eoc", "rna_frame"), target_mode
        self.oggzip = oggzip
        self.alignment_hdfs = alignment_hdfs
        self.vocab_ext_dim_int = vocab_ext_dim_int
        self.blank_idx = blank_idx
        self.chunk_size = chunk_size
        self.target_mode = target_mode
        self.train_main_key = train_main_key
        self.dev_main_key = dev_main_key
        self.eval_subset = eval_subset
        self.aug_vocab = aug_vocab  # RETURNN vocab opts for the target; train_v4 requires a vocab
        # if set, wrap the train MetaDataset in MPD for parallel OggZip decode (the heavy data work):
        self.train_mpd_num_workers = train_mpd_num_workers
        self.mpd_buffer_size = mpd_buffer_size
        # if > 0, parallelize the map_seq postproc (chunk/RNA target derivation) across worker procs:
        self.postproc_num_workers = postproc_num_workers
        # Audio source key within the inner audio dataset ("data" for OggZip, "audio" for the HF
        # Loquacious dataset) and whether that audio carries an explicit feature axis. Raw HF audio is
        # [B, T] (no feature dim); the streaming model squeezes/handles the missing axis (see base.py).
        self.audio_data_key = audio_data_key
        self.audio_has_feature_dim = audio_has_feature_dim

        self._time_dim = Dim(None, name="time", kind=Dim.Types.Spatial)
        self._feature_dim = (
            Dim(oggzip.audio_dim, name="audio", kind=Dim.Types.Feature) if audio_has_feature_dim else None
        )
        self._vocab_ext_dim = Dim(vocab_ext_dim_int, name="spm_ext", kind=Dim.Types.Feature)
        if target_mode == "chunk_eoc":
            self._target_name = "aug_targets"
            self._target_spatial_dim = Dim(None, name="aug_spatial", kind=Dim.Types.Spatial)
        else:
            self._target_name = "rna_targets"
            self._target_spatial_dim = Dim(None, name="rna_spatial", kind=Dim.Types.Spatial)

    def _build_map_seq(self) -> Callable:
        builder = chunk_augment_map_seq if self.target_mode == "chunk_eoc" else rna_frame_map_seq
        return builder(self._vocab_ext_dim, blank_idx=self.blank_idx, chunk_size=self.chunk_size)

    def get_default_input(self) -> str:
        return "data"

    def get_default_target(self) -> str:
        return self._target_name

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        from returnn.tensor import batch_dim

        target = {"dim_tags": [batch_dim, self._target_spatial_dim], "sparse_dim": self._vocab_ext_dim}
        if self.aug_vocab is not None:
            target["vocab"] = self.aug_vocab
        data_dims = [batch_dim, self._time_dim]
        if self._feature_dim is not None:
            data_dims.append(self._feature_dim)
        return {
            "data": {"dim_tags": data_dims},
            self._target_name: target,
        }

    def _wrap(self, main_key: str, *, training: bool, subset: Optional[int] = None) -> Dict[str, Any]:
        ogg = self.oggzip.get_dataset(main_key, training=training, subset=subset)
        align_paths = self.alignment_hdfs[main_key]
        align_files = list(align_paths) if isinstance(align_paths, (list, tuple)) else [align_paths]
        hdf = {"class": "HDFDataset", "files": align_files, "use_cache_manager": True}
        meta = {
            "class": "MetaDataset",
            "datasets": {"ogg_zip": ogg, "align": hdf},
            "data_map": {"data": ("ogg_zip", self.audio_data_key), "alignment": ("align", "data")},
            "seq_order_control_dataset": "ogg_zip",
        }
        # Parallelize the heavy OggZip decode by wrapping the *MetaDataset* (not the inner OggZip: MPD has
        # no get_current_seq_order, so it cannot be MetaDataset's seq_order_control_dataset). MetaDataset
        # delegates supports_sharding / get_current_seq_order to ogg_zip, so MPD "seq_order" sharding works.
        # Train only (dev is small). The outer train_v4 auto-MPD must be off (__multi_proc_dataset: False)
        # so this stays the single MPD layer.
        if training and self.train_mpd_num_workers:
            from i6_experiments.users.zeyer.datasets.utils.multi_proc import multi_proc_dataset_opts

            meta = multi_proc_dataset_opts(
                meta, num_workers=self.train_mpd_num_workers, buffer_size=self.mpd_buffer_size
            )
        post = {
            "class": "PostprocessingDataset",
            # Explicit "default" required: PostprocessingDataset rejects any non-default seq_ordering
            # on itself, and RETURNN would otherwise inject one (train: config "batching"; dev:
            # eval-default "sorted") into this top-level dataset. The actual order is controlled by
            # the inner ogg_zip (laplace / sorted_reverse) via seq_order_control_dataset.
            "seq_ordering": "default",
            "dataset": meta,
            "map_seq": self._build_map_seq(),
            "map_outputs": {
                "data": {"dims": [self._time_dim, self._feature_dim] if self._feature_dim is not None else [self._time_dim], "dtype": "float32"},
                self._target_name: {
                    "dims": [self._target_spatial_dim],
                    "sparse_dim": self._vocab_ext_dim,
                    "dtype": "int32",
                },
            },
        }
        # Parallelize map_seq without replicating the inner dataset: unlike MPD, PostprocessingDataset
        # instantiates the wrapped dataset only once and only fans out the map_seq across workers.
        if self.postproc_num_workers:
            post["num_workers"] = self.postproc_num_workers
        return post

    def get_train_dataset(self) -> Dict[str, Any]:
        return self._wrap(self.train_main_key, training=True)

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        return {"dev": self._wrap(self.dev_main_key, training=False, subset=self.eval_subset)}

    def get_main_name(self) -> str:
        return self.train_main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self._wrap(self.train_main_key, training=False)


# ``map_seq`` must accept ``**kwargs`` (PostprocessingDataset passes randomly named params for
# forward-compat); the bound params come via the functools.partial in the *_map_seq builders.
def _chunk_augment_map_seq(
    seq, *, rng=None, vocab_ext_dim: Dim, blank_idx: int, chunk_size: int, alignment_key: str, **kwargs
):
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


def _rna_frame_map_seq(
    seq, *, rng=None, vocab_ext_dim: Dim, blank_idx: int, chunk_size: int, alignment_key: str, **kwargs
):
    from returnn.tensor import Tensor, TensorDict

    align = seq[alignment_key]
    frames = np.asarray(align.raw_tensor).reshape(-1)
    rna = rna_frame_targets(frames, blank_idx=blank_idx, pad_to_multiple=chunk_size).astype("int32")

    out = TensorDict()
    out.data["data"] = seq["data"]
    spatial = Dim(None, name="rna_spatial")
    out.data["rna_targets"] = Tensor(
        "rna_targets", dims=[spatial], dtype="int32", sparse_dim=vocab_ext_dim, raw_tensor=rna
    )
    return out
