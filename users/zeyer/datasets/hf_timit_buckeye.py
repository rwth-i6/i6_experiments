"""
HF TIMIT / Buckeye dataset wrappers for forced-align + WBE evaluation.

Both ``nh0znoisung/timit`` and ``nh0znoisung/buckeye``
ship word-level reference boundaries in ``word_detail.{utterance, start, stop}``.
The two corpora differ in how the boundary offsets are encoded:

- TIMIT: ``start``/``stop`` are sample indices at the audio sample-rate (16 kHz),
  so the conversion to seconds is ``offset / sampling_rate``.
- Buckeye: ``start``/``stop`` are in milliseconds.
  To get to seconds, multiply by 1000 then divide by ``sampling_rate``.
  Equivalently, scale by a factor of 1000 before dividing by the sample-rate,
  matching the ``dataset_offset_factors`` constant
  from :mod:`exp2025_07_07_in_grads.jobs.extract_in_grad_scores`.

This module exposes:

- :func:`get_hf_word_align_dataset_dir` --
  a hash-stable preprocessed HF dataset dir
  (adds ``text``, ``duration``, and ``id`` columns),
  suitable for both RETURNN's ``HuggingFaceDataset``
  and a downstream metrics job that reads ``word_detail``
  directly via ``datasets.load_dataset``.
- :func:`get_hf_word_align_dataset_config` --
  a :class:`DatasetConfigStatic` wrapping that dir as
  audio (raw float32) + text (SPM-tokenized int32),
  ready to pass to ``ctc_forced_align`` / ``forward_to_hdf``.
- :data:`DATASET_OFFSET_FACTORS` -- the per-corpus offset scale.
"""

from __future__ import annotations

from typing import Any, Dict
from functools import cache, partial

from sisyphus import tk

from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic, VocabConfig
from returnn.tensor import batch_dim, Dim

from i6_core.datasets.huggingface import TransformAndMapHuggingFaceDatasetJob


__all__ = [
    "DATASET_OFFSET_FACTORS",
    "get_dataset_offset_factor",
    "get_hf_word_align_dataset_config",
    "get_hf_word_align_dataset_dir",
]


_HF_REPOS = {
    "timit": "nh0znoisung/timit",
    "buckeye": "nh0znoisung/buckeye",
}

# word_detail offset unit -> multiplier to convert to "samples at sample-rate".
# TIMIT word_detail offsets are sample indices; Buckeye's are milliseconds and
# Buckeye is sampled at 16 kHz, so 1 ms == 16 samples == multiplier 1000 when
# we later divide by the (16 kHz) sample rate to land on seconds.
DATASET_OFFSET_FACTORS = {"timit": 1, "buckeye": 1000}


def _map_add_text_and_duration(example: Dict[str, Any], idx: int, *, text_case: str) -> Dict[str, Any]:
    """
    Add ``text``, ``duration`` columns derived from ``word_detail`` and ``audio``.

    Run via :class:`TransformAndMapHuggingFaceDatasetJob` ``map_func``,
    typically wrapped with :func:`functools.partial` to bind ``text_case``
    -- the bound kwarg participates in the Sis hash,
    so different case-folding choices produce distinct prepared-dataset hashes.

    :param text_case: how to case-fold the joined utterance.
        Must be one of ``"as_is"``, ``"upper"``, ``"lower"``.
        Pick to match the SPM vocab actually used downstream
        (e.g. the Loquacious-trained SPM10k model has an all-uppercase vocab,
        so its callers should pass ``"upper"``;
        a lowercase-trained LibriSpeech SPM would use ``"lower"`` instead).
        There is no default --
        the right choice depends on the vocab,
        and the silent-``<unk>`` failure mode is too easy to hit otherwise.
    """
    text = " ".join(example["word_detail"]["utterance"])
    if text_case == "upper":
        text = text.upper()
    elif text_case == "lower":
        text = text.lower()
    elif text_case == "as_is":
        pass
    else:
        raise ValueError(f"unknown text_case={text_case!r}; expected 'as_is' / 'upper' / 'lower'")
    example["text"] = text
    example["duration"] = float(len(example["audio"]["array"])) / float(example["audio"]["sampling_rate"])
    # Unique seq tag: raw ``id`` collides (TIMIT's SA1/SA2 appear across speakers), so use the row
    # index. Downstream uses this as the HDF seq-tag for forced-align + WBE matching.
    example["uid"] = str(idx)
    return example


@cache
def get_hf_word_align_dataset_dir(name: str, *, text_case: str) -> tk.Path:
    """
    Get a hash-stable preprocessed HF dataset dir for TIMIT or Buckeye.

    Adds a ``text`` column (joined ``word_detail.utterance``,
    case-folded per the ``text_case`` argument)
    and a ``duration`` column (audio length in seconds).
    Keeps the original ``word_detail`` block
    so a downstream metrics job can still read reference word boundaries
    via ``load_dataset``.

    Different ``text_case`` choices produce distinct cached dataset dirs
    (the kwarg is bound into the map_func via :func:`functools.partial`,
    which participates in the Sis hash).

    :param name: ``"timit"`` or ``"buckeye"``.
    :param text_case: ``"as_is"`` / ``"upper"`` / ``"lower"`` --
        see :func:`_map_add_text_and_duration` for the trade-offs.
        Must match the SPM vocab used downstream
        (the Loquacious SPM10k model in this repo wants ``"upper"``).
    """
    assert name in _HF_REPOS, f"unknown dataset {name!r}; expected one of {sorted(_HF_REPOS)}"
    job = TransformAndMapHuggingFaceDatasetJob(
        _HF_REPOS[name],
        map_func=partial(_map_add_text_and_duration, text_case=text_case),
        map_opts={"batched": False, "with_indices": True},
    )
    alias = f"datasets/hf_word_align/{name}-{text_case}"
    job.add_alias(alias)
    tk.register_output(alias, job.out_dir)
    return job.out_dir


def get_hf_word_align_dataset_config(
    *,
    name: str,
    split: str,
    vocab: VocabConfig,
    text_case: str,
    seq_ordering: str = "sorted_reverse",
) -> DatasetConfigStatic:
    """
    Wrap a preprocessed HF TIMIT/Buckeye split as a RETURNN ``DatasetConfig``.

    Audio = raw float32 (re-cast to 16 kHz),
    text = SPM-tokenized int32 from the case-folded word-joined transcript
    (case-folding per ``text_case``).
    ``default_input`` is ``"audio"``,
    ``default_target`` is ``"text"``,
    matching what ``ctc_forced_align`` expects.

    :param name: ``"timit"`` or ``"buckeye"``.
    :param split: HF split key, e.g. ``"val"`` / ``"test"``.
    :param vocab: SPM vocab (typically ``get_vocab_by_str("spm10k")``).
    :param text_case: ``"as_is"`` / ``"upper"`` / ``"lower"`` --
        must match what the SPM vocab expects;
        see :func:`_map_add_text_and_duration`.
    :param seq_ordering: RETURNN seq ordering.
        Default sorts by duration desc.
    """
    hf_data_dir = get_hf_word_align_dataset_dir(name, text_case=text_case)
    vocab_opts = vocab.get_opts()
    # Explicit ``sparse_dim`` for downstream callers (e.g. ``ctc_forced_align``)
    # that read it off ``get_extern_data()`` without consulting the vocab itself.
    classes_dim = Dim(vocab.get_num_classes(), name="vocab")
    extern_data_dict = {
        "audio": {"dtype": "float32", "dim_tags": [batch_dim, Dim(None, name="time")]},
        "text": {
            "dtype": "int32",
            "dim_tags": [batch_dim, Dim(None, name="text_spatial")],
            "sparse": True,
            "sparse_dim": classes_dim,
            "vocab": vocab_opts,
        },
    }
    main_dataset = {
        "class": "HuggingFaceDataset",
        "dataset_opts": hf_data_dir.join_right(split),
        "use_file_cache": True,
        "seq_tag_column": "uid",  # unique row-index tag; ``id`` is non-unique (TIMIT SA1/SA2 etc.)
        "sorting_seq_len_column": "duration",
        "cast_columns": {"audio": {"_type": "Audio", "sample_rate": 16_000}},
        # Keep data_format consistent with extern_data_dict.
        "data_format": {
            "audio": {"dtype": "float32", "shape": [None]},
            "text": {"dtype": "int32", "shape": [None], "sparse": True, "vocab": vocab_opts},
        },
        "seq_ordering": seq_ordering,
    }
    return DatasetConfigStatic(
        main_name=split,
        main_dataset=main_dataset,
        extern_data=extern_data_dict,
        default_input="audio",
        default_target="text",
        use_deep_copy=True,
    )


# Convenience pass-through to keep call sites short.
def get_dataset_offset_factor(name: str) -> int:
    """
    Return the per-corpus offset multiplier for word_detail.{start,stop}.

    See module docstring.
    Used by the WBE-metric job
    to map raw offsets to seconds via ``offset * factor / sampling_rate``.
    """
    assert name in DATASET_OFFSET_FACTORS, name
    return DATASET_OFFSET_FACTORS[name]
