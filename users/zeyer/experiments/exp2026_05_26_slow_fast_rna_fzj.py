"""
Slow-fast-RNA streaming chunked-decoder experiments.

Run from ``~/setups/2026-05-26-fast-slow-rna/``. Variants share the chunked Conformer
encoder + base-model CTC forced alignment, and differ only in the decoder:

- ``chunkwise``: chunk-synchronous decoder (EOC-augmented per-chunk targets), chunk-masked
  cross-attention.
- ``framewise``: frame-synchronous RNA fast-only decoder (one label/blank per encoder
  frame); the fast stack that ``ext_transducer`` will extend with a slow label-rate stack.

All wired via :func:`_train_streaming_variant` (train + dev-other greedy recog -> WER), with
**consistent settings across variants** for a fair comparison: speed-pert OFF (the forced
alignment is on un-perturbed audio; the planned co-perturb variant will re-enable it
consistently) and length filtering by audio duration (not the inherited target-length cap,
which is wrong for the long per-frame targets).
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

from sisyphus import tk
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerConvSubsample

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module, disable_register_output
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg
from i6_experiments.users.zeyer import train_v4
from i6_experiments.users.zeyer.datasets.librispeech import (
    LibrispeechOggZip,
    _raw_audio_opts,
    get_vocab_by_str,
    get_librispeech_task_raw_v2,
)
from i6_experiments.users.zeyer.recog_batched import recog_training_exp_batched
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import _raw_sample_rate
from i6_experiments.users.zeyer.experiments.exp2025_10_21_chunked_ctc import _aed_ctc_forced_align
from i6_experiments.users.zeyer.experiments.exp2026_05_26_base_fzj import (
    _train_librispeech_base,
    _train_loquacious_base,
)
from i6_experiments.users.zeyer.datasets.loquacious import get_loquacious_task_raw_v2
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
    ChunkedConformerEncoderV2,
    ChunkedConformerEncoderLayerV2,
    ChunkedRotaryPosSelfAttentionV2,
)
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.base import streaming_model_def
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.chunkwise import (
    ChunkwiseDecoder,
    chunkwise_training,
    model_recog as chunkwise_model_recog,
)
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.framewise import (
    FramewiseDecoder,
    framewise_training,
    model_recog as framewise_model_recog,
)
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.ext_transducer import (
    ExtTransducerDecoder,
    ext_transducer_training,
    model_recog as ext_transducer_model_recog,
)
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.two_tower import (
    TwoTowerDecoder,
    two_tower_training,
    model_recog as two_tower_model_recog,
)
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.dataset import ChunkAlignDataset, ExtendVocabWithEocJob

# Prefix for alias/ and output/ paths. Does not enter any Job hash (only naming);
# the helper walks the module hierarchy for this attribute.
__setup_root_prefix__ = "exp2026_05_26_slow_fast_rna_fzj"

_CHUNK_SIZE = 10  # encoder frames (60ms each) -> 600ms chunks

# Loquacious scale-up: by default align + train a bounded subset of the large train split (validates
# the HF-align -> ChunkAlignDataset -> train path end to end). Flip _LOQ_FULL_TRAIN to align + train
# the full ~25k h split (sharded gpu=4). Subset selection is deterministic
# (get_hf_random_sorted_subset_v2), so the align HDFs and the train audio see the identical seq set.
_LOQ_SUBSET_TRAIN_SEQS = 50_000
_LOQ_FULL_TRAIN = True


def py():
    _validate_batched_forced_align()
    _train_chunkwise_smoke()
    _train_framewise_smoke()
    _train_ext_transducer_smoke()
    _train_two_tower_smoke()
    # Loquacious scale-up (subset smoke by default; full ~25k h gated behind _LOQ_FULL_TRAIN).
    _train_chunkwise_loq_smoke()
    _train_framewise_loq_smoke()
    _train_ext_transducer_loq_smoke()
    _train_two_tower_loq_smoke()


def _ls_align_hdfs(model, *, keys, aux_ctc_layer: int = 16, num_shards: int = 8) -> Dict[str, List[tk.Path]]:
    """Per-frame CTC forced-align shard HDFs per LS split, fanned out across the full 4-GPU node.

    Multi-GPU only: FZJ flat-per-node billing forbids single-GPU forward jobs, so the forced-align
    is sharded via ``num_shards`` (batched engine) and returns the list of shard HDFs per key
    (loaded as one HDFDataset by ChunkAlignDataset).
    """
    vocab = get_vocab_by_str("spm10k")
    out = {}
    for key in keys:
        ds = LibrispeechOggZip(audio=_raw_audio_opts.copy(), audio_dim=1, vocab=vocab, main_key=key)
        out[key] = _aed_ctc_forced_align(model, ds, aux_ctc_layer=aux_ctc_layer, num_shards=num_shards)
    return out


def _validate_batched_forced_align(*, num_shards: int = 8, aux_ctc_layer: int = 16) -> None:
    """Smoke-test the multi-GPU BatchedReturnnForwardJob on a small LS split before the large Loquacious run.

    Borrows the LS-base model and forced-aligns dev-other (the smallest split, ~2.9 h) with
    ``num_shards`` so the same forward fans out across the full 4-GPU node. This exercises the parts
    of BatchedReturnnForwardJob that have no other coverage -- worker spawn, round-robin shard assignment,
    atomic per-shard HDF write + existing-HDF resume-skip, and the walltime barrier-stop / sis
    resubmit -- on a real model + data. num_shards=8 (> 4 GPUs) gives each GPU two rounds, so the
    round-robin and the per-shard-time stop heuristic both get touched even on this tiny split.
    """
    prefix = get_setup_prefix_for_module(__name__)
    vocab = get_vocab_by_str("spm10k")
    with disable_register_output():
        base_model = _train_librispeech_base().get_last_fixed_epoch()
    ds = LibrispeechOggZip(audio=_raw_audio_opts.copy(), audio_dim=1, vocab=vocab, main_key="dev-other")
    shard_hdfs = _aed_ctc_forced_align(base_model, ds, aux_ctc_layer=aux_ctc_layer, num_shards=num_shards)
    for i, hdf in enumerate(shard_hdfs):
        tk.register_output(prefix + "/batched-align-validate/dev-other/shard_%03i.hdf" % i, hdf)


def _loq_vocab():
    """The Loquacious-native spm10k (a distinct SentencePiece model from the LibriSpeech spm10k)."""
    from i6_experiments.users.zeyer.datasets.loquacious import get_spm_vocab

    return get_spm_vocab(dim="10k")


def _loq_dataset_config(main_key: str, *, subset_seqs: Optional[int]):
    """A Loquacious HF audio+text DatasetConfig for one split (audio = raw [B, T], seq-tag = ``id``).

    ``subset_seqs`` selects a deterministic random-sorted subset (so the align HDF and the train audio
    cover the identical seqs); ``None`` uses the full split (train via DistributeFilesDataset).
    """
    from i6_experiments.users.zeyer.datasets.loquacious import _make_hf_dataset, get_loquacious_hf_ogg

    hf_dir = get_loquacious_hf_ogg(name="large")
    split = {"train": "train", "dev": "dev"}[main_key]
    if subset_seqs is not None:
        return _make_hf_dataset(
            hf_data_dir=hf_dir,
            split=split,
            vocab=_loq_vocab(),
            take_random_sorted_subset=subset_seqs,
            take_random_sorted_subset_version=2,
        )
    return _make_hf_dataset(hf_data_dir=hf_dir, split=split, vocab=_loq_vocab(), use_distrib_files=(split == "train"))


class _LoqAlignSource(DatasetConfig):
    """Loquacious HF dataset as a forced-align source: re-exposes the wrapped split but adds the
    explicit target ``sparse_dim`` that ``_make_hf_dataset`` omits and ``_aed_ctc_forced_align`` needs.
    """

    def __init__(self, base: DatasetConfig, *, classes_dim):
        super().__init__()
        self._base = base
        self._classes_dim = classes_dim

    def get_extern_data(self) -> Dict[str, Any]:
        import copy

        d = copy.deepcopy(self._base.get_extern_data())
        d[self._base.get_default_target()]["sparse_dim"] = self._classes_dim
        return d

    def get_default_input(self) -> str:
        return self._base.get_default_input()

    def get_default_target(self) -> str:
        return self._base.get_default_target()

    def get_main_name(self) -> str:
        return self._base.get_main_name()

    def get_main_dataset(self) -> Dict[str, Any]:
        return self._base.get_main_dataset()


class _LoqAudioProvider:
    """Audio-only Loquacious provider for ChunkAlignDataset (mirrors the OggZip interface it needs:
    ``audio_dim`` + ``get_dataset(main_key, training, subset)``). HF audio is raw [B, T] -> pair with
    ``audio_has_feature_dim=False`` so ChunkAlignDataset declares no feature axis.
    """

    audio_dim = 1

    def __init__(self, *, train_subset_seqs: Optional[int]):
        self._train_subset_seqs = train_subset_seqs

    def get_dataset(self, main_key: str, *, training: bool, subset: Optional[int] = None) -> Dict[str, Any]:
        subset_seqs = self._train_subset_seqs if main_key == "train" else None
        return _loq_dataset_config(main_key, subset_seqs=subset_seqs).main_dataset


def _loq_align_hdfs(
    model, *, num_shards: int, aux_ctc_layer: int, subset_seqs: Optional[int], keys=("train", "dev")
) -> Dict[str, List[tk.Path]]:
    """Sharded multi-GPU CTC forced-align HDFs for the Loquacious train (subset) + full dev splits.

    The train subset here must match :class:`_LoqAudioProvider`'s train subset (same ``subset_seqs``)
    so the alignment HDF and the training audio cover the identical seqs. ``keys`` selects which splits
    to build (the full-train path takes only ``dev`` here; its train alignment is co-sharded instead, via
    :func:`_loq_coshard_align_dir`).
    """
    from returnn.tensor import Dim

    vocab = _loq_vocab()
    classes_dim = Dim(vocab.get_num_classes(), name="spm", kind=Dim.Types.Feature)
    key_subset = {"train": subset_seqs, "dev": None}
    out = {}
    for key in keys:
        src = _LoqAlignSource(_loq_dataset_config(key, subset_seqs=key_subset[key]), classes_dim=classes_dim)
        out[key] = _aed_ctc_forced_align(model, src, aux_ctc_layer=aux_ctc_layer, num_shards=num_shards)
    return out


def _loq_make_shard_align_dataset(arrow_files, *, hf_data_dir, classes_dim):
    """Forced-align source over one audio arrow shard's file(s): the loq HF audio+text dataset restricted
    to ``arrow_files`` (reusing the audio DistributeFilesDataset's per-subepoch builder), plus the explicit
    target ``sparse_dim`` that :func:`_aed_ctc_forced_align` needs. Module-level + picklable (bound to a
    tk.Path + Dim) so it can be the ``make_dataset`` of :func:`batched_forward_per_arrow_shard_to_hdf`.
    """
    import copy
    from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic
    from i6_experiments.users.zeyer.datasets.loquacious import _make_hf_dataset

    base = _make_hf_dataset(hf_data_dir=hf_data_dir, split="train", vocab=_loq_vocab(), use_distrib_files=True)
    audio_dict = base.main_dataset["get_sub_epoch_dataset"](arrow_files)
    extern = copy.deepcopy(base.extern_data)
    extern[base.default_target]["sparse_dim"] = classes_dim
    return DatasetConfigStatic(
        main_name="train",
        main_dataset=audio_dict,
        extern_data=extern,
        default_input=base.default_input,
        default_target=base.default_target,
        use_deep_copy=True,
    )


def _loq_coshard_align_dir(model, *, aux_ctc_layer: int) -> tk.Path:
    """One CTC forced-align HDF per audio arrow shard (co-named), as a single dir, via the dynamic batched
    forward job. Mirrors :func:`_aed_ctc_forced_align`'s forward config, but enumerates cells per arrow
    shard so the HDFs pair 1:1 with the audio DistributeFilesDataset shards for the co-shard train path.
    """
    from returnn.tensor import Dim
    from i6_experiments.users.zeyer.forward_batched import batched_forward_per_arrow_shard_to_hdf
    from i6_experiments.users.zeyer.datasets.loquacious import get_loquacious_hf_ogg
    from i6_experiments.users.zeyer.experiments.exp2025_10_21_chunked_ctc import _aed_ctc_model_forced_align_step

    vocab = _loq_vocab()
    classes_dim = Dim(vocab.get_num_classes(), name="spm", kind=Dim.Types.Feature)
    classes_with_blank_dim = classes_dim + 1
    fwd_config = {
        "model_outputs": {
            "output": {"shape": (None,), "sparse_dim": classes_with_blank_dim},
            "scores": {"shape": ()},
        },
        "aux_loss_layers": [aux_ctc_layer],
        # Forced align is forward-only (no backward / optimizer state), so a large batch fits easily on
        # the 96GB GH200 and keeps each per-shard cell GPU-bound. Without this the forward inherits a
        # tiny default batch and leaves the GPU ~95% empty (per-cell model reload is only ~3% of wall).
        "batch_size": 200_000 * configs._batch_size_factor,
        "max_seqs": 200,
    }
    hf_dir = get_loquacious_hf_ogg(name="large")
    return batched_forward_per_arrow_shard_to_hdf(
        hf_data_dir=hf_dir.join_right("train"),
        make_dataset=functools.partial(_loq_make_shard_align_dataset, hf_data_dir=hf_dir, classes_dim=classes_dim),
        model=model,
        forward_step=_aed_ctc_model_forced_align_step,
        config=fwd_config,
    )


def _loq_coshard_train_parts(model, *, aux_ctc_layer: int, partition_epoch: int = 25) -> Dict[str, Any]:
    """Ingredients for :attr:`ChunkAlignDataset.train_coshard` (full ~25k h train): the audio
    DistributeFilesDataset's own lazy arrow-shard ``files`` callable + per-subepoch audio builder, the
    per-arrow-shard alignment dir, and the DFD ``partition_epoch`` (25 -> ~1000 h/subepoch, matching the
    loq base full train's ``train_epoch_split``).
    """
    from i6_experiments.users.zeyer.datasets.loquacious import _make_hf_dataset, get_loquacious_hf_ogg

    hf_dir = get_loquacious_hf_ogg(name="large")
    audio_dfd = _make_hf_dataset(
        hf_data_dir=hf_dir, split="train", vocab=_loq_vocab(), use_distrib_files=True
    ).main_dataset
    assert audio_dfd["class"] == "DistributeFilesDataset", audio_dfd["class"]
    return dict(
        audio_files=audio_dfd["files"],
        audio_sub_epoch_dataset=audio_dfd["get_sub_epoch_dataset"],
        align_dir=_loq_coshard_align_dir(model, aux_ctc_layer=aux_ctc_layer),
        partition_epoch=partition_epoch,
    )


def _enc_build_dict():
    return rf.build_dict(
        ChunkedConformerEncoderV2,
        input_layer=rf.build_dict(
            ConformerConvSubsample,
            out_dims=[32, 64, 64],
            filter_sizes=[(3, 3), (3, 3), (3, 3)],
            pool_sizes=[(1, 2)],
            strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6 (matches the alignment frame rate)
        ),
        encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
        num_layers=4,
        out_dim=256,
        num_heads=4,
        chunk_size=_CHUNK_SIZE,
        chunk_history_size=_CHUNK_SIZE * 8,
        chunk_lookahead_size=0,  # strictly causal by chunk for the first test
        version=3,
    )


def _train_streaming_variant(
    name: str,
    *,
    dec_build_dict: Dict[str, Any],
    train_def,
    recog_def=None,
    target_mode: str,
    speed_pert: bool = False,
    recog_extra: Optional[Dict[str, Any]] = None,
    corpus: str = "ls",
):
    """Train a streaming-decoder variant + greedy recog -> dev-other WER.

    Shared across variants: the LS-base model (for the forced alignment), the chunked
    encoder, the EOC-extended vocab, and the small smoke training config. The decoder
    (``dec_build_dict``), train/recog defs, target derivation (``target_mode``), and
    speed-pert are the per-variant knobs.
    """
    prefix = get_setup_prefix_for_module(__name__)

    # Corpus bundle: base model (for the forced alignment), vocab, per-split alignment HDFs, audio
    # provider, and recog task. The "ls" branch is byte-identical to the original LibriSpeech wiring;
    # "loq" swaps in the Loquacious base + native spm10k + HF audio (raw [B, T], no feature axis -> the
    # streaming model squeezes it) over a (gated) train subset.
    if corpus == "ls":
        vocab = get_vocab_by_str("spm10k")
        # Borrow the LS-base checkpoint only (suppress its own recog outputs); use it to forced-align.
        with disable_register_output():
            exp_base = _train_librispeech_base()
        base_model = exp_base.get_last_fixed_epoch()
        align_hdfs = _ls_align_hdfs(base_model, keys=["train", "dev-other"])
        # Frame-sync variants need the target 1:1 with encoder frames -> no speed-pert (would desync).
        oggzip_kw = {} if speed_pert else {"train_audio_preprocess": None}
        audio_provider = LibrispeechOggZip(audio=_raw_audio_opts.copy(), audio_dim=1, vocab=None, **oggzip_kw)
        # Full-node data pipeline: MPD parallelizes the OggZip decode (heavy), PostprocessingDataset's own
        # workers parallelize the map_seq target derivation. Outer train_v4 auto-MPD is off (post_config).
        chunk_data_kw: Dict[str, Any] = dict(
            train_main_key="train", dev_main_key="dev-other", train_mpd_num_workers=4, postproc_num_workers=2
        )
        recog_task = lambda: get_librispeech_task_raw_v2(vocab="spm10k")  # noqa: E731
        recog_dev_sets = ["dev-other"]
    elif corpus == "loq":
        assert not speed_pert, "loq smoke runs speed-pert off (forced align is on un-perturbed audio)"
        vocab = _loq_vocab()
        with disable_register_output():
            exp_base, base_aux_ctc_layer = _train_loquacious_base()
        base_model = exp_base.get_last_fixed_epoch()
        # HF audio is raw [B, T]: declare no feature axis and map the "audio" key. No inner MPD (the HF
        # dataset is the MetaDataset seq-order control); the map_seq postproc is still parallelized.
        chunk_data_kw = dict(
            train_main_key="train",
            dev_main_key="dev",
            audio_data_key="audio",
            audio_has_feature_dim=False,
            train_mpd_num_workers=None,
            postproc_num_workers=2,
        )
        if _LOQ_FULL_TRAIN:
            # Full ~25k h: co-shard the train alignment with the audio arrow shards (one align HDF per
            # shard, distributed together by a DistributeFilesDataset). Dev stays on the normal sharded
            # MetaDataset path. The audio provider is used only for dev (the DFD builds train per subepoch).
            align_hdfs = _loq_align_hdfs(
                base_model, num_shards=8, aux_ctc_layer=base_aux_ctc_layer, subset_seqs=None, keys=("dev",)
            )
            audio_provider = _LoqAudioProvider(train_subset_seqs=None)
            chunk_data_kw["train_coshard"] = _loq_coshard_train_parts(base_model, aux_ctc_layer=base_aux_ctc_layer)
        else:
            subset_seqs = _LOQ_SUBSET_TRAIN_SEQS
            align_hdfs = _loq_align_hdfs(
                base_model, num_shards=8, aux_ctc_layer=base_aux_ctc_layer, subset_seqs=subset_seqs
            )
            audio_provider = _LoqAudioProvider(train_subset_seqs=subset_seqs)
        recog_task = lambda: get_loquacious_task_raw_v2(vocab="spm10k")  # noqa: E731
        recog_dev_sets = ["dev"]
    else:
        raise ValueError(f"unknown corpus {corpus!r}")
    vocab_size = vocab.get_num_classes()  # spm10k -> 10240

    # Extended target vocab (spm pieces + 1 extra symbol = EOC / RNA-blank); train_v4 needs a vocab.
    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

    aug_vocab_file = ExtendVocabWithEocJob(ExtractSentencePieceVocabJob(vocab.model_file).out_vocab).out_vocab
    aug_vocab = {"class": "Vocabulary", "vocab_file": aug_vocab_file, "unknown_label": None}

    dataset = ChunkAlignDataset(
        oggzip=audio_provider,
        alignment_hdfs=align_hdfs,
        vocab_ext_dim_int=vocab_size + 1,
        blank_idx=vocab_size,
        chunk_size=_CHUNK_SIZE,
        target_mode=target_mode,
        aug_vocab=aug_vocab,
        **chunk_data_kw,
    )

    model_config = {
        "enc_build_dict": _enc_build_dict(),
        "dec_build_dict": dec_build_dict,
        "chunk_size": _CHUNK_SIZE,
        "aux_loss_layers": [4],
        "feature_batch_norm": True,
        "__serialization_version": 2,
    }
    config = dict_update_deep(
        configs.config_96gb_bf16_accgrad1,
        {
            **configs._get_cfg_lrlin_oclr_by_bs_nep_v4(10, base_lr=0.5),
            "batch_size": 10_000 * configs._batch_size_factor,
            "max_seqs": 100,
            "optimizer.weight_decay": 1e-2,
            # Multi-GPU: use the full JUPITER node (4x GH200). Single-GPU bills 4x for 1/4 use.
            "torch_distributed": {},  # DDP; every forward touches all params -> no find_unused_parameters
            "__cpu_rqmt": 72,  # per-proc; x4 = 288 = full node (overrides the CLAIX-tuned base value)
            "__mem_rqmt": 100,  # per-proc; x4 = 400 GiB
        },
    )
    # Consistent length filtering for all variants: cap by audio length, and drop the
    # inherited max_seq_length_default_target=75 (catastrophic for the long per-frame
    # targets of the frame-sync variants, arbitrary for the chunk ones).
    config = dict_update_deep(
        config,
        {"max_seq_length_default_target": None, "max_seq_length_default_input": 19.5 * _raw_sample_rate},
    )
    if corpus == "loq" and _LOQ_FULL_TRAIN:
        # Full ~25k h Loquacious-large: 4 full epochs. n_ep counts subepochs and partition_epoch=25
        # -> 25 subepochs = 1 pass, so n_ep = 4 * 25 = 100 (matches exp2026_05_26_base_fzj). The DFD
        # shards across the 4 DDP ranks, so the data is covered once per cycle (no division by num_gpus).
        # Full-node batch_size like the base full-train (the shared smoke default above is 10x smaller).
        config = dict_update_deep(
            config,
            {
                **configs._get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
                "batch_size": 100_000 * configs._batch_size_factor,
            },
        )

    exp = train_v4.train(
        prefix + "/" + name,
        train_dataset=dataset,
        config=config,
        # Opt out of train_v4's outer auto-MPD: the MPD now lives inside ChunkAlignDataset (around the
        # MetaDataset), and PostprocessingDataset runs its own map_seq workers. See ChunkAlignDataset.
        post_config={"log_grad_norm": True, "__multi_proc_dataset": False},
        model_def=ModelDefWithCfg(streaming_model_def, model_config),
        train_def=train_def,
        num_processes=4,  # 4 GPUs / JUPITER node -> torchrun DDP
        gpu_mem=96,
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )  # walltime is clipped globally to the FZJ 12h QOS cap in settings.py (check_engine_limits)

    # Greedy recog -> dev-other WER (sclite). The decoder emits over spm+extra, but the LS
    # reference is plain spm10k, so pin the model's extended output vocab via the search config.
    # (Skipped when recog_def is None -- e.g. ext_transducer, whose two-rate recog is WIP.)
    if recog_def is not None:
        task = recog_task()
        recog_cfg = {
            "target_dim_ext_int": vocab_size + 1,
            "aug_vocab": aug_vocab,
            "batch_size": 10_000 * configs._batch_size_factor,
            "max_seqs": 100,
        }
        if recog_extra:
            recog_cfg.update(recog_extra)
        # Multi-GPU batched recog over the trained epochs on the full 4-GPU node: one
        # BatchedReturnnForwardJob instead of a single-GPU ReturnnForwardJobV2 (the latter is
        # forbidden on FZJ by the flat-per-node billing). Registers recog_results_best /
        # recog_results_all_epochs under the recog prefix.
        recog_training_exp_batched(
            prefix + "/" + name + "/recog",
            task=task,
            model=exp,
            recog_def=recog_def,
            search_config=recog_cfg,
            dev_sets=recog_dev_sets,
        )
    return exp


def _train_chunkwise_smoke():
    """Chunk-synchronous decoder, first end-to-end pipeline test on FZJ."""
    return _train_streaming_variant(
        "chunkwise-smoke",
        dec_build_dict=rf.build_dict(ChunkwiseDecoder, model_dim=256, ff_dim=512, num_layers=4, num_heads=4),
        train_def=chunkwise_training,
        recog_def=chunkwise_model_recog,
        target_mode="chunk_eoc",
        recog_extra={"max_labels_per_chunk": 20},
    )


def _train_framewise_smoke():
    """Frame-synchronous RNA fast-only decoder (speed-pert off)."""
    return _train_streaming_variant(
        "framewise-smoke",
        dec_build_dict=rf.build_dict(FramewiseDecoder, model_dim=256, ff_dim=512, num_layers=4, num_heads=4),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame",
    )


def _train_ext_transducer_smoke():
    """Extended-transducer slow+fast decoder (slow label-rate stack injected into the fast frame stack)."""
    return _train_streaming_variant(
        "ext-transducer-smoke",
        dec_build_dict=rf.build_dict(ExtTransducerDecoder, model_dim=256, ff_dim=512, num_layers=4, num_heads=4),
        train_def=ext_transducer_training,
        recog_def=ext_transducer_model_recog,
        target_mode="rna_frame",
    )


def _train_two_tower_smoke():
    """Two-tower fast-slow decoder (text + speech stacks coupled by cross-attention)."""
    return _train_streaming_variant(
        "two-tower-smoke",
        dec_build_dict=rf.build_dict(TwoTowerDecoder, model_dim=256, ff_dim=512, num_layers=4, num_heads=4),
        train_def=two_tower_training,
        recog_def=two_tower_model_recog,
        target_mode="rna_frame",
    )


# Loquacious counterparts of the four smoke variants: identical decoders / train+recog defs / target
# modes, but corpus="loq" routes them through the Loquacious base align + HF audio (subset-gated).


def _train_chunkwise_loq_smoke():
    """Chunk-synchronous decoder on Loquacious (subset smoke by default)."""
    return _train_streaming_variant(
        "chunkwise-loq-smoke",
        dec_build_dict=rf.build_dict(ChunkwiseDecoder, model_dim=256, ff_dim=512, num_layers=4, num_heads=4),
        train_def=chunkwise_training,
        recog_def=chunkwise_model_recog,
        target_mode="chunk_eoc",
        recog_extra={"max_labels_per_chunk": 20},
        corpus="loq",
    )


def _train_framewise_loq_smoke():
    """Frame-synchronous RNA fast-only decoder on Loquacious (subset smoke by default)."""
    return _train_streaming_variant(
        "framewise-loq-smoke",
        dec_build_dict=rf.build_dict(FramewiseDecoder, model_dim=256, ff_dim=512, num_layers=4, num_heads=4),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame",
        corpus="loq",
    )


def _train_ext_transducer_loq_smoke():
    """Extended-transducer slow+fast decoder on Loquacious (subset smoke by default)."""
    return _train_streaming_variant(
        "ext-transducer-loq-smoke",
        dec_build_dict=rf.build_dict(ExtTransducerDecoder, model_dim=256, ff_dim=512, num_layers=4, num_heads=4),
        train_def=ext_transducer_training,
        recog_def=ext_transducer_model_recog,
        target_mode="rna_frame",
        corpus="loq",
    )


def _train_two_tower_loq_smoke():
    """Two-tower fast-slow decoder on Loquacious (subset smoke by default)."""
    return _train_streaming_variant(
        "two-tower-loq-smoke",
        dec_build_dict=rf.build_dict(TwoTowerDecoder, model_dim=256, ff_dim=512, num_layers=4, num_heads=4),
        train_def=two_tower_training,
        recog_def=two_tower_model_recog,
        target_mode="rna_frame",
        corpus="loq",
    )
