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
from i6_experiments.users.zeyer.experiments.exp2026_05_26_base_fzj import _train_librispeech_base
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


def py():
    _validate_batched_forced_align()
    _train_chunkwise_smoke()
    _train_framewise_smoke()
    _train_ext_transducer_smoke()
    _train_two_tower_smoke()


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
):
    """Train a streaming-decoder variant + greedy recog -> dev-other WER.

    Shared across variants: the LS-base model (for the forced alignment), the chunked
    encoder, the EOC-extended vocab, and the small smoke training config. The decoder
    (``dec_build_dict``), train/recog defs, target derivation (``target_mode``), and
    speed-pert are the per-variant knobs.
    """
    prefix = get_setup_prefix_for_module(__name__)
    vocab = get_vocab_by_str("spm10k")
    vocab_size = vocab.get_num_classes()  # spm10k -> 10240

    # Borrow the LS-base checkpoint only (suppress its own recog outputs); use it to forced-align.
    with disable_register_output():
        exp_base = _train_librispeech_base()
    base_model = exp_base.get_last_fixed_epoch()
    align_hdfs = _ls_align_hdfs(base_model, keys=["train", "dev-other"])

    # Extended target vocab (spm pieces + 1 extra symbol = EOC / RNA-blank); train_v4 needs a vocab.
    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

    aug_vocab_file = ExtendVocabWithEocJob(ExtractSentencePieceVocabJob(vocab.model_file).out_vocab).out_vocab
    aug_vocab = {"class": "Vocabulary", "vocab_file": aug_vocab_file, "unknown_label": None}

    # Frame-sync variants need the target 1:1 with encoder frames -> no speed-pert (would desync).
    oggzip_kw = {} if speed_pert else {"train_audio_preprocess": None}
    audio_oggzip = LibrispeechOggZip(audio=_raw_audio_opts.copy(), audio_dim=1, vocab=None, **oggzip_kw)
    dataset = ChunkAlignDataset(
        oggzip=audio_oggzip,
        alignment_hdfs=align_hdfs,
        vocab_ext_dim_int=vocab_size + 1,
        blank_idx=vocab_size,
        chunk_size=_CHUNK_SIZE,
        target_mode=target_mode,
        train_main_key="train",
        dev_main_key="dev-other",
        aug_vocab=aug_vocab,
        # Full-node data pipeline: MPD parallelizes the OggZip decode (heavy), PostprocessingDataset's own
        # workers parallelize the map_seq target derivation. Outer train_v4 auto-MPD is off (post_config).
        train_mpd_num_workers=4,
        postproc_num_workers=2,
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
        task = get_librispeech_task_raw_v2(vocab="spm10k")
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
            dev_sets=["dev-other"],
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
