"""
Slow-fast-RNA streaming experiments -- RZ single-GPU continuation.

Sibling of :mod:`exp2026_05_26_slow_fast_rna_fzj` for the RWTH RZ cluster (single 96 GB GPU),
started while FZJ JUPITER is in maintenance. It **reuses** the FZJ module's base-model + forced-align
builders unchanged, so the Job hashes match the artifacts rsync'd from FZJ
(``ReturnnTrainingJob.WQbKY49ztiXt`` = the AED+CTC align base, the co-shard train alignment
``BatchedReturnnForwardDynamicJob.eD69Sx31Hy1O``, and the dev alignment
``BatchedReturnnForwardJob.kXaTtWYwj6MV``) -- sis recognises them as finished, no recompute.

Only the training + recog assembly differs from FZJ: single 96 GB GPU (no ``torch_distributed`` /
``num_processes`` / per-node rqmt / ``file_cache_opts``), and single-GPU recog (``recog_training_exp``,
not the FZJ multi-GPU ``recog_training_exp_batched``). Everything else -- the dyn-rope-ctembed chunked
encoder, the EOC-extended vocab, the ChunkAlignDataset, the decoders + train defs -- is shared.

Variants (all single-GPU, same dyn-rope-ctembed encoder + regime, differing only in the decoder):
- ``standard-aed`` control (CTC-only scored; reproduces base 9.41);
- ``chunkwise`` (chunk-sync AED); ``framewise`` (fast-only RNA); ``ext-transducer`` (slow+fast state
  injection); ``two-tower`` (text+speech cross-att). All use the **dynamic** encoder (the frame-rate
  variants re-align their per-frame target onto the encoder length via ``rna_targets_on_enc_spatial``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import returnn.frontend as rf

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module, disable_register_output
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg
from i6_experiments.users.zeyer import train_v4
from i6_experiments.users.zeyer.recog import recog_training_exp
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import _raw_sample_rate
from i6_experiments.users.zeyer.datasets.loquacious import get_loquacious_task_raw_v2
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.base import streaming_model_def, model_recog_ctc
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.dataset import ChunkAlignDataset, ExtendVocabWithEocJob
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.standard_aed import standard_aed_training
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
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.rnnt import (
    RnntDecoder,
    rnnt_training,
    model_recog as rnnt_model_recog,
)
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.rnnt_fullsum import rnnt_fullsum_training

# Reused verbatim from the FZJ module -> identical Job hashes -> the rsync'd base + alignment are found.
from i6_experiments.users.zeyer.experiments.exp2026_05_26_base_fzj import _train_loquacious_base
from i6_experiments.users.zeyer.experiments.exp2026_05_26_slow_fast_rna_fzj import (
    _CHUNK_SIZE,
    _enc_build_dict,
    _loq_vocab,
    _loq_align_hdfs,
    _loq_coshard_train_parts,
    _LoqAudioProvider,
)

# Prefix for alias/ and output/ paths only (does not enter any Job hash).
__setup_root_prefix__ = "exp2026_05_26_slow_fast_rna_rz"


def py():
    # The standard-AED control (CTC-only = base 9.41 metric) + the four streaming decoder variants.
    _train_standard_aed_rz()
    _train_chunkwise_rz()
    _train_framewise_rz()
    _train_ext_transducer_rz()
    _train_two_tower_rz()
    _train_rnnt_mono_framewise_rz()
    _train_rnnt_mono_fullsum_rz()
    _train_rnnt_mono_framewise_small_rz()
    _train_rnnt_mono_fullsum_small_rz()
    _train_framewise_enc8dec12_rz()
    _train_framewise_enc4dec24_rz()
    _train_framewise_delay_rz()
    _train_framewise_wordchunk_rz()
    _train_framewise_wordchunk_end_rz()
    _train_framewise_delay0p3_rz()
    _train_framewise_wordchunk_end_delay0p3_rz()


def _loq_chunk_align_dataset(base_model, *, base_aux_ctc_layer: int, target_mode: str):
    """The full ~25k h Loquacious ChunkAlignDataset, byte-identical to the FZJ full-train wiring.

    The train alignment is co-sharded with the audio arrow shards (``train_coshard``, -> ``eD69``) and dev
    uses the normal sharded align (``kXa``); both reference ``base_model`` (= ``WQbKY``). ``target_mode``
    selects the per-frame/per-chunk target derivation; the alignment refs are the same regardless. Returns
    the dataset plus the EOC-extended aug-vocab (needed again for recog).
    """
    vocab = _loq_vocab()
    vocab_size = vocab.get_num_classes()  # spm10k -> 10240

    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

    aug_vocab_file = ExtendVocabWithEocJob(ExtractSentencePieceVocabJob(vocab.model_file).out_vocab).out_vocab
    aug_vocab = {"class": "Vocabulary", "vocab_file": aug_vocab_file, "unknown_label": None}

    align_hdfs = _loq_align_hdfs(
        base_model, num_shards=8, aux_ctc_layer=base_aux_ctc_layer, subset_seqs=None, keys=("dev",)
    )
    chunk_data_kw: Dict[str, Any] = dict(
        train_main_key="train",
        dev_main_key="dev",
        audio_data_key="audio",
        audio_has_feature_dim=False,
        train_mpd_num_workers=None,
        postproc_num_workers=2,
        train_coshard=_loq_coshard_train_parts(base_model, aux_ctc_layer=base_aux_ctc_layer),
    )
    dataset = ChunkAlignDataset(
        oggzip=_LoqAudioProvider(train_subset_seqs=None),
        alignment_hdfs=align_hdfs,
        vocab_ext_dim_int=vocab_size + 1,
        blank_idx=vocab_size,
        chunk_size=_CHUNK_SIZE,
        target_mode=target_mode,
        aug_vocab=aug_vocab,
        **chunk_data_kw,
    )
    return dataset, aug_vocab, vocab_size


def _train_variant_rz(
    name: str,
    *,
    dec_build_dict: Dict[str, Any],
    train_def,
    target_mode: str,
    recog_def=None,
    recog_extra: Optional[Dict[str, Any]] = None,
    dec_aux_loss_layers: Sequence[int] = (),
    enc_num_layers: int = 16,
    aux_loss_layers: Sequence[int] = (4, 10, 16),
    nep: int = 100,
    extra_config: Optional[Dict[str, Any]] = None,
):
    """Train one streaming-decoder variant on a single 96 GB RZ GPU + recog.

    Same dyn-rope-ctembed chunked encoder + ChunkAlignDataset + regime for every variant (batch 8M,
    OCLR nep100 base_lr0.5, aux [4,10,16], dec_aux [3], label_smoothing 0.1 -- the base 9.41 regime),
    single-GPU. The decoder (``dec_build_dict``), its train def, ``target_mode``, and recog are the
    per-variant knobs. Always runs the CTC-only recog (encoder aux head = base 9.41 metric); also runs
    the decoder recog when ``recog_def`` is given.
    """
    prefix = get_setup_prefix_for_module(__name__)

    with disable_register_output():
        exp_base, base_aux_ctc_layer = _train_loquacious_base()
    base_model = exp_base.get_last_fixed_epoch()  # WQbKY, checkpoint copied from FZJ

    dataset, aug_vocab, vocab_size = _loq_chunk_align_dataset(
        base_model, base_aux_ctc_layer=base_aux_ctc_layer, target_mode=target_mode
    )

    model_config = {
        "enc_build_dict": _enc_build_dict(num_layers=enc_num_layers, out_dim=1024, num_heads=8, dynamic=True),
        "dec_build_dict": dec_build_dict,
        "chunk_size": _CHUNK_SIZE,
        "aux_loss_layers": list(aux_loss_layers),
        "feature_batch_norm": True,
        "__serialization_version": 2,
    }

    # Single-GPU regime = the base 9.41 config (config_96gb_bf16_accgrad1, batch 50k=8M, max_seqs 200,
    # OCLR nep100 base_lr0.5, wd 1e-2), plus the base's dec-aux + label smoothing.
    config = dict_update_deep(
        configs.config_96gb_bf16_accgrad1,
        {
            **configs._get_cfg_lrlin_oclr_by_bs_nep_v4(nep, base_lr=0.5),
            "batch_size": 50_000 * configs._batch_size_factor,
            "max_seqs": 200,
            "optimizer.weight_decay": 1e-2,
            "label_smoothing": 0.1,  # decoder-CE label smoothing, as in the AED baseline
            # dec_aux only for the std-AED control (to match base);
            # the slow+fast decoders (ext-transducer / two-tower) have no top-level final_ln/logits to tap.
            **({"dec_aux_loss_layers": list(dec_aux_loss_layers)} if dec_aux_loss_layers else {}),
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
    )
    if extra_config:
        config = dict_update_deep(config, extra_config)

    exp = train_v4.train(
        prefix + "/" + name,
        train_dataset=dataset,
        config=config,
        # Single-GPU: no torch_distributed / num_processes / file_cache_opts (all FZJ-node-specific).
        post_config={
            "log_grad_norm": True,
            "__multi_proc_dataset": False,
            "stop_for_resubmission_when_low_time_left": True,
        },
        model_def=ModelDefWithCfg(streaming_model_def, model_config),
        train_def=train_def,
        gpu_mem=96,
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    task = get_loquacious_task_raw_v2(vocab="spm10k")
    recog_cfg: Dict[str, Any] = {
        "target_dim_ext_int": vocab_size + 1,
        "aug_vocab": aug_vocab,
        "batch_size": 10_000 * configs._batch_size_factor,
        "max_seqs": 100,
        "aux_loss_layers": model_config["aux_loss_layers"],
    }
    if recog_extra:
        recog_cfg = {**recog_cfg, **recog_extra}

    # Decoder recog (skipped for the CTC-only control).
    if recog_def is not None:
        recog_training_exp(
            prefix + "/" + name + "/recog", task=task, model=exp, recog_def=recog_def, search_config=recog_cfg
        )
    # CTC-only recog on the encoder aux CTC head = the base 9.41 metric (all variants).
    recog_training_exp(
        prefix + "/" + name + "/recog-ctc", task=task, model=exp, recog_def=model_recog_ctc, search_config=recog_cfg
    )
    return exp


def _train_standard_aed_rz():
    """Standard (full-attention) AED control -- reproduces base 9.41 (CTC-only scored)."""
    return _train_variant_rz(
        "standard-aed-loq-1gpu",
        dec_build_dict=rf.build_dict(ChunkwiseDecoder, model_dim=1024, num_layers=6, num_heads=8, version=2),
        train_def=standard_aed_training,
        target_mode="labels",
        recog_def=None,  # CTC-only recog (= base 9.41 metric); AED decoder search deferred
        dec_aux_loss_layers=[3],  # match the base AED (dec-layer-3 aux CE); only the std-AED decoder taps it
    )


def _train_chunkwise_rz():
    """Chunk-synchronous decoder (EOC-augmented per-chunk targets, chunk-masked cross-att)."""
    return _train_variant_rz(
        "chunkwise-1gpu",
        dec_build_dict=rf.build_dict(ChunkwiseDecoder, model_dim=1024, num_layers=6, num_heads=8, version=2),
        train_def=chunkwise_training,
        recog_def=chunkwise_model_recog,
        target_mode="chunk_eoc",
        recog_extra={"max_labels_per_chunk": 20},
    )


def _train_framewise_rz():
    """Frame-synchronous RNA fast-only decoder (one label/blank per encoder frame, no cross-att)."""
    return _train_variant_rz(
        "framewise-1gpu",
        dec_build_dict=rf.build_dict(FramewiseDecoder, model_dim=1024, num_layers=6, num_heads=8, version=2),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame",
    )


def _train_ext_transducer_rz():
    """Extended-transducer slow+fast decoder (slow label-rate stack injected into the fast frame stack)."""
    return _train_variant_rz(
        "ext-transducer-1gpu",
        dec_build_dict=rf.build_dict(
            ExtTransducerDecoder, model_dim=1024, num_layers=3, num_heads=8, version=2
        ),  # 3 slow + 3 fast = 6 dec layers
        train_def=ext_transducer_training,
        recog_def=ext_transducer_model_recog,
        target_mode="rna_frame",
    )


def _train_two_tower_rz():
    """Two-tower fast-slow decoder (text + speech stacks coupled by cross-attention)."""
    return _train_variant_rz(
        "two-tower-1gpu",
        dec_build_dict=rf.build_dict(
            TwoTowerDecoder, model_dim=1024, num_layers=3, num_heads=8, version=2
        ),  # 3 text + 3 speech = 6 dec layers
        train_def=two_tower_training,
        recog_def=two_tower_model_recog,
        target_mode="rna_frame",
    )


def _train_rnnt_mono_framewise_rz():
    """Standard monotonic RNN-T (v3, label-only pred net), framewise-CE on our RNA alignment.

    Corrects the earlier rnnt-1gpu (v2 pred net was conditioned on the emission frame -> alignment-dependent);
    directly comparable to the full-sum variant (same model, only the objective differs).
    """
    return _train_variant_rz(
        "rnnt-mono-framewise-1gpu",
        dec_build_dict=rf.build_dict(RnntDecoder, model_dim=1024, num_layers=6, num_heads=8, version=4),
        train_def=rnnt_training,
        recog_def=rnnt_model_recog,
        target_mode="rna_frame",
    )


def _train_rnnt_mono_fullsum_rz():
    """Standard monotonic RNN-T (v3), marginalized full-sum loss (i6_native_ops.monotonic_rnnt).

    Same model as the framewise variant;
    the target is the plain transcription (target_mode="labels"),
    and the loss marginalizes over all monotonic alignments.
    Same batch/regime as the framewise variant,
    so the two differ ONLY in the loss.
    The packed joiner grid Sum_b T_b*(S_b+1) x V is the extra memory,
    on top of the fixed model/optimizer;
    reduce the batch only if it actually OOMs.
    """
    return _train_variant_rz(
        "rnnt-mono-fullsum-1gpu",
        dec_build_dict=rf.build_dict(RnntDecoder, model_dim=1024, num_layers=6, num_heads=8, version=4),
        train_def=rnnt_fullsum_training,
        recog_def=rnnt_model_recog,
        target_mode="labels",
        # The full-size model OOMs at the standard 8M batch:
        # the packed joiner grid comes on top of the fixed model/optimizer memory.
        # 4.8M fits with margin; the -small pair keeps the standard batch,
        # so the controlled framewise-vs-fullsum comparison lives there.
        extra_config={"batch_size": 30_000 * configs._batch_size_factor},
    )


def _train_rnnt_mono_small(name: str, *, train_def, target_mode: str, extra_config=None):
    """Scaled-down monotonic RNN-T (6L enc, 3L dec, nep20) -- fast same-day framewise-vs-full-sum comparison."""
    return _train_variant_rz(
        name,
        dec_build_dict=rf.build_dict(RnntDecoder, model_dim=1024, num_layers=3, num_heads=8, version=4),
        train_def=train_def,
        recog_def=rnnt_model_recog,
        target_mode=target_mode,
        enc_num_layers=6,
        aux_loss_layers=(2, 4, 6),
        nep=20,
        extra_config=extra_config,
    )


def _train_rnnt_mono_framewise_small_rz():
    return _train_rnnt_mono_small("rnnt-mono-framewise-small-1gpu", train_def=rnnt_training, target_mode="rna_frame")


def _train_rnnt_mono_fullsum_small_rz():
    return _train_rnnt_mono_small(
        "rnnt-mono-fullsum-small-1gpu",
        train_def=rnnt_fullsum_training,
        target_mode="labels",
    )


def _train_framewise_enc8dec12_rz():
    """enc:dec ratio control -- framewise, smaller encoder (8L) + larger decoder (12L).

    Not param-matched (404M vs framewise 520M) -- kept for its training progress.
    """
    return _train_variant_rz(
        "framewise-enc8-dec12-1gpu",
        dec_build_dict=rf.build_dict(FramewiseDecoder, model_dim=1024, num_layers=12, num_heads=8, version=2),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame",
        enc_num_layers=8,
        aux_loss_layers=[2, 5, 8],  # aux CTC within the 8-layer encoder (top=8 = the CTC recog head)
    )


def _train_framewise_delay_rz():
    """DSM ablation: framewise + a fixed ~2.5 s audio->text delay (42 enc frames @ ~16.67 Hz); sole change vs framewise."""
    return _train_variant_rz(
        "framewise-delay2p5-1gpu",
        dec_build_dict=rf.build_dict(
            FramewiseDecoder, model_dim=1024, num_layers=6, num_heads=8, delay_frames=42, version=2
        ),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame",
    )


def _train_framewise_enc4dec24_rz():
    """enc:dec ratio (decoder-heavy) -- framewise, 4L enc + 24L dec; follows enc8-dec12 (halve enc, double dec), not param-matched."""
    return _train_variant_rz(
        "framewise-enc4-dec24-1gpu",
        dec_build_dict=rf.build_dict(FramewiseDecoder, model_dim=1024, num_layers=24, num_heads=8, version=2),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame",
        enc_num_layers=4,
        aux_loss_layers=[1, 2, 4],  # aux CTC within the 4-layer encoder (top=4 = the CTC recog head)
    )


def _train_framewise_wordchunk_rz():
    """DSM ablation: framewise + word-chunked target layout (sub-words packed at word onset); sole change vs framewise."""
    return _train_variant_rz(
        "framewise-wordchunk-1gpu",
        dec_build_dict=rf.build_dict(FramewiseDecoder, model_dim=1024, num_layers=6, num_heads=8, version=2),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame_wordchunk",
    )


def _train_framewise_wordchunk_end_rz():
    """DSM ablation: framewise + END-anchored word-chunk target.

    Each word's sub-words are emitted at the word OFFSET,
    packed ending at the last sub-word's frame,
    so the whole word is only emitted once its acoustics are in
    -- a delay to the word boundary.
    Contrast ``framewise-wordchunk`` (onset-anchored),
    which emits later sub-words BEFORE their acoustics (anticipation).
    Sole change vs that variant is the target layout.
    """
    return _train_variant_rz(
        "framewise-wordchunk-end-1gpu",
        dec_build_dict=rf.build_dict(FramewiseDecoder, model_dim=1024, num_layers=6, num_heads=8, version=2),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame_wordchunk_end",
    )


def _train_framewise_delay0p3_rz():
    """DSM latency point: framewise + a fixed ~0.3 s audio->text delay (5 enc frames @ ~16.67 Hz); vs delay2p5 (42 frames)."""
    return _train_variant_rz(
        "framewise-delay0p3-1gpu",
        dec_build_dict=rf.build_dict(
            FramewiseDecoder, model_dim=1024, num_layers=6, num_heads=8, delay_frames=5, version=2
        ),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame",
    )


def _train_framewise_wordchunk_end_delay0p3_rz():
    """DSM combo: END-anchored word-chunk target + a fixed ~0.3 s delay (5 frames) on top of the word-boundary delay."""
    return _train_variant_rz(
        "framewise-wordchunk-end-delay0p3-1gpu",
        dec_build_dict=rf.build_dict(
            FramewiseDecoder, model_dim=1024, num_layers=6, num_heads=8, delay_frames=5, version=2
        ),
        train_def=framewise_training,
        recog_def=framewise_model_recog,
        target_mode="rna_frame_wordchunk_end",
    )
