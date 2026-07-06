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
encoder, the EOC-extended vocab, the ChunkAlignDataset, the decoder + train def -- is shared.

First (decisive) run: the **standard-AED control** on single GPU. On FZJ/4-GPU it scored CTC-only 11.11
vs the base 9.41; single-GPU restores the base regime (batch 8M, ~660k optimizer steps, warmup = 3%),
so it should reproduce ~9.4 if the reimplementation is faithful. See the project notes.
"""

from __future__ import annotations

from typing import Any, Dict

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
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.chunkwise import ChunkwiseDecoder
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.dataset import ChunkAlignDataset, ExtendVocabWithEocJob
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.standard_aed import standard_aed_training

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
    _train_standard_aed_rz()


def _loq_chunk_align_dataset(base_model, *, base_aux_ctc_layer: int, target_mode: str):
    """The full ~25k h Loquacious ChunkAlignDataset, byte-identical to the FZJ full-train wiring.

    The train alignment is co-sharded with the audio arrow shards (``train_coshard``, -> ``eD69``) and dev
    uses the normal sharded align (``kXa``); both reference ``base_model`` (= ``WQbKY``). Returns the
    dataset plus the EOC-extended aug-vocab (needed again for recog).
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


def _train_standard_aed_rz():
    """Standard (full-attention) AED control on a single 96 GB RZ GPU.

    Same dyn-rope-ctembed chunked encoder + ChunkAlignDataset + reimplemented AED decoder as the FZJ
    ``standard-aed-loq`` run, but single-GPU (restores the base 8M-batch / ~660k-step regime) and with
    ``dec_aux_loss_layers=[3]`` + ``label_smoothing=0.1`` to match the base recipe exactly. Scored
    CTC-only on the encoder aux head = the base 9.41 metric.
    """
    prefix = get_setup_prefix_for_module(__name__)
    name = "standard-aed-loq-1gpu"

    with disable_register_output():
        exp_base, base_aux_ctc_layer = _train_loquacious_base()
    base_model = exp_base.get_last_fixed_epoch()  # WQbKY, checkpoint copied from FZJ

    dataset, aug_vocab, vocab_size = _loq_chunk_align_dataset(
        base_model, base_aux_ctc_layer=base_aux_ctc_layer, target_mode="labels"
    )

    model_config = {
        "enc_build_dict": _enc_build_dict(num_layers=16, out_dim=1024, num_heads=8, dynamic=True),
        "dec_build_dict": rf.build_dict(ChunkwiseDecoder, model_dim=1024, num_layers=6, num_heads=8, version=2),
        "chunk_size": _CHUNK_SIZE,
        "aux_loss_layers": [4, 10, 16],
        "feature_batch_norm": True,
        "__serialization_version": 2,
    }

    # Single-GPU regime = the base 9.41 config (config_96gb_bf16_accgrad1, batch 50k=8M, max_seqs 200,
    # OCLR nep100 base_lr0.5, warmup 20k, wd 1e-2), plus the base's dec-aux + label smoothing.
    config = dict_update_deep(
        configs.config_96gb_bf16_accgrad1,
        {
            **configs._get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 50_000 * configs._batch_size_factor,
            "max_seqs": 200,
            "optimizer.weight_decay": 1e-2,
            "label_smoothing": 0.1,  # decoder-CE label smoothing, as in the AED baseline
            "dec_aux_loss_layers": [3],  # decoder-layer-3 aux CE, as in the AED baseline
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
    )

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
        train_def=standard_aed_training,
        gpu_mem=96,
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # CTC-only recog on the encoder aux CTC head (single-GPU) = the base 9.41 metric.
    task = get_loquacious_task_raw_v2(vocab="spm10k")
    recog_training_exp(
        prefix + "/" + name + "/recog-ctc",
        task=task,
        model=exp,
        recog_def=model_recog_ctc,
        search_config={
            "target_dim_ext_int": vocab_size + 1,
            "aug_vocab": aug_vocab,
            "batch_size": 10_000 * configs._batch_size_factor,
            "max_seqs": 100,
            "aux_loss_layers": model_config["aux_loss_layers"],
        },
    )
    return exp
