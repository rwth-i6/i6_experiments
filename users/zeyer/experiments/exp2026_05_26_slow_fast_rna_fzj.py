"""
Slow-fast-RNA streaming chunked-decoder experiments.
"""

from __future__ import annotations

from typing import Dict

from sisyphus import tk
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerConvSubsample

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module, disable_register_output
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg
from i6_experiments.users.zeyer import train_v4
from i6_experiments.users.zeyer.datasets.librispeech import LibrispeechOggZip, _raw_audio_opts, get_vocab_by_str
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
from i6_experiments.users.zeyer.experiments.exp2025_10_21_chunked_ctc import _aed_ctc_forced_align
from i6_experiments.users.zeyer.experiments.exp2026_05_26_base_fzj import _train_librispeech_base
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
    ChunkedConformerEncoderV2,
    ChunkedConformerEncoderLayerV2,
    ChunkedRotaryPosSelfAttentionV2,
)
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.base import streaming_model_def
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.chunkwise import ChunkwiseDecoder, chunkwise_training
from i6_experiments.users.zeyer.nn_rf.decoder.streaming.dataset import ChunkAlignDataset, ExtendVocabWithEocJob

# Prefix for alias/ and output/ paths. Does not enter any Job hash (only naming);
# the helper walks the module hierarchy for this attribute.
__setup_root_prefix__ = "exp2026_05_26_slow_fast_rna_fzj"


def py():
    _train_chunkwise_smoke()


def _train_chunkwise_smoke():
    """Small chunk-synchronous model, first end-to-end pipeline test on FZJ."""
    prefix = get_setup_prefix_for_module(__name__)
    vocab = get_vocab_by_str("spm10k")
    vocab_size = vocab.get_num_classes()  # spm10k -> 10240
    chunk_size = 10  # encoder frames (60ms each) -> 600ms chunks

    # Borrow the LS-base model checkpoint only. Suppress its output registrations so the
    # base model's recog (timesync + scale tuning, ~16 forward jobs) does not become a
    # target in this streaming setup -- we just need the trained encoder/CTC for alignment.
    with disable_register_output():
        exp = _train_librispeech_base()
    model = exp.get_last_fixed_epoch()
    align_hdfs = _ls_align_hdfs(model, keys=["train", "dev-other"])

    # aug_targets vocab (spm pieces + EOC): train_v4 requires a vocab on the target sparse dim.
    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

    aug_vocab_file = ExtendVocabWithEocJob(ExtractSentencePieceVocabJob(vocab.model_file).out_vocab).out_vocab
    aug_vocab = {"class": "Vocabulary", "vocab_file": aug_vocab_file, "unknown_label": None}

    audio_oggzip = LibrispeechOggZip(audio=_raw_audio_opts.copy(), audio_dim=1, vocab=None)  # audio only
    dataset = ChunkAlignDataset(
        oggzip=audio_oggzip,
        alignment_hdfs=align_hdfs,
        vocab_ext_dim_int=vocab_size + 1,  # + EOC
        blank_idx=vocab_size,
        chunk_size=chunk_size,
        train_main_key="train",
        dev_main_key="dev-other",
        aug_vocab=aug_vocab,
    )

    enc_build_dict = rf.build_dict(
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
        chunk_size=chunk_size,
        chunk_history_size=chunk_size * 8,
        chunk_lookahead_size=0,  # strictly causal by chunk for the first test
        version=3,
    )
    dec_build_dict = rf.build_dict(
        ChunkwiseDecoder, model_dim=256, ff_dim=512, num_layers=4, num_heads=4
    )  # encoder_dim / vocab_dim / chunk_size / eoc_idx injected by StreamingModel

    model_config = {
        "enc_build_dict": enc_build_dict,
        "dec_build_dict": dec_build_dict,
        "chunk_size": chunk_size,
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
            "__multi_proc_dataset": False,  # keep the pipeline simple for the smoke test
        },
    )

    train_v4.train(
        prefix + "/chunkwise-smoke",
        train_dataset=dataset,
        config=config,
        post_config={"log_grad_norm": True},
        model_def=ModelDefWithCfg(streaming_model_def, model_config),
        train_def=chunkwise_training,
    )  # walltime is clipped globally to the FZJ 12h QOS cap in settings.py (check_engine_limits)


def _ls_align_hdfs(model, *, keys, aux_ctc_layer: int = 16) -> Dict[str, tk.Path]:
    """Per-frame CTC forced-align HDF per LS split (dedups with the base recipe's alignments)."""
    vocab = get_vocab_by_str("spm10k")
    out = {}
    for key in keys:
        ds = LibrispeechOggZip(audio=_raw_audio_opts.copy(), audio_dim=1, vocab=vocab, main_key=key)
        out[key] = _aed_ctc_forced_align(model, ds, aux_ctc_layer=aux_ctc_layer)
    return out
