"""
FZJ baselines port: Loquacious AED+CTC + chunked-CTC, plus a LibriSpeech AED+CTC sibling.

Three trainings:
- ``base`` (Loquacious, offline) -- copied from
  ``exp2025_10_21_chunked_ctc.py:py():train('base', {})``;
  arch + train config in :data:`_base_config` (copied verbatim from that recipe).
- ``chunked-L80-C5-R4-v2.3-dyn-rope-ctembed`` (Loquacious, chunked) --
  copied from the corresponding ``train(name, {...})`` block of the same recipe.
- ``base-librispeech`` (LibriSpeech) -- copied from
  ``exp2025_08_05_aed_large.py:py()``, the
  ``EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k``
  entry (around line 2859). Architecturally identical to the Loquacious base
  (bhv24, featBN, aux4_10_16, auxDec3, spm10k); LS-specific speed-pert enabled.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, Tuple
from functools import cache

from sisyphus import tk

from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
    ctc_recog_recomb_labelwise_prior_auto_scale,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    model_recog as ctc_model_recog,
)

from i6_experiments.users.zeyer.datasets.loquacious import (
    get_loquacious_task_raw_v2,
    get_loquacious_train_subset_dataset_v2,
)
from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)

from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
    ChunkedConformerEncoderV2,
    ChunkedConformerEncoderLayerV2,
    ChunkedRotaryPosSelfAttentionV2,
)

# Prefix for alias/ and output/ paths. Doesn't enter any Job hash; pick
# anything readable (the recipe name matches the file).
__setup_root_prefix__ = "exp2026_05_26_base_fzj"


def py():
    # ---- Loquacious AED+CTC, offline ----
    # From exp2025_10_21_chunked_ctc.py:py() -- the ``train("base", {})`` call.
    # Arch / train config defined in :data:`_base_config` below
    # (16-layer Conformer 1024d + 6-layer Transformer decoder 1024d, spm10k,
    # bhv24, featBN, aux_loss_layers=[4,10,16], dec_aux_loss_layers=[3]).
    # RZ WER: 7.32 / 6.12 (ctc-only / ctc+lm).
    train("base", {})

    # ---- Loquacious AED+CTC, chunked (tight, streaming-realistic) ----
    # From exp2025_10_21_chunked_ctc.py:py() -- the chunked-L80-C5-R4-v2.3-dyn-rope-ctembed
    # variant (around lines 410-441), same encoder/decoder as ``base`` but with
    # ChunkedConformerEncoderV2 (rotary self-att, chunk-type embedding, dynamic chunk pool).
    # RZ WER: 9.41 / 7.09 (ctc-only / ctc+lm).
    left_n, center_size, right_size, bs, max_seqs = 16, 5, 4, 50_000, 200
    name = f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-ctembed"
    train(
        name,
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # ---- LibriSpeech AED+CTC, offline ----
    # From exp2025_08_05_aed_large.py:py() (around line 2859) --
    # name "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k"
    # there; renamed here to "base-librispeech" for brevity.
    # Architecturally identical to the Loquacious ``base`` above
    # (bhv24, featBN, aux_loss_layers=[4,10,16], dec_aux_loss_layers=[3], spm10k);
    # the LS-specific difference is speed-pert ([0.7..1.1]) since LS is much smaller.
    # RZ WER (AED+CTC joint): dev-clean 1.87, dev-other 4.06, test-clean 2.06, test-other 4.38.
    _train_librispeech_base()


def _train_librispeech_base():
    """Verbatim copy of the LS aed_train_exp call from exp2025_08_05_aed_large.py.
    Don't refactor -- the call shape determines the Job hashes which we want to
    match RZ so rsync'd work/ dirs are recognised."""
    prefix = get_setup_prefix_for_module(__name__)
    ls_task = get_librispeech_task_raw_v2(vocab="spm10k")
    name = "base-librispeech"
    exp = aed_train_exp(
        name,
        configs.config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            # More futureproof, but also required for some funcs / setups.
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **configs._get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * configs._batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=ls_task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )


# ----------------------------------------------------------------------------
# Below: verbatim copies of ``_base_config``, ``train()``, and ``get_lm()``
# from ``exp2025_10_21_chunked_ctc.py``. **Do not refactor** without
# re-verifying that all downstream Job hashes still match the RZ recipe.
# ----------------------------------------------------------------------------


_base_config = {
    # ("large", 100),  # 100kh in total, 4 full epochs
    # ("large", 150),  # 150kh in total, 6 full epochs
    # ("large", 200),  # 200kh in total, 8 full epochs
    # ("large", 250),  # 250kh in total, 10 full epochs
    # ("large", 500),  # 500kh in total, 20 full epochs
    "subset": "large",
    "total_k_hours": 100,
    "vocab": "spm10k",
    "model": {
        "behavior_version": 24,
        "__serialization_version": 2,
        "enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
            ),
            num_layers=16,
            out_dim=1024,
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
        ),
        # Default AED decoder size: 6 layers, 512 dim
        "dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=6,
            model_dim=1024,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            # When only trained on LS ASR data, keep the default dropout?
            # dropout=0.0,
            # att_dropout=0.0,
        ),
        "feature_batch_norm": True,
    },
    "train_update_func_from_n_ep": lambda n_ep: {"train": configs._get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=0.5)},
    "train": dict_update_deep(
        configs.config_96gb_bf16_accgrad1,
        {
            "batch_size": 100_000 * configs._batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            # "__train_audio_preprocess": speed_pert_librosa_config,
            # "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
    ),
    "train_post": dict_update_deep(
        configs.post_config, {"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}}
    ),
    # TODO (for later): Bug: train_vocab_opts/dataset_train_opts are not actually plumbed through aed_train_exp.
    #   Not a quick fix: patching now would silently change every training run and break
    #   comparability with all existing results.
    #   Revisit when starting a fresh batch of experiments where breakage is acceptable.
    "train_vocab_opts": {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    "dataset_train_opts": {"train_epoch_split": 1, "train_epoch_wise_filter": None},
    "env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    "lm_recog_extra": {},
}


def train(
    name: str,
    config: Dict[str, Any],
    config_overrides: Optional[Dict[str, Any]] = None,
    *,
    recog_def_ctc_only: bool = True,
):
    prefix = get_setup_prefix_for_module(__name__)

    config = dict_update_deep(_base_config.copy(), config.copy())
    config = dict_update_deep(config, config_overrides, dict_value_merge=False)

    train_epoch_split_per_subset = {"clean": 13, "small": 1, "medium": 2, "large": 25}
    hours_per_subset = {"clean": 13_000, "small": 250, "medium": 2_500, "large": 25_000}
    subset = config.pop("subset")
    total_k_hours = config.pop("total_k_hours")
    train_epoch_split = train_epoch_split_per_subset[subset]
    num_full_ep = total_k_hours * 1_000 / hours_per_subset[subset]
    n_ep = round(num_full_ep * train_epoch_split)

    train_update_func_from_n_ep = config.pop("train_update_func_from_n_ep")
    if train_update_func_from_n_ep:
        config = dict_update_deep(config, train_update_func_from_n_ep(n_ep))

    model_config = config.pop("model")
    train_config: Dict[str, Any] = config.pop("train")
    post_config = config.pop("train_post")

    vocab = config.pop("vocab", "spm10k")
    task = get_loquacious_task_raw_v2(vocab=vocab, subset_name=subset, train_epoch_split=train_epoch_split)

    train_vocab_opts = config.pop("train_vocab_opts")
    dataset_train_opts = config.pop("dataset_train_opts")
    env_updates = config.pop("env_updates")
    lm_recog_extra_config = config.pop("lm_recog_extra")

    assert not config

    aux_ctc_layer = max(
        [i for i in train_config["aux_loss_layers"] if i <= model_config["enc_build_dict"]["num_layers"]]
    )

    exp = aed_train_exp(
        name,
        train_config,
        prefix=prefix + "/aed/",
        task=task,
        model_config=model_config,
        post_config_updates=post_config,
        vocab=vocab,
        train_vocab_opts=train_vocab_opts,
        dataset_train_opts=dataset_train_opts,
        env_updates=env_updates,
        recog_def=ctc_model_recog if recog_def_ctc_only else None,
        search_config={"aux_loss_layers": [aux_ctc_layer]} if recog_def_ctc_only else None,
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=aux_ctc_layer,
    )
    if vocab == "spm10k":
        lm_name, lm = get_lm(prefix=prefix, vocab=vocab)
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/aed/{name}/ctc+lm-v2/{lm_name}",
            task=task,
            ctc_model=exp.get_last_fixed_epoch(),
            extra_config={
                "aux_loss_layers": [
                    max(
                        [
                            i
                            for i in train_config["aux_loss_layers"]
                            if i <= model_config["enc_build_dict"]["num_layers"]
                        ]
                    )
                ],
                **lm_recog_extra_config,
            },
            lm=lm,
            prior_dataset=get_loquacious_train_subset_dataset_v2(vocab="spm10k"),
        )

    return exp, task, aux_ctc_layer


@cache
def get_lm(*, prefix: str, vocab: str, num_full_ep: int = 5, split: int = 10) -> Tuple[str, ModelWithCheckpoint]:
    from sisyphus import tk
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
        config_96gb_bf16_accgrad1,
        _get_cfg_lrlin_oclr_by_bs_nep_v4,
    )
    from i6_experiments.users.zeyer.decoding.perplexity import (
        get_lm_perplexities_for_task_evals,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import lm_model_def, lm_train_def
    from i6_experiments.users.zeyer.train_v4 import train as _train, ModelDefWithCfg

    from i6_experiments.users.zeyer.datasets.loquacious import (
        get_loquacious_task_raw_v2,
        get_loquacious_text_only_dataset,
    )

    import returnn.frontend as rf
    from returnn.frontend.decoder.transformer import TransformerDecoder

    n_ep = round(num_full_ep * split)
    # orig name: trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop01-b400_20k-nEp...-spm10k
    name = f"trafo-n32-d1024-nFullEp{num_full_ep}-nEp{n_ep}-{vocab}"
    exp = _train(
        f"{prefix}/lm/{name}",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep),
                "batch_size": 20_000,
                "max_seqs": 400,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        train_dataset=get_loquacious_text_only_dataset(vocab="spm10k", train_epoch_split=split),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=32,
                    model_dim=1024,
                    pos_enc=None,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    decoder_layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    dropout=0.1,
                    att_dropout=0.1,
                )
            },
        ),
        train_def=lm_train_def,
    )

    task = get_loquacious_task_raw_v2(vocab=vocab)
    perplexities_nlm = get_lm_perplexities_for_task_evals(task, label_level="task", lm=exp.get_last_fixed_epoch())
    for eval_set_name, ppl in perplexities_nlm.items():
        tk.register_output(f"{prefix}/lm/{name}/ppl/{eval_set_name}", ppl)

    return name, exp.get_last_fixed_epoch()
