"""
Alignment behavior of CTC: full-sum training variants on LibriSpeech.

Project note: ``2026-05-25-align-behavior.md``.

Trains / re-uses four CTC variants on a 16-layer Conformer (D1024) baseline
with AED auxiliary loss, SPM-10k vocabulary, 100 epochs of LibriSpeech-960:

- ``L16-D1024-spm10k-auxAED-b100k`` -- vanilla.
- ``L16-D1024-spm10k-auxAED-blankSep-b100k`` -- hierarchical-softmax CTC
  with a sigmoid blank unit.
- ``L16-D1024-spm10k-auxAED-b100k-lpNormedGradC05_11P1`` -- normalized
  gradient, batch-based prior estimate, clamp 0.5..1.1, prior_exp=1.0.
- ``L16-D1024-spm10k-auxAED-b100k-lpNormedGradC05_11P1Seq`` -- as above
  but per-sequence prior estimate.

The first two are already trained in :mod:`exp2024_04_23_baselines.ctc_claix2023`.
The ``ctc_train_exp`` calls here are byte-identical to those, so the hashes
match and the existing checkpoints in ``setups/combined/2021-05-31/work``
can be symlinked in via ``import_work_directory.py``. The last two are new
and need training (~3 days wall-clock each).

For each trained variant: forced-align on a Librispeech-960 training subset
with several prior / blank-logit-shift opt combinations, then TSE / WER
metrics against the GMM reference alignment.

Predecessor for forced-align + metric helpers: :mod:`exp2024_09_16_grad_align`.
The grad-align extraction block from that recipe is deliberately not called
here -- it is driven by :mod:`exp2026_05_23_grad_align` instead.
"""

from __future__ import annotations

from sisyphus import tk, Path, gs

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerPositionwiseFeedForward,
    ConformerConvSubsample,
)
from i6_experiments.common.setups import serialization

# CTC training entry point + shared baseline config knobs.
from .exp2024_04_23_baselines.ctc import (
    train_exp as ctc_train_exp,
    _raw_sample_rate,
)
from .exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
    _batch_size_factor,
)
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config

# Forced-align + metric helpers (imported, never modified -- shared with predecessor recipes).
from .exp2024_09_09_grad_align import CalcAlignmentMetrics
from .exp2024_09_16_grad_align import (
    ctc_forced_align,
    get_ctc_prior,
)


# LibriSpeech GMM reference alignment (Tina's setup) + Schmitt's seq_list.
# The data originally lived on i6 (``/work/common/...``) and ``/u/schmitt/...``.
# For the RWTH HPC, the same files were mirrored under
# ``/hpcwork/az668407/setups-data/librispeech/data-from-i6/`` (the
# ``.cache.bundle`` files were sed-rewritten to point at the new locations);
# see project note ``2026-05-25-align-behavior.md``.
#
# Cluster detection: ``/hpcwork`` only exists on the RWTH HPC. ``_i6_ref``
# picks the rz mirror if so, otherwise the canonical i6 path. ``hash_overwrite``
# is always pinned to the i6 path so the resulting Sis job hashes are stable
# across clusters.
import os as _os

_IS_RZ = _os.path.exists("/hpcwork")

# Each entry maps an i6 canonical prefix to its rz mirror. Longest-prefix wins,
# so list more specific entries first. Exact-file mappings are fine here too --
# the seq_list lives in a different i6 tree and has a renamed mirror.
_I6_TO_RZ_MIRRORS = [
    (
        "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1",
        "/hpcwork/az668407/setups-data/librispeech/data-from-i6/seq-list-from-schmitt.1",
    ),
    (
        "/work/common/asr/librispeech/data",
        "/hpcwork/az668407/setups-data/librispeech/data-from-i6",
    ),
]


def _i6_ref(i6_path: str) -> Path:
    """Resolve a canonical i6 reference path to its rz mirror (or the original).

    ``hash_overwrite`` is pinned to the i6 form so Sis job hashes stay stable
    regardless of which cluster the recipe runs on.
    """
    if _IS_RZ:
        for i6_pre, rz_pre in _I6_TO_RZ_MIRRORS:
            if i6_path == i6_pre or i6_path.startswith(i6_pre + "/"):
                return Path(rz_pre + i6_path[len(i6_pre) :], hash_overwrite=i6_path)
        raise ValueError(f"_i6_ref: no rz mirror configured for {i6_path!r}")
    return Path(i6_path, hash_overwrite=i6_path)


_GMM_ALIGNMENT_ALLOPHONES = _i6_ref(
    "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
)
_GMM_ALIGNMENT_SPRINT_CACHE = _i6_ref(
    "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"
)
_FEATURES_SPRINT_CACHE = _i6_ref(
    "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/features/extraction/FeatureExtractionJob.VTLN.upmU2hTb8dNH/output/vtln.cache.bundle"
)
_SEQ_LIST_REF = _i6_ref(
    "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
)


def _enc_build_dict_l16_d1024():
    """Encoder spec for the L16-D1024 baseline. Mirrors ``ctc_claix2023.py``."""
    return rf.build_dict(
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
    )


def _train_base():
    """``L16-D1024-spm10k-auxAED-b100k`` -- byte-identical to ctc_claix2023.py."""
    return ctc_train_exp(
        "L16-D1024-spm10k-auxAED-b100k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_build_dict": _enc_build_dict_l16_d1024(),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )


def _train_blanksep():
    """``L16-D1024-spm10k-auxAED-blankSep-b100k`` -- byte-identical to ctc_claix2023.py."""
    return ctc_train_exp(
        "L16-D1024-spm10k-auxAED-blankSep-b100k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_build_dict": _enc_build_dict_l16_d1024(),
            "enc_logits_with_bias": False,
            "out_blank_separated": True,
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
            "use_fixed_ctc_grad": "v2",
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


_LPNORMEDGRAD_VARIANTS = {
    "lpNormedGradC05_11P1": {
        "func": {"clamp_min": 0.5, "clamp_max": 1.1, "scale_type": "inv_num_labels", "prior_exp": 1.0},
    },
    "lpNormedGradC05_11P1Seq": {
        "prior": "seq_grad",
        "func": {"clamp_min": 0.5, "clamp_max": 1.1, "scale_type": "inv_num_labels", "prior_exp": 1.0},
    },
}


def _train_lpnormedgrad(name_suffix: str, log_prob_normed_grad_opts: dict):
    """New: normalized-gradient variant on the L16-D1024 baseline.

    The ``epilog`` adds ``2024-alignment-analysis`` (the synth-framework repo,
    source of the ``NormedGradientFuncInvPrior`` runtime impl) to ``sys.path``
    at config-load time. Mirrors the n12 NG block in ``ctc_claix2023.py``.
    """
    return ctc_train_exp(
        f"L16-D1024-spm10k-auxAED-b100k-{name_suffix}",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_build_dict": _enc_build_dict_l16_d1024(),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
            "use_fixed_ctc_grad": "v2",
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "log_prob_normed_grad": log_prob_normed_grad_opts,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        epilog=[
            serialization.NonhashedCode(f"sys.path.append({gs.BASE_DIR + '/projects/2024-alignment-analysis'!r})\n")
        ],
    )


def py():
    """Sisyphus entry point."""
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_raw_v2,
        get_vocab_by_str,
        seq_list_960_to_split_100_360_500,
    )
    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

    prefix = "exp2026_05_25_align_behavior/"

    # ---- Train (or import if hash matches existing) ----
    variants = [
        ("base", _train_base()),
        ("blankSep", _train_blanksep()),
    ]
    for ng_name, ng_opts in _LPNORMEDGRAD_VARIANTS.items():
        variants.append((ng_name, _train_lpnormedgrad(ng_name, ng_opts)))

    # ---- Forced-align + TSE / WER metrics on a 960h training subset ----
    vocab = "spm10k"
    vocab_meta = ("spm", ExtractSentencePieceVocabJob(get_vocab_by_str(vocab).model_file).out_vocab, 10_240)
    seq_list = seq_list_960_to_split_100_360_500(_SEQ_LIST_REF)
    task = get_librispeech_task_raw_v2(vocab=vocab)
    train_dataset = task.train_dataset.copy_train_as_static()
    train_dataset.main_dataset["seq_list_filter_file"] = seq_list

    for shortname, model_with_checkpoints in variants:
        if model_with_checkpoints is None:
            continue  # train_exp returned None (e.g. enabled=False)
        ctc_model = model_with_checkpoints.get_last_fixed_epoch()

        # Prior on full train dataset (no seq_list filter).
        prior_stats = get_ctc_prior(ctc_model, task.train_dataset.copy_train_as_static(), {"fix_log_probs": True})
        prior_stats.mean.creator.add_alias(f"{prefix}ctc_prior/{shortname}/prior_stats_full")
        tk.register_output(f"{prefix}ctc_prior/{shortname}/prior_stats_full.mean.txt", prior_stats.mean)

        opts_variants = [{}]
        for shift in [0, -5, -10, -15, -18, -20, -25]:
            opts_variants.append({"fix_log_probs": True, "blank_logit_shift": shift})
        for am_scale, prior_scale in [(1.0, 1.0), (1.0, 1.5), (1.0, 2.0), (1.0, 3.0)]:
            opts_variants.append(
                {
                    "fix_log_probs": True,
                    "ctc_prior_type": "static",
                    "static_prior": {"type": "prob", "file": prior_stats.mean},
                    "ctc_am_scale": am_scale,
                    "ctc_prior_scale": prior_scale,
                }
            )
        for shift in [-5, -10, -15, -20]:
            opts_variants.append(
                {
                    "fix_log_probs": True,
                    "blank_logit_shift": shift,
                    "ctc_prior_type": "static",
                    "static_prior": {"type": "prob", "file": prior_stats.mean},
                    "ctc_am_scale": 1.0,
                    "ctc_prior_scale": 1.0,
                }
            )

        for opts in opts_variants:
            name = shortname
            for k, v in opts.items():
                if k == "fix_log_probs" and v:
                    name += "-fix"
                    continue
                if k == "static_prior":
                    continue
                name += f"-{k}{v}"

            prefix_ = f"{prefix}ctc_forced_align/{name}/"
            alignment = ctc_forced_align(ctc_model, train_dataset, opts)
            alignment.creator.add_alias(f"{prefix_}align")
            tk.register_output(f"{prefix_}align.hdf", alignment)

            job = CalcAlignmentMetrics(
                seq_list=seq_list,
                seq_list_ref=_SEQ_LIST_REF,
                alignment_hdf=alignment,
                alignment_label_topology="ctc",
                alignment_bpe_vocab=vocab_meta[1],
                alignment_bpe_style=vocab_meta[0],
                alignment_blank_idx=vocab_meta[2],
                features_sprint_cache=_FEATURES_SPRINT_CACHE,
                ref_alignment_sprint_cache=_GMM_ALIGNMENT_SPRINT_CACHE,
                ref_alignment_allophones=_GMM_ALIGNMENT_ALLOPHONES,
                ref_alignment_len_factor=6,
            )
            job.add_alias(f"{prefix_}align-metrics")
            tk.register_output(f"{prefix_}align-metrics.json", job.out_scores)
            tk.register_output(f"{prefix_}align-metrics.short_report.txt", job.out_short_report_str)

    # NOTE Synthetic-framework experiments (FFNN / Conv / LSTM on artificial data)
    # run from the standalone repo ``2024-alignment-analysis`` -- laptop-CPU, no Sis.
