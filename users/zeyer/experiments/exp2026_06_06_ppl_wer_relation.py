"""
PPL-WER relation study on LibriSpeech.

Gathers (word-level PPL, WER) per eval set across the LM scaling-law family
and plots one log-log scatter per eval set.

Reuses the LM family and CTC+LM WERs from
:mod:`i6_experiments.users.zeyer.experiments.exp2025_11_19_lm_scaling_laws`.
The only new compute here is per-LM PPL on each eval set
via :func:`...decoding.perplexity.get_lm_perplexities_for_task_evals_v2`.

Also hosts the LibriSpeech chunked-CTC encoder-context-length experiments
(RQ4: how encoder context length affects external-LM gain and the PPL-WER relation),
merged in from the former ``exp2026_05_27_chunked_ctc_ls`` recipe.
They register under the pinned ``exp2026_05_27_chunked_ctc_ls`` prefix (:data:`_LS_CHUNKED_PREFIX`),
so their output/alias paths are unchanged.
Sis job hashes are prefix-independent, so the merge re-runs nothing.
See :func:`_train_ls_chunked`.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from sisyphus import tk

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.utils.table_data import WriteTableDataJob
from i6_experiments.users.zeyer.plots.scaling_laws import ScalingLawPlotJob
from i6_experiments.users.zeyer.decoding.perplexity import get_lm_perplexities_for_task_evals_v2

# Imports for the merged LibriSpeech chunked-CTC experiments (see _train_ls_chunked).
from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    model_recog as ctc_model_recog,
)
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
    ChunkedConformerEncoderV2,
    ChunkedConformerEncoderLayerV2,
)

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)


__setup_root_prefix__ = "exp2026_06_06_ppl_wer_relation"

# Same baseline used by exp2025_11_19_lm_scaling_laws.
_CTC_MODEL_NAME = "L16-D1024-spm10k-auxAED-b100k"

# Pinned output/alias prefix for the merged LibriSpeech chunked-CTC experiments.
# Kept at the original recipe name so result/alias paths don't move;
# Sis hashes don't depend on it, so nothing re-runs.
_LS_CHUNKED_PREFIX = "exp2026_05_27_chunked_ctc_ls"


def py():
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_claix2023 import recog_ext_with_lm
    from i6_experiments.users.zeyer.train_v4 import train_models_by_prefix as train_v4_models
    from i6_experiments.users.zeyer.experiments.exp2025_11_19_lm_scaling_laws import (
        train_base_asr_models,
        train_lms,
    )

    # Populates train_v4_models with the LS LM scaling-law family + the CTC baseline.
    # Hashes match the imported work dirs (2025-08-22-dlm/work).
    train_base_asr_models()
    train_lms()

    # LibriSpeech chunked-CTC context-length experiments (RQ4 encoder-context axis),
    # merged in from the former exp2026_05_27_chunked_ctc_ls recipe.
    baseline_exp = _train_ls_chunked()

    prefix = get_setup_prefix_for_module(__name__)
    task = get_librispeech_task_raw_v2(vocab="spm10k")

    # eval set name -> list of (ppl_var, wer_var) across all LMs.
    points_per_eval: Dict[str, List[Tuple[tk.Variable, tk.Variable]]] = {}
    # Flat list of all (lm_name, eval_set, ppl, wer) points for the raw-number table.
    table_rows: List[dict] = []

    for lm_name, exp in train_v4_models.items():
        if not lm_name.startswith("lm/"):
            continue
        lm = exp.get_last_fixed_epoch()

        # Existing WER hash; sis just reads the imported result.
        res = recog_ext_with_lm(
            ctc_model_name=_CTC_MODEL_NAME,
            lm_name=lm_name[len("lm/") :],
            lm=lm,
            ctc_soft_collapse_threshold=0.9,
        )
        if res.individual_results is None:
            continue

        # New compute: word-level PPL per eval set.
        _task_ppl, word_ppl = get_lm_perplexities_for_task_evals_v2(task, lm=lm)

        for eval_name, score_res in res.individual_results.items():
            if eval_name not in word_ppl:
                continue
            wer_var = tk.Variable(path=score_res.main_measure_value.path, creator=score_res.main_measure_value.creator)
            ppl_var = word_ppl[eval_name]
            table_rows.append(
                {"lm_name": lm_name, "eval_set": eval_name, "ppl": ppl_var, "wer": score_res.main_measure_value}
            )
            # Gate on WER only (imported, so already available). PPL is new compute,
            # so ppl_var.available() is False until the PPL job runs -- don't gate on it.
            if not wer_var.available():
                continue
            points_per_eval.setdefault(eval_name, []).append((ppl_var, wer_var))

            tk.register_output(f"{prefix}/per_lm/{lm_name}/{eval_name}/ppl.txt", ppl_var)

    for eval_name, pts in points_per_eval.items():
        if not pts:
            continue
        tk.register_output(
            f"{prefix}/ppl_wer_{eval_name}.pdf",
            ScalingLawPlotJob(
                x_label="PPL",
                y_label="WER [%]",
                x_scale="log",
                y_scale="log",
                points={f"CTC+LM ({eval_name})": pts},
                filter_outliers=False,
            ).out_plot_pdf,
        )

    # Raw PPL-WER numbers (all LMs, all eval sets) as a TSV + JSON table.
    table_job = WriteTableDataJob(
        columns=["lm_name", "eval_set", "ppl", "wer"],
        rows=table_rows,
        sort_by=["eval_set", "ppl"],
    )
    tk.register_output(f"{prefix}/ppl_wer_table.tsv", table_job.out_tsv)
    tk.register_output(f"{prefix}/ppl_wer_table.json", table_job.out_json)

    # LM softmax-temperature sweep on the offline baseline (new compute).
    _lm_softmax_temperature_sweep(baseline_exp)


def _lm_softmax_temperature_sweep(baseline_exp) -> None:
    """
    LM softmax-temperature sweep on the offline LS baseline CTC + LS Transformer-LM.

    Varies the external-LM softmax temperature T (LM score = ``log_softmax(lm_logits / T)``),
    which simultaneously changes the LM word-level perplexity
    and the CTC+LM rescoring WER,
    tracing a temperature-parametrized PPL-WER curve for one fixed AM + LM pair.

    For each T the LM/prior scales are re-tuned on the N-best list,
    and we use the *rescore* WER (``rescore-res.txt``),
    which goes through the tempered ``lm_rescore_def`` and is thus consistent with the tempered PPL.
    The first-pass decode does not see T, so its ``recog-1stpass-res.txt`` is not used here.
    T == 1.0 omits the temperature key,
    so its recog/PPL hashes match the untempered baseline and reuse the finished result.
    """
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
        ctc_recog_recomb_labelwise_prior_auto_scale,
        _get_lm_model,
        _lms,
    )

    prefix = get_setup_prefix_for_module(__name__)
    task = get_librispeech_task_raw_v2(vocab="spm10k")
    lm_name = "n32-d1024-claix2023"
    lm = _get_lm_model(_lms[lm_name])
    ctc_model = baseline_exp.get_last_fixed_epoch()

    temperatures = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    for temp in temperatures:
        lm_rescore_config = None if temp == 1.0 else {"lm_softmax_temperature": temp}
        t_tag = f"T{temp:g}"
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/lm_temp_sweep/{t_tag}/ctc+lm",
            task=task,
            ctc_model=ctc_model,
            extra_config={"aux_loss_layers": [16]},
            lm=lm,
            lm_rescore_config=lm_rescore_config,
        )
        _task_ppl, word_ppl = get_lm_perplexities_for_task_evals_v2(task, lm=lm, config=lm_rescore_config)
        for eval_name, ppl_var in word_ppl.items():
            tk.register_output(f"{prefix}/lm_temp_sweep/{t_tag}/{eval_name}/ppl.txt", ppl_var)


def _train_ls_chunked():
    """
    LibriSpeech chunked-CTC encoder-context-length experiments (RQ4).

    LS counterpart of :mod:`exp2025_10_21_chunked_ctc` (Loquacious):
    same LS data + recipe as the offline AED baseline
    ``EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k``
    in :mod:`exp2025_08_05_aed_large`,
    only the encoder is swapped for :class:`ChunkedConformerEncoderV2`.

    Sweeps ``(left_ctx, center, right_lookahead)`` configs
    (encoder frames after the ``ConformerConvSubsample`` /6 downsampling, so one frame ~= 60 ms):
    a limited-context causal sweep (zero history, zero lookahead) over chunk size,
    plus an (80, 5, 4) streaming reference.

    Merged here from the former ``exp2026_05_27_chunked_ctc_ls`` recipe;
    registers under the pinned :data:`_LS_CHUNKED_PREFIX`.
    """
    # Register the offline LS AED baseline under the pinned prefix
    # with a byte-identical ``aed_train_exp(...)`` call,
    # so its ``ReturnnTrainingJob`` hash matches the existing trained model
    # at ``~/setups/2025-08-aed-large/work/.../ReturnnTrainingJob.IVB5xAuHZZA3``.
    # The bulk-import script (``import_work_directory.py``) then symlinks
    # that finished training in here -- no re-training.
    baseline_exp = _train_ls_offline_baseline()

    # ``(left_ctx, center, right_lookahead)`` in encoder frames (~60 ms each).
    # Causal variants (R=0) explore the encoder-only-latency budget;
    # (80, 5, 4) is the reference-streaming config matching the Loquacious sweep.
    # Batch size shrinks for the larger-center variants
    # because their effective per-step compute grows;
    # the values mirror the limited-history block of :mod:`exp2025_10_21_chunked_ctc`.
    for left, center, right, bs in [
        (0, 5, 0, 75_000),
        (0, 10, 0, 75_000),
        (0, 20, 0, 75_000),
        (0, 40, 0, 75_000),
        (0, 80, 0, 75_000),
        (0, 160, 0, 75_000),
        (0, 200, 0, 75_000),
        (0, 260, 0, 75_000),
        (0, 320, 0, 75_000),
        (0, 160, 0, 50_000),
        (80, 5, 4, 50_000),
    ]:
        _name_post = ""
        if bs != 75_000:
            _name_post += f"-bs{bs // 1000}k"
        _train_ls(
            f"chunked-L{left}-C{center}-R{right}-v2.3{_name_post}",
            chunk_size=center,
            chunk_history_size=left,
            chunk_lookahead_size=right,
            batch_size=bs,
        )

    return baseline_exp


def _train_ls_offline_baseline():
    """
    Re-register the offline LS AED baseline
    ``EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k``
    (originally trained in :mod:`exp2025_08_05_aed_large`)
    under the pinned :data:`_LS_CHUNKED_PREFIX`.

    The :func:`aed_train_exp` call is byte-identical to the original
    (encoder = :class:`ConformerEncoder`, no chunking;
    no ``recog_def`` / ``search_config`` / ``max_seqs`` -- those are not passed in the original).
    The resulting ``ReturnnTrainingJob`` hash therefore matches the existing
    ``IVB5xAuHZZA3`` in ``~/setups/2025-08-aed-large/``,
    so the bulk-import via ``import_work_directory.py`` symlinks the trained model
    instead of triggering a new training.

    The ``aed_ctc_timesync_recog_recomb_auto_scale`` call mirrors the original;
    it depends on ``get_librispeech_task_raw_v2(vocab="spm10k")`` *without* the
    ``train_vocab_opts`` / ``train_epoch_split`` kwargs (the training task uses those,
    but the recog task is built default-style, matching the original).
    """
    prefix = _LS_CHUNKED_PREFIX
    name = "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k"
    exp = aed_train_exp(
        name,
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
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
                        ConformerPositionwiseFeedForward,
                        activation=rf.build_dict(rf.relu_square),
                        with_bias=False,
                    ),
                    num_heads=8,
                ),
            ),
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    task_spm10k = get_librispeech_task_raw_v2(vocab="spm10k")
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )
    _run_ls_ctc_lm(name, exp, task_spm10k)
    _run_ls_align_stats("base-librispeech", exp, 16)
    return exp


def _run_ls_align_stats(name: str, exp, aux_ctc_layer: int) -> None:
    """TIMIT WBE/TSE quality check for the LS-trained offline AED+CTC baseline
    (``ReturnnTrainingJob.IVB5xAuHZZA3``, same arch as Loquacious ``base`` but trained on LS-960h only).
    Output: ``align-stats/<name>/timit-{val,test}/{alignment.hdf, wbe.txt, report.txt}``
    under the pinned :data:`_LS_CHUNKED_PREFIX`.

    Mirrors :func:`exp2025_10_21_chunked_ctc._run_align_stats` but registers under
    the pinned ``exp2026_05_27_chunked_ctc_ls`` prefix; the underlying
    forced-align + WBE Job hashes are identical since we reuse the same job-construction
    helpers from chunked-CTC -- the same Job dirs are shared with the Loquacious
    ``base`` align-stats jobs.

    Loquacious ``base`` reference (chunked-ctc setup): WBE 0.140 s (test), 0.125 s (val).
    """
    from sisyphus import tk
    from i6_experiments.users.zeyer.datasets.loquacious import get_vocab_by_str
    from i6_experiments.users.zeyer.datasets.hf_timit_buckeye import (
        get_dataset_offset_factor,
        get_hf_word_align_dataset_config,
        get_hf_word_align_dataset_dir,
    )
    from i6_experiments.users.zeyer.alignment.ctc_wbe_from_hf_dataset import CalcCtcWbeFromHfDatasetJob
    from i6_experiments.users.zeyer.experiments.exp2025_10_21_chunked_ctc import _aed_ctc_forced_align

    prefix = _LS_CHUNKED_PREFIX + "/align-stats/" + name
    vocab = get_vocab_by_str("spm10k")
    model = exp.get_last_fixed_epoch()

    # The spm10k vocab is all-uppercase (Loquacious-trained, shared with the LS
    # spm10k tokenizer), so we must uppercase the joined TIMIT utterance --
    # otherwise every word tokenizes to <unk> silently. Matches the recent
    # chunked-CTC ``_run_align_stats`` (line ~1375 there).
    text_case = "upper"
    for corpus in ["timit"]:
        ds_dir = get_hf_word_align_dataset_dir(corpus, text_case=text_case)
        offset_factor = get_dataset_offset_factor(corpus)
        for split in ["val", "test"]:
            tag = f"{corpus}-{split}"
            ds = get_hf_word_align_dataset_config(name=corpus, split=split, vocab=vocab, text_case=text_case)
            alignment_hdf = _aed_ctc_forced_align(model, ds, aux_ctc_layer=aux_ctc_layer)
            alignment_hdf.creator.add_alias(f"{prefix}/{tag}/forced-align")
            tk.register_output(f"{prefix}/{tag}/alignment.hdf", alignment_hdf)

            metrics_job = CalcCtcWbeFromHfDatasetJob(
                alignment_hdf=alignment_hdf,
                spm_model_file=vocab.model_file,
                blank_idx=vocab.dim,
                dataset_dir=ds_dir,
                dataset_key=split,
                dataset_offset_factor=offset_factor,
            )
            metrics_job.add_alias(f"{prefix}/{tag}/wbe-metric")
            tk.register_output(f"{prefix}/{tag}/wbe.txt", metrics_job.out_wbe)
            tk.register_output(f"{prefix}/{tag}/report.txt", metrics_job.out_report)


def _train_ls(name: str, *, chunk_size: int, chunk_history_size: int, chunk_lookahead_size: int, batch_size: int):
    """
    Train one chunked-CTC LS experiment.

    All hyperparameters mirror the offline LS baseline
    ``EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k``
    in :mod:`exp2025_08_05_aed_large`.
    The only changes are:
    encoder = :class:`ChunkedConformerEncoderV2` (v3) with the given chunking,
    and a smaller ``batch_size`` to absorb the chunked memory overhead.
    """
    prefix = _LS_CHUNKED_PREFIX
    # ``get_librispeech_task_raw_v2`` is cached, so the same call inside ``aed_train_exp``
    # (which we don't pass ``task=`` to, letting it default to LS via the same factory)
    # returns this exact Task object -- they stay consistent without explicit threading.
    train_vocab_opts = {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}}
    dataset_train_opts = {"train_epoch_split": 1, "train_epoch_wise_filter": None}
    task = get_librispeech_task_raw_v2(vocab="spm10k", train_vocab_opts=train_vocab_opts, **dataset_train_opts)
    exp = aed_train_exp(
        name,
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
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
                    ChunkedConformerEncoderLayerV2,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
                chunk_size=chunk_size,
                chunk_history_size=chunk_history_size,
                chunk_lookahead_size=chunk_lookahead_size,
                version=3,
            ),
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": batch_size * _batch_size_factor,
            "max_seqs": 200,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts=train_vocab_opts,
        dataset_train_opts=dataset_train_opts,
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        recog_def=ctc_model_recog,
        search_config={"aux_loss_layers": [16]},
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )
    _run_ls_ctc_lm(name, exp, task)
    return exp


def _run_ls_ctc_lm(name: str, exp, task, aux_ctc_layer: int = 16) -> None:
    """
    CTC + LibriSpeech-LM recog (``ctc+lm-v2``) for an LS chunked-CTC model.

    Uses the proper LibriSpeech-trained spm10k Transformer LM ``n32-d1024-claix2023``
    (``trafo-n32-d1024-...-b400_20k-spm10k``, trained on ``get_librispeech_lm_dataset``)
    resolved via the shared :func:`_get_lm_model` / :data:`_lms` registry,
    so no new LM training is triggered -- the finished LM job is reused (import it if absent).

    Recog is ``ctc_recog_recomb_labelwise_prior_auto_scale`` (same ``ctc+lm-v2`` path as the Loquacious setup):
    the CTC head is read from aux layer ``aux_ctc_layer``,
    and the label prior is estimated on this task's LS train data.
    """
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
        ctc_recog_recomb_labelwise_prior_auto_scale,
        _get_lm_model,
        _lms,
    )

    prefix = _LS_CHUNKED_PREFIX
    lm_name = "n32-d1024-claix2023"
    lm = _get_lm_model(_lms[lm_name])
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-v2/{lm_name}",
        task=task,
        ctc_model=exp.get_last_fixed_epoch(),
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=lm,
    )
