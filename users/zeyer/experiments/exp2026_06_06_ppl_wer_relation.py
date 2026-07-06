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
    # LibriSpeech chunked-CTC context-length experiments (RQ4 encoder-context axis),
    # merged in from the former exp2026_05_27_chunked_ctc_ls recipe.
    # Passes the offline-baseline CTC+LM result for the context-length table.
    _train_ls_chunked()

    # LM softmax-temperature sweep on the offline baseline (new compute).
    _lm_softmax_temperature_sweep()

    # Extra small/short LMs to fill the sparse high-PPL end of the ppl-vs-WER curve.
    # Registered before _lm_family_ppl_wer so they are picked up for recog + PPL.
    _extra_high_ppl_lms()

    # PPL-WER relation across the external-LM scaling-law family.
    _lm_family_ppl_wer()

    # CTC + LibriSpeech-finetuned LLM recognition.
    _ctc_llm_recog()


def _lm_family_ppl_wer() -> None:
    """
    PPL-WER relation across the LM scaling-law family on LibriSpeech.

    Gathers spm10k-label-level and word-level PPL plus WER per eval set for every external LM,
    registers one log-log scatter (word PPL vs WER) per eval set,
    and writes the raw numbers as TSV + JSON tables, one per PPL level.
    """
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

        # New compute: spm10k-label-level and word-level PPL per eval set.
        spm10k_ppl, word_ppl = get_lm_perplexities_for_task_evals_v2(task, lm=lm)

        for eval_name, score_res in res.individual_results.items():
            if eval_name not in word_ppl:
                continue
            wer_var = tk.Variable(path=score_res.main_measure_value.path, creator=score_res.main_measure_value.creator)
            word_ppl_var = word_ppl[eval_name]
            spm10k_ppl_var = spm10k_ppl[eval_name]
            table_rows.append(
                {
                    "lm_name": lm_name,
                    "eval_set": eval_name,
                    "spm10k_ppl": spm10k_ppl_var,
                    "word_ppl": word_ppl_var,
                    "wer": score_res.main_measure_value,
                }
            )
            # Gate on WER only (imported, so already available). PPL is new compute,
            # so the PPL vars' .available() is False until the PPL job runs -- don't gate on them.
            if not wer_var.available():
                continue
            # Scatter uses word PPL vs WER.
            points_per_eval.setdefault(eval_name, []).append((word_ppl_var, wer_var))

            tk.register_output(f"{prefix}/per_lm/{lm_name}/{eval_name}/word_ppl.txt", word_ppl_var)
            tk.register_output(f"{prefix}/per_lm/{lm_name}/{eval_name}/spm10k_ppl.txt", spm10k_ppl_var)

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

    # Raw PPL-WER numbers (all LMs, all eval sets) as TSV + JSON tables,
    # one table per PPL level (spm10k-label PPL and word PPL).
    for ppl_col, tag in [("spm10k_ppl", "spm10k"), ("word_ppl", "word")]:
        table_job = WriteTableDataJob(
            columns=["lm_name", "eval_set", ppl_col, "wer"],
            rows=table_rows,
            sort_by=["eval_set", ppl_col],
        )
        tk.register_output(f"{prefix}/ppl_wer_table_{tag}.tsv", table_job.out_tsv)
        tk.register_output(f"{prefix}/ppl_wer_table_{tag}.json", table_job.out_json)


def _extra_high_ppl_lms() -> None:
    """
    Extra small / short-schedule LM trainings to fill the sparse high-PPL region
    (word PPL >~ 90) of the PPL-vs-WER curve.

    The scaling-law family (:func:`exp2025_11_19_lm_scaling_laws.train_lms`) has only a
    handful of LMs above ~90 word PPL, so that end of the curve is thinly sampled.
    These are all tiny (few-layer, small-dim) and/or short-schedule Transformer LMs,
    which train fast, and are aimed at high perplexity to extend the curve.

    They use the exact scaling-law LM shape and the same ``lm/trafo-v2-...-spm10k`` naming,
    and register into the shared ``train_v4`` registry,
    so :func:`_lm_family_ppl_wer` picks them up automatically for CTC+LM recog + PPL --
    no extra per-LM wiring is needed here.
    """
    # Knobs: n=num_layers, d=model_dim, nEp=sub-epochs (default 100), lr (default 1.0), a=num_heads.
    # None of these overlap the existing family opts (would otherwise re-register the same job).
    for opts in [
        {"n": 1, "d": 32},
        {"n": 1, "d": 64},
        {"n": 1, "d": 128},
        {"n": 1, "d": 256},
        {"n": 1, "d": 512},
        {"n": 2, "d": 64},
        {"n": 2, "d": 128, "nEp": 20},
        {"n": 2, "d": 128, "nEp": 50},
        {"n": 2, "d": 256},
        {"n": 2, "d": 256, "nEp": 50},
        {"n": 2, "d": 512},
        {"n": 3, "d": 128},
        {"n": 4, "d": 128},
        {"n": 6, "d": 256},
    ]:
        _train_lm(opts)

    # Low-epoch sweep on a small base (n4-d256): undertraining pushes word PPL up.
    # Extended down to nEp1 (=1/20 data pass) to reach the far high-PPL end (~word 200-450),
    # needed to approach the No-LM WER bar; nEp10-50 land ~110-150.
    for n_ep in [1, 2, 3, 5, 10, 20, 30, 40, 50]:
        _train_lm({"n": 4, "d": 256, "nEp": n_ep})

    # Second low-epoch sweep on a larger base (n6-d512), better GPU util than n4-d256.
    # Extended down to nEp1 for the high-PPL end; nEp10-50 land ~85-112.
    for n_ep in [1, 2, 3, 5, 10, 20, 30, 40, 50]:
        _train_lm({"n": 6, "d": 512, "nEp": n_ep})


def _train_lm(opts: dict) -> None:
    """
    Train one LM in the scaling-law family shape
    (mirrors the ``train(...)`` call in :func:`exp2025_11_19_lm_scaling_laws.train_lms`),
    registered into the shared ``train_v4`` registry as ``lm/trafo-v2-{opts}-spm10k``.

    Kept local here (not added to the shared scaling-law module) on purpose.
    """
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from i6_experiments.users.zeyer.train_v4 import train
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
    from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
        config_96gb_bf16_accgrad1,
        _get_cfg_lrlin_oclr_by_bs_nep_v3,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm_claix2023 import (
        lm_model_def,
        lm_train_def,
    )

    # Reorder + defaults, exactly as in the scaling-law loop, so names/hashes stay consistent.
    opts = {"n": opts.pop("n", 32), "d": opts.pop("d", 1024), **opts}
    name = f"lm/trafo-v2-{_name_for_lm_opts(opts)}-spm10k"
    n_l = opts.pop("n")
    dim = opts.pop("d")
    n_ep = opts.pop("nEp", 100)
    lr = opts.pop("lr", 1.0)
    num_heads = opts.pop("a", None)
    drop = opts.pop("drop", 0.0)
    att_drop = opts.pop("adrop", drop)
    assert not opts, f"unexpected LM opts left: {opts}"
    train(
        name,
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(20_000, n_ep, base_lr=lr, batch_size_factor=1),
                "max_seqs": 400,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                # These LMs run <=~4.5h; request 5h (not the 80h default capped to 24h) so they backfill.
                # __time_rqmt is popped in train_v4.train before hashing, so this does not move any job hash.
                "__time_rqmt": 5,
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k", train_epoch_split=20),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=n_l,
                    model_dim=dim,
                    pos_enc=None,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    decoder_layer_opts=dict(
                        self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False),
                        **({"num_heads": num_heads} if num_heads is not None else {}),
                    ),
                    dropout=drop,
                    att_dropout=att_drop,
                )
            },
        ),
        train_def=lm_train_def,
    )


def _name_for_lm_opts(cfg: dict) -> str:
    """Name suffix for an LM opts dict; matches exp2025_11_19_lm_scaling_laws._name_for_dict."""
    parts = []
    for k, v in cfg.items():
        v = str(v).replace(".", "_").replace("-", "_")
        parts.append(f"{k}{v}")
    return "-".join(parts)


def _lm_softmax_temperature_sweep() -> None:
    """
    LM softmax-temperature sweep on the offline LS baseline CTC + LS Transformer-LM.

    Varies the external-LM softmax temperature T (LM score = ``log_softmax(lm_logits / T)``),
    which simultaneously changes the LM word-level perplexity and the CTC+LM WER,
    tracing a temperature-parametrized PPL-WER curve for one fixed AM + LM pair.

    For each T the LM/prior scales are re-tuned on the N-best list.
    T is applied wherever the LM is scored (scale tuning, rescore, and the first-pass search),
    so the reported first-pass WER is consistent with the tempered PPL.
    T == 1.0 omits the temperature key,
    so its recog/PPL hashes match the untempered baseline and reuse the finished result.

    Writes the raw (T, eval set, spm10k/word PPL, first-pass WER) numbers as TSV + JSON tables, one per PPL level.
    """
    from i6_experiments.users.zeyer.experiments.exp2025_11_19_lm_scaling_laws import train_base_asr_models
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import _train_experiments
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
        ctc_recog_recomb_labelwise_prior_auto_scale,
        _get_lm_model,
        _lms,
    )

    # Populates _train_experiments with the CTC baseline (same one _lm_family_ppl_wer recogs).
    train_base_asr_models()
    prefix = get_setup_prefix_for_module(__name__)
    task = get_librispeech_task_raw_v2(vocab="spm10k")
    lm_name = "n32-d1024-claix2023"
    lm = _get_lm_model(_lms[lm_name])
    ctc_model = _train_experiments[_CTC_MODEL_NAME].get_last_fixed_epoch()

    temperatures = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    sweep_rows: List[dict] = []
    for temp in temperatures:
        lm_rescore_config = None if temp == 1.0 else {"lm_softmax_temperature": temp}
        t_tag = f"T{temp:g}"
        res = ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/lm_temp_sweep/{t_tag}/ctc+lm",
            task=task,
            ctc_model=ctc_model,
            lm=lm,
            lm_rescore_config=lm_rescore_config,
        )
        spm10k_ppl, word_ppl = get_lm_perplexities_for_task_evals_v2(task, lm=lm, config=lm_rescore_config)
        wer_per_eval = res.individual_results or {}
        for eval_name, word_ppl_var in word_ppl.items():
            tk.register_output(f"{prefix}/lm_temp_sweep/{t_tag}/{eval_name}/word_ppl.txt", word_ppl_var)
            tk.register_output(f"{prefix}/lm_temp_sweep/{t_tag}/{eval_name}/spm10k_ppl.txt", spm10k_ppl[eval_name])
            score_res = wer_per_eval.get(eval_name)
            sweep_rows.append(
                {
                    "temperature": temp,
                    "eval_set": eval_name,
                    "spm10k_ppl": spm10k_ppl[eval_name],
                    "word_ppl": word_ppl_var,
                    "wer": score_res.main_measure_value if score_res is not None else None,
                }
            )

    # One table per PPL level (spm10k-label PPL and word PPL).
    for ppl_col, tag in [("spm10k_ppl", "spm10k"), ("word_ppl", "word")]:
        table_job = WriteTableDataJob(
            columns=["temperature", "eval_set", ppl_col, "wer"],
            rows=sweep_rows,
            sort_by=["eval_set", "temperature"],
        )
        tk.register_output(f"{prefix}/lm_temp_sweep/ppl_wer_table_{tag}.tsv", table_job.out_tsv)
        tk.register_output(f"{prefix}/lm_temp_sweep/ppl_wer_table_{tag}.json", table_job.out_json)


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
    # Register the offline LS AED baseline under the pinned prefix.
    # Also returns the CTC+LM result used for the context-length table.
    _, baseline_ctc_lm_result = _train_ls_offline_baseline()

    # ``(left_ctx, center, right_lookahead)`` in encoder frames (~60 ms each).
    # Causal variants (R=0) explore the encoder-only-latency budget;
    # (80, 5, 4) is the reference-streaming config matching the Loquacious sweep.
    # Batch size shrinks for the larger-center variants
    # because their effective per-step compute grows;
    # the values mirror the limited-history block of :mod:`exp2025_10_21_chunked_ctc`.
    # (context_frames, ctc_lm_result): 9999 = offline / full context (sorts last).
    context_ctc_lm: List[Tuple[int, object]] = [(9999, baseline_ctc_lm_result)]

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
        ctc_lm_result = _train_ls(
            f"chunked-L{left}-C{center}-R{right}-v2.3{_name_post}",
            chunk_size=center,
            chunk_history_size=left,
            chunk_lookahead_size=right,
            batch_size=bs,
        )
        # Only the causal (R=0, L=0) configs contribute to the context-length table.
        if left == 0 and right == 0 and bs == 75_000:
            context_ctc_lm.append((center, ctc_lm_result))

    _ls_context_length_table(context_ctc_lm)


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
    _offline_ctc_lm = _run_ls_ctc_lm(name, exp, task_spm10k)
    _run_ls_align_stats("base-librispeech", exp, 16)
    return exp, _offline_ctc_lm


def _ls_context_length_table(context_ctc_lm: "List[Tuple[int, object]]") -> None:
    """
    Context-length vs CTC+LM first-pass WER table for the LibriSpeech chunked-CTC sweep.

    ``context_ctc_lm`` is a list of ``(context_frames, ctc_lm_result)`` pairs.
    ``context_frames == 9999`` marks the offline / full-context baseline (sorts last).
    Writes a TSV + JSON under :data:`_LS_CHUNKED_PREFIX`.
    """
    table_rows: List[dict] = []
    for context_frames, ctc_lm_result in context_ctc_lm:
        if ctc_lm_result is None or ctc_lm_result.individual_results is None:
            continue
        for eval_name, score_res in ctc_lm_result.individual_results.items():
            table_rows.append(
                {
                    "context_frames": context_frames,
                    "eval_set": eval_name,
                    "ctc_lm_wer": score_res.main_measure_value,
                }
            )

    table_job = WriteTableDataJob(
        columns=["context_frames", "eval_set", "ctc_lm_wer"],
        rows=table_rows,
        sort_by=["eval_set", "context_frames"],
    )
    tk.register_output(f"{_LS_CHUNKED_PREFIX}/context_length_table.tsv", table_job.out_tsv)
    tk.register_output(f"{_LS_CHUNKED_PREFIX}/context_length_table.json", table_job.out_json)


def _ctc_llm_recog() -> None:
    """
    CTC + LibriSpeech-finetuned LLM recognition on LibriSpeech.

    Five LLMs from Mohammad (Qwen2 0.5B/1.5B/7B, with LLM-native and spm10k vocabs):

    - spm10k-vocab variants (0.5B, 1.5B): standard time-sync CTC+LM,
      same pipeline as the scaling-law LM family.
    - LLM-vocab variants (0.5B, 1.5B, 7B): delayed-fusion CTC+LM
      (every-20-frame fusion schedule, beam size 8).

    CTC model: :data:`_CTC_MODEL_NAME` (same as the LM scaling-law experiments).
    """
    import functools

    from i6_experiments.users.zeyer.experiments.exp2025_11_19_lm_scaling_laws import train_base_asr_models
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import _train_experiments
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
        ctc_recog_recomb_labelwise_prior_auto_scale,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc import (
        model_recog_with_recomb,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_delayed_fusion_v2 import (
        model_recog_with_recomb_delayed_fusion_v2,
        enable_by_interval,
        convert_labels_func,
        spm_space_first_is_word_start,
        spm_label_merge_v2,
        seq_str_postprocess_lower_case,
    )
    from i6_experiments.users.zeyer.external_models.qwen2_finetuned import (
        get_qwen2_0_5b_lm_finetuned_librispeech,
        get_qwen2_0_5b_lm_finetuned_librispeech_spm10k_vocab,
        get_qwen2_1_5b_lm_finetuned_librispeech,
        get_qwen2_1_5b_lm_finetuned_librispeech_spm10k_vocab,
        get_qwen2_7b_lm_finetuned_librispeech,
    )

    # Populates _train_experiments with the CTC baseline (same one _lm_family_ppl_wer recogs).
    train_base_asr_models()
    prefix = get_setup_prefix_for_module(__name__)
    task = get_librispeech_task_raw_v2(vocab="spm10k")
    ctc_model = _train_experiments[_CTC_MODEL_NAME].get_last_fixed_epoch()

    # Standard time-sync CTC+LM for spm10k-vocab LLMs.
    for lm_tag, lm in [
        ("qwen2-0.5b-ls-spm10k", get_qwen2_0_5b_lm_finetuned_librispeech_spm10k_vocab()),
        ("qwen2-1.5b-ls-spm10k", get_qwen2_1_5b_lm_finetuned_librispeech_spm10k_vocab()),
    ]:
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/ctc+lm-v2/{lm_tag}",
            task=task,
            ctc_model=ctc_model,
            lm=lm,
            # The Qwen LM is a raw partial model def; select recog serialization v2 so the
            # partial serializes as a proper import (not repr()) in the first-pass config.
            first_pass_extra_config={"__serialization_version": 2},
        )

    # Delayed-fusion CTC+LM for LLM-vocab models (every-20-frame fusion, beam 8).
    enable_every20 = functools.partial(enable_by_interval, interval=20)
    convert_labels_func_spm = functools.partial(
        convert_labels_func,
        is_am_label_word_start=spm_space_first_is_word_start,
        custom_am_label_merge=spm_label_merge_v2,
        seq_str_postprocess_func=seq_str_postprocess_lower_case,
    )
    for lm_tag, lm in [
        ("qwen2-0.5b-ls", get_qwen2_0_5b_lm_finetuned_librispeech()),
        ("qwen2-1.5b-ls", get_qwen2_1_5b_lm_finetuned_librispeech()),
        ("qwen2-7b-ls", get_qwen2_7b_lm_finetuned_librispeech()),
    ]:
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/ctc+lm-delayed-v2-always-beamSize8/{lm_tag}",
            task=task,
            ctc_model=ctc_model,
            lm=lm,
            lm_rescore_config={
                "default_data_convert_labels_func": convert_labels_func_spm,
                "chunk_size_for_lm_rescoring": 16,
                "max_seqs": 32,
            },
            ctc_only_recog_version=10,
            ctc_only_recog_def=model_recog_with_recomb,
            recog_version=12,
            recog_def=model_recog_with_recomb_delayed_fusion_v2,
            first_pass_extra_config={
                # v2 serialization so the Qwen LM partial + these partial funcs serialize as
                # proper imports (not repr()) in the first-pass config. Set here in the recipe
                # (not the shared builder) to avoid moving any other recipe's recog hashes.
                "__serialization_version": 2,
                "should_convert_labels_now_func": enable_every20,
                "should_fuse_now_func": enable_every20,
                "convert_labels_func": convert_labels_func_spm,
                "max_seqs": 32,
            },
            first_pass_recog_beam_size=8,
        )

    # Larger-beam (32) delayed-fusion retry for the small LLMs, to test whether the
    # first-pass < rescore gap is a beam/pruning artifact (delayed fusion applies the LM
    # only at boundaries, so a narrow beam prunes good hyps before the LM score lands).
    # Keep the batch narrow (first-pass max_seqs 32 -> 8) so LLM memory stays ~constant vs
    # beam 8 (8 seqs x 32 beam == 32 seqs x 8 beam), and request gpu_mem=80 for headroom.
    # lm_rescore_config is left identical to the beam-8 variant above, so the CTC-only search,
    # scale-tuning and rescore jobs are reused by hash; only the first-pass search job is new.
    for lm_tag, lm in [
        ("qwen2-0.5b-ls", get_qwen2_0_5b_lm_finetuned_librispeech()),
        ("qwen2-1.5b-ls", get_qwen2_1_5b_lm_finetuned_librispeech()),
    ]:
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/ctc+lm-delayed-v2-always-beamSize32/{lm_tag}",
            task=task,
            ctc_model=ctc_model,
            lm=lm,
            lm_rescore_config={
                "default_data_convert_labels_func": convert_labels_func_spm,
                "chunk_size_for_lm_rescoring": 16,
                "max_seqs": 32,
            },
            ctc_only_recog_version=10,
            ctc_only_recog_def=model_recog_with_recomb,
            recog_version=12,
            recog_def=model_recog_with_recomb_delayed_fusion_v2,
            first_pass_extra_config={
                "__serialization_version": 2,
                "should_convert_labels_now_func": enable_every20,
                "should_fuse_now_func": enable_every20,
                "convert_labels_func": convert_labels_func_spm,
                "max_seqs": 8,
            },
            first_pass_recog_beam_size=32,
            first_pass_search_rqmt={"gpu_mem": 80},
        )

    # Finer fusion interval (every 10 frames instead of 20) at beam 32: flush a completed word's
    # LM score into the beam sooner (less pruning lag), to test whether it beats the interval-20
    # point. Same memory trade (first-pass max_seqs=8, gpu_mem=80); lm_rescore_config identical so
    # scale-tuning/rescore/ctc-only reuse by hash; only the first-pass search is new.
    enable_every10 = functools.partial(enable_by_interval, interval=10)
    for lm_tag, lm in [
        ("qwen2-0.5b-ls", get_qwen2_0_5b_lm_finetuned_librispeech()),
        ("qwen2-1.5b-ls", get_qwen2_1_5b_lm_finetuned_librispeech()),
    ]:
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/ctc+lm-delayed-v2-always-int10-beamSize32/{lm_tag}",
            task=task,
            ctc_model=ctc_model,
            lm=lm,
            lm_rescore_config={
                "default_data_convert_labels_func": convert_labels_func_spm,
                "chunk_size_for_lm_rescoring": 16,
                "max_seqs": 32,
            },
            ctc_only_recog_version=10,
            ctc_only_recog_def=model_recog_with_recomb,
            recog_version=12,
            recog_def=model_recog_with_recomb_delayed_fusion_v2,
            first_pass_extra_config={
                "__serialization_version": 2,
                "should_convert_labels_now_func": enable_every10,
                "should_fuse_now_func": enable_every10,
                "convert_labels_func": convert_labels_func_spm,
                "max_seqs": 8,
            },
            first_pass_recog_beam_size=32,
            first_pass_search_rqmt={"gpu_mem": 80},
        )


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
    _ctc_lm = _run_ls_ctc_lm(name, exp, task)
    return _ctc_lm


def _run_ls_ctc_lm(name: str, exp, task, aux_ctc_layer: int = 16):
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
    return ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-v2/{lm_name}",
        task=task,
        ctc_model=exp.get_last_fixed_epoch(),
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=lm,
    )
