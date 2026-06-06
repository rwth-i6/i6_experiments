"""
PPL-WER relation study on LibriSpeech.

Gathers (word-level PPL, WER) per eval set across the LM scaling-law family
and plots one log-log scatter per eval set.

Reuses the LM family and CTC+LM WERs from
:mod:`i6_experiments.users.zeyer.experiments.exp2025_11_19_lm_scaling_laws`.
The only new compute here is per-LM PPL on each eval set
via :func:`...decoding.perplexity.get_lm_perplexities_for_task_evals_v2`.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from sisyphus import tk

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.plots.scaling_laws import ScalingLawPlotJob
from i6_experiments.users.zeyer.decoding.perplexity import get_lm_perplexities_for_task_evals_v2


__setup_root_prefix__ = "exp2026_06_06_ppl_wer_relation"

# Same baseline used by exp2025_11_19_lm_scaling_laws.
_CTC_MODEL_NAME = "L16-D1024-spm10k-auxAED-b100k"


def py():
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_claix2023 import recog_ext_with_lm
    from i6_experiments.users.zeyer.train_v4 import train_models_by_prefix as train_v4_models
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
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
            wer_var = tk.Variable(
                path=score_res.main_measure_value.path, creator=score_res.main_measure_value.creator
            )
            ppl_var = word_ppl[eval_name]
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
