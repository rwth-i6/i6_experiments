"""
CTC recognition with LM
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional, Union, Any, Tuple, Dict
import functools
import numpy as np

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

from sisyphus import Job, tk
from sisyphus.delayed_ops import DelayedBase

from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.datasets.score_results import ScoreResultCollection
from i6_experiments.users.zeyer.datasets.utils.vocab import get_vocab_w_blank_file_from_task
from i6_experiments.users.zeyer.decoding.rescoring import rescore, RescoreDef
from i6_experiments.users.zeyer.decoding.prior_rescoring import Prior

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.recog import recog_model, search_dataset, ctc_alignment_to_label_seq, RecogOutput
from i6_experiments.users.zeyer.collect_model_dataset_stats import collect_statistics
from i6_experiments.users.zeyer.returnn.config import config_dict_update_

from .ctc import Model, ctc_model_def, _batch_size_factor
from .lm import lm_model_def


_ctc_model_name = (
    # last epoch: {"dev-clean": 2.38, "dev-other": 5.67, "test-clean": 2.63, "test-other": 5.93}
    "v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2"
    "-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001"
)

_dep_bound_hash_by_ctc_model_name = {
    "v6-relPosAttDef-noBias-aedLoss"
    "-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
    "-featBN-speedpertV2-spm10k-bpeSample001": "CzG0AHg5psm5",
    "v6-relPosAttDef"
    "-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
    "-featBN-speedpertV2-spm512-bpeSample0005": "YSvtF9CcL6WF",
    "v6-relPosAttDef"
    "-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
    "-featBN-speedpertV2-spm128": "hD9XELdfeFO7",
}

# trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b32_1k*: 34.03  -- still running...
# trafo-n96-d512-gelu-drop0-b32_1k: 34.96
# trafo-n24-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k-ep40: 35.60
# ...
# trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k
_Lm = namedtuple("Lm", ["name", "train_version", "setup"])
# _lm_name = "trafo-n96-d512-gelu-drop0-b32_1k"
_lms = {
    "n24-d512": _Lm("trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k", "v3", "lm"),
    "n96-d512": _Lm("trafo-n96-d512-gelu-drop0-b32_1k", "v3", "lm"),
    "n32-d1024": _Lm("trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b32_1k", "v3", "lm"),
    "n32-d1024-claix2023": _Lm(
        "trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-spm10k", "v4", "lm_claix2023"
    ),
    "n32-d1024-nEp200-claix2023": _Lm(
        "trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-nEp200-spm10k", "v4", "lm_claix2023"
    ),
    "n32-d1024-nEp300-claix2023": _Lm(
        "trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-nEp300-spm10k", "v4", "lm_claix2023"
    ),
    "n32-d1024-nEp400-claix2023": _Lm(
        "trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-nEp400-spm10k", "v4", "lm_claix2023"
    ),
    "n32-d1280-claix2023": _Lm(
        "trafo-n32-d1280-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_15k-spm10k", "v4", "lm_claix2023"
    ),
}


def py():
    """Sis entry point"""
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2, get_vocab_by_str

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = f"{_sis_prefix}/ctc+lm"

    vocab = "spm10k"
    vocab_ = get_vocab_by_str(vocab)
    task = get_librispeech_task_raw_v2(vocab=vocab)
    ctc_model = _get_ctc_model(_ctc_model_name)
    prior = get_ctc_prior_probs(ctc_model, task.train_dataset.copy_train_as_static())
    tk.register_output(f"{prefix}/ctc-prior", prior)

    label_2gram_prior = get_prior_ngram(order=2, vocab=vocab)

    for lm_out_name in ["n24-d512"]:  # lm_out_name, lm_name in _lms.items():
        lm_name = _lms[lm_out_name]
        lm = _get_lm_model(lm_name)

        # Our own beam search implementation.
        for beam_size, prior_scale, lm_scale in [
            # (12, 1.0, 1.0),
            # (12, 0.0, 1.0),
            # (1, 0.0, 1.0),
            # (1, 1.0, 1.0),
            # (12, 0.5, 1.0),
            # (16, 0.5, 1.0),
            # (1, 0.5, 1.0),
            (1, 0.0, 0.0),  # sanity check
            (4, 0.0, 0.0),  # sanity check
            (4, 0.0, 1.0),
            (1, 0.3, 0.5),
            (4, 0.3, 0.5),
            (16, 0.3, 0.5),
            (32, 0.3, 0.5),
            (32, 0.25, 0.5),
        ]:
            model = get_ctc_with_lm_and_framewise_prior(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            res = recog_model(
                task=task,
                model=model,
                recog_def=model_recog,
                config={
                    "beam_size": beam_size,
                    "recog_version": 9,
                    "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                    "__trigger_hash_change": 1,
                },
                search_rqmt={"time": 24},
                name=f"{prefix}/recog-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}",
            )
            tk.register_output(
                f"{prefix}/recog-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}-res",
                res.output,
            )

        # For debugging:
        model = get_ctc_with_lm_and_framewise_prior(
            ctc_model=ctc_model, prior=prior, prior_scale=0.3, language_model=lm, lm_scale=0.5
        )
        res = recog_model(
            task=task,
            model=model,
            recog_def=model_recog,
            config={
                "beam_size": 32,
                "recog_version": 8,
                "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                # "__trigger_hash_change": 1,
                "max_seqs": 1,
            },
            search_rqmt={"time": 24},
            name=f"{prefix}/recog-beam{beam_size}-bsN1-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}",
        )
        tk.register_output(
            f"{prefix}/recog-beam{beam_size}-bsN1-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}-res",
            res.output,
        )

        # Flashlight beam search implementation.
        # Play around with beam size here.
        for prior_scale, lm_scale in [
            (0.0, 1.0),
            # (0.2, 2.0),
            (0.3, 0.5),
        ]:
            model = get_ctc_with_lm_and_framewise_prior(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            for name, opts in [
                # This takes forever (more than 2h for only the first (longest) seq of the corpora),
                # and then at some point runs out of CPU memory (OOM killer kills it).
                # (
                #     "beam1024-beamToken128-cache1024",
                #     {
                #         "n_best": 32,
                #         "beam_size": 1024,
                #         "beam_size_token": 128,
                #         "beam_threshold": 14,
                #         "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                #         "torch_amp": {"dtype": "bfloat16"},
                #         "lm_state_lru_initial_cache_size": 1024,
                #     },
                # ),
                (  # {"dev-clean": 3.51, "dev-other": 5.79, "test-clean": 3.66, "test-other": 6.27}
                    "beam16-beamToken16-cache1024",
                    {
                        "n_best": 16,
                        "beam_size": 16,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 1024,  # -- default
                    },
                ),
                (
                    "beam16-beamToken16-cache2pow16",
                    {
                        "n_best": 16,
                        "beam_size": 16,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 2**16,
                    },
                ),
                (
                    "beam16-beamToken128-cache1024",
                    {
                        "n_best": 16,
                        "beam_size": 16,
                        "beam_size_token": 128,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 1024,  # -- default
                    },
                ),
                (
                    "beam4-beamToken16-cache1024",
                    {
                        "n_best": 16,
                        "beam_size": 4,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 1024,
                    },
                ),
                (
                    "beam4-beamToken16-cache1024-f32",
                    {
                        "n_best": 16,
                        "beam_size": 4,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "lm_state_lru_initial_cache_size": 1024,
                    },
                ),
            ]:
                res = recog_model(
                    task=task,
                    model=model,
                    recog_def=model_recog_flashlight,
                    config=opts,
                    search_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                )
                tk.register_output(
                    f"{prefix}/recog-fl-{name}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}",
                    res.output,
                )

        # Flashlight beam search implementation.
        # Play around with scales.
        for prior_scale, lm_scale in [
            (0.0, 0.0),
            (0.0, 1.0),
            (0.2, 2.0),
            (0.2, 1.0),
            (0.3, 0.5),
            (0.5, 1.0),
            (0.7, 2.0),
        ]:
            model = get_ctc_with_lm_and_framewise_prior(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            res = recog_model(
                task=task,
                model=model,
                recog_def=model_recog_flashlight,
                config={
                    "n_best": 16,
                    "beam_size": 16,
                    "beam_size_token": 128,
                    "beam_threshold": 14,
                    "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                    "torch_amp": {"dtype": "bfloat16"},
                },
                search_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
            )
            tk.register_output(
                f"{prefix}/recog-fl-beamToken128-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}", res.output
            )

        from .recog_ext.ctc_label_sync_espnet import model_recog_espnet

        # ESPnet label-sync beam search implementation.
        # Play around with beam size here.
        for prior_scale, lm_scale in [
            (0.0, 1.0),
            (0.2, 2.0),
            (0.0, 0.0),
            (0.5, 1.0),
        ]:
            model = get_ctc_with_lm_and_framewise_prior(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            for name, opts in [
                (
                    "beam12",
                    {
                        "beam_search_opts": {"beam_size": 12},
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                    },
                ),
                (
                    "beam12-f32",
                    {
                        "beam_search_opts": {"beam_size": 12},
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                    },
                ),
                (
                    "beam32-f32",
                    {
                        "beam_search_opts": {"beam_size": 32},
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                    },
                ),
                (
                    "beam1-f32",
                    {
                        "beam_search_opts": {"beam_size": 1},
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                    },
                ),
            ]:
                res = recog_model(
                    task=task,
                    model=model,
                    recog_def=model_recog_espnet,
                    config=opts,
                    search_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                )
                tk.register_output(
                    f"{prefix}/recog-espnet-{name}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}",
                    res.output,
                )

        # Rescoring.
        from .ctc import model_recog as model_recog_ctc_only, _ctc_model_def_blank_idx
        from i6_experiments.users.zeyer.decoding.lm_rescoring import (
            lm_framewise_prior_rescore,
            lm_labelwise_prior_rescore,
            lm_am_labelwise_prior_rescore,
            lm_am_labelwise_prior_ngram_rescore,
        )
        from i6_experiments.users.zeyer.decoding.prior_rescoring import Prior, PriorRemoveLabelRenormJob
        from i6_experiments.users.zeyer.datasets.utils.vocab import (
            ExtractVocabLabelsJob,
            ExtractVocabSpecialLabelsJob,
            ExtendVocabLabelsByNewLabelJob,
        )

        vocab_file = ExtractVocabLabelsJob(vocab_.get_opts()).out_vocab
        tk.register_output(f"{prefix}/vocab.txt.gz", vocab_file)
        vocab_opts_file = ExtractVocabSpecialLabelsJob(vocab_.get_opts()).out_vocab_special_labels_dict
        tk.register_output(f"{prefix}/vocab_opts.py", vocab_opts_file)
        vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
            vocab=vocab_file, new_label=model_recog_ctc_only.output_blank_label, new_label_idx=_ctc_model_def_blank_idx
        ).out_vocab
        tk.register_output(f"{prefix}/vocab_w_blank.txt.gz", vocab_w_blank_file)
        log_prior_wo_blank = PriorRemoveLabelRenormJob(
            prior_file=prior,
            prior_type="prob",
            vocab=vocab_w_blank_file,
            remove_label=model_recog_ctc_only.output_blank_label,
            out_prior_type="log_prob",
        ).out_prior
        tk.register_output(f"{prefix}/log_prior_wo_blank.txt", log_prior_wo_blank)

        for beam_size, prior_scale, lm_scale in [
            (16, 0.5, 1.0),
            (32, 0.5, 1.0),
            (64, 0.5, 1.0),
            (128, 0.5, 1.0),
            # Those will run out of CPU memory.
            # Unfortunately, those also use mini_task=True.
            # We can reset that in settings.py check_engine_limits,
            # but not sure how to easily do that automatically...
            # (256, 0.5, 1.0),
            # (512, 0.5, 1.0),
            # For reference, getting some scores:
            (128, 0.0, 1.0),
            (128, 0.3, 0.5),
        ]:
            # Note, can use diff priors: framewise using the found alignment.
            #  or label-based. label-based prior can be estimated simply by counting over the transcriptions.
            #    or could also use framewise prior, remove blank, renorm.
            res = recog_model(
                task=task,
                model=ctc_model,
                recog_def=model_recog_ctc_only,
                config={"beam_size": beam_size},
                recog_pre_post_proc_funcs_ext=[
                    functools.partial(
                        lm_framewise_prior_rescore,
                        # framewise standard prior
                        prior=Prior(file=prior, type="prob", vocab=vocab_w_blank_file),
                        prior_scale=prior_scale,
                        lm=lm,
                        lm_scale=lm_scale,
                        vocab=vocab_file,
                        vocab_opts_file=vocab_opts_file,
                    )
                ],
            )
            tk.register_output(
                f"{prefix}/rescore-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}",
                res.output,
            )

    for lm_out_name, lm_name in _lms.items():
        # Tune scales on N-best list with rescoring. (Efficient.)
        lm = _get_lm_model(lm_name)
        beam_size = 128
        scales_results = {}
        for prior_scale in np.linspace(0.0, 1.0, 11):
            for lm_scale in np.linspace(0.0, 1.0, 11):
                res = recog_model(
                    task=task,
                    model=ctc_model,
                    recog_def=model_recog_ctc_only,
                    config={"beam_size": beam_size},
                    recog_pre_post_proc_funcs_ext=[
                        functools.partial(
                            lm_framewise_prior_rescore,
                            # framewise standard prior
                            prior=Prior(file=prior, type="prob", vocab=vocab_w_blank_file),
                            prior_scale=prior_scale,
                            lm=lm,
                            lm_scale=lm_scale,
                            lm_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                            vocab=vocab_file,
                            vocab_opts_file=vocab_opts_file,
                        )
                    ],
                )
                tk.register_output(
                    f"{prefix}/rescore-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}",
                    res.output,
                )
                scales_results[(prior_scale, lm_scale)] = res.output
        _plot_scales(f"rescore-beam{beam_size}-lm_{lm_out_name}", scales_results)

        # TODO note: currently, the task.dev_dataset is only dev-other. maybe change that?
        for fp_beam_size in [128, 64, 32, 16, 4, 1]:
            ctc_recog_framewise_prior_auto_scale(
                prefix=f"{prefix}/opt-beam128-fp{fp_beam_size}-lm_n24-d512-frameprior",
                task=task,
                ctc_model=ctc_model,
                # labelwise_prior=Prior(file=log_prior_wo_blank, type="log_prob", vocab=vocab_file),
                framewise_prior=Prior(file=prior, type="prob", vocab=vocab_w_blank_file),
                lm=_get_lm_model(_lms["n24-d512"]),
                vocab_file=vocab_file,
                vocab_opts_file=vocab_opts_file,
                n_best_list_size=128,
                first_pass_recog_beam_size=fp_beam_size,
            )
            ctc_recog_labelwise_prior_auto_scale(
                prefix=f"{prefix}/opt-beam128-fp{fp_beam_size}-lm_n24-d512-labelprior",
                task=task,
                ctc_model=ctc_model,
                labelwise_prior=Prior(file=log_prior_wo_blank, type="log_prob", vocab=vocab_file),
                lm=_get_lm_model(_lms["n24-d512"]),
                vocab_file=vocab_file,
                vocab_opts_file=vocab_opts_file,
                n_best_list_size=128,
                first_pass_recog_beam_size=fp_beam_size,
            )

        # output/ctc_recog_ext/ctc+lm/opt-beam128-fp128-lm_n32-d1024-labelprior/recog-1stpass-res.txt
        # {"dev-clean": 2.04, "dev-other": 4.06, "test-clean": 2.08, "test-other": 4.36}
        ctc_recog_labelwise_prior_auto_scale(
            prefix=f"{prefix}/opt-beam128-fp128-lm_{lm_out_name}-labelprior",
            task=task,
            ctc_model=ctc_model,
            labelwise_prior=Prior(file=log_prior_wo_blank, type="log_prob", vocab=vocab_file),
            lm=_get_lm_model(_lms[lm_out_name]),
            vocab_file=vocab_file,
            vocab_opts_file=vocab_opts_file,
            n_best_list_size=128,
            first_pass_recog_beam_size=128,
            first_pass_search_rqmt={"gpu_mem": 24 if lm_out_name in {"n96-d512", "n32-d1024"} else 11},
        )

        scales_results = {}
        for lm_scale in np.linspace(0.0, 1.0, 11):
            for prior_scale_rel in np.linspace(0.0, 1.0, 11):
                res = recog_model(
                    task=task,
                    model=ctc_model,
                    recog_def=model_recog_ctc_only,
                    config={"beam_size": beam_size},
                    recog_pre_post_proc_funcs_ext=[
                        functools.partial(
                            lm_framewise_prior_rescore,
                            # framewise standard prior
                            prior=Prior(file=prior, type="prob", vocab=vocab_w_blank_file),
                            prior_scale=lm_scale * prior_scale_rel,
                            lm=lm,
                            lm_scale=lm_scale,
                            lm_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                            vocab=vocab_file,
                            vocab_opts_file=vocab_opts_file,
                        )
                    ],
                )
                tk.register_output(
                    f"{prefix}/rescore-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScaleRel{prior_scale_rel}",
                    res.output,
                )
                scales_results[(prior_scale_rel, lm_scale)] = res.output
        _plot_scales(
            f"rescore-beam{beam_size}-lm_{lm_out_name}-priorScaleRel", scales_results, x_axis_name="prior_scale_rel"
        )

        # Try labelwise prior (lm_labelwise_prior_rescore).
        scales_results = {}
        for lm_scale in np.linspace(0.0, 1.0, 11):
            for prior_scale_rel in np.linspace(0.0, 1.0, 11):
                res = recog_model(
                    task=task,
                    model=ctc_model,
                    recog_def=model_recog_ctc_only,
                    config={"beam_size": beam_size},
                    recog_pre_post_proc_funcs_ext=[
                        functools.partial(
                            lm_labelwise_prior_rescore,
                            # labelwise prior
                            prior=Prior(file=log_prior_wo_blank, type="log_prob", vocab=vocab_file),
                            prior_scale=lm_scale * prior_scale_rel,
                            lm=lm,
                            lm_scale=lm_scale,
                            lm_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                            vocab=vocab_file,
                            vocab_opts_file=vocab_opts_file,
                        )
                    ],
                )
                tk.register_output(
                    f"{prefix}/rescore-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}"
                    f"-labelPrior-priorScaleRel{prior_scale_rel}",
                    res.output,
                )
                scales_results[(prior_scale_rel, lm_scale)] = res.output
        _plot_scales(
            f"rescore-beam{beam_size}-lm_{lm_out_name}-labelPrior-priorScaleRel",
            scales_results,
            x_axis_name="prior_scale_rel",
        )

        # Try rescaling AM scores with full sum (lm_am_labelwise_prior_rescore).
        scales_results = {}
        for lm_scale in np.linspace(0.0, 1.0, 11):
            for prior_scale_rel in np.linspace(0.0, 1.0, 11):
                res = recog_model(
                    task=task,
                    model=ctc_model,
                    recog_def=model_recog_ctc_only,
                    config={"beam_size": beam_size},
                    recog_pre_post_proc_funcs_ext=[
                        functools.partial(
                            lm_am_labelwise_prior_rescore,
                            am=ctc_model,
                            am_rescore_def=_ctc_model_rescore,
                            am_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                            # labelwise prior
                            prior=Prior(file=log_prior_wo_blank, type="log_prob", vocab=vocab_file),
                            prior_scale=lm_scale * prior_scale_rel,
                            lm=lm,
                            lm_scale=lm_scale,
                            lm_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                            vocab=vocab_file,
                            vocab_opts_file=vocab_opts_file,
                        )
                    ],
                )
                tk.register_output(
                    f"{prefix}/rescore-beam{beam_size}-amFSRescore-lm_{lm_out_name}-lmScale{lm_scale}"
                    f"-labelPrior-priorScaleRel{prior_scale_rel}",
                    res.output,
                )
                scales_results[(prior_scale_rel, lm_scale)] = res.output
        _plot_scales(
            f"rescore-beam{beam_size}-amFSRescore-lm_{lm_out_name}-labelPrior-priorScaleRel",
            scales_results,
            x_axis_name="prior_scale_rel",
        )

        # Try using ngram prior (lm_am_labelwise_prior_ngram_rescore).
        scales_results = {}
        for lm_scale in np.linspace(0.0, 1.0, 11):
            for prior_scale_rel in np.linspace(0.0, 1.0, 11):
                res = recog_model(
                    task=task,
                    model=ctc_model,
                    recog_def=model_recog_ctc_only,
                    config={"beam_size": beam_size},
                    recog_pre_post_proc_funcs_ext=[
                        functools.partial(
                            lm_am_labelwise_prior_ngram_rescore,
                            am=ctc_model,
                            am_rescore_def=_ctc_model_rescore,
                            am_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                            # labelwise prior
                            prior_ngram_lm=label_2gram_prior,
                            prior_scale=lm_scale * prior_scale_rel,
                            lm=lm,
                            lm_scale=lm_scale,
                            lm_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                            vocab=vocab_file,
                            vocab_opts_file=vocab_opts_file,
                        )
                    ],
                )
                tk.register_output(
                    f"{prefix}/rescore-beam{beam_size}-amFSRescore-lm_{lm_out_name}-lmScale{lm_scale}"
                    f"-labelPrior2gram-priorScaleRel{prior_scale_rel}",
                    res.output,
                )
                scales_results[(prior_scale_rel, lm_scale)] = res.output
        _plot_scales(
            f"rescore-beam{beam_size}-amFSRescore-lm_{lm_out_name}-labelPrior2gram-priorScaleRel",
            scales_results,
            x_axis_name="prior_scale_rel",
        )


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


_called_ctc_py_once = False
_model_cache_by_name = {}


def _get_ctc_model(name: str, *, use_dependency_boundary: bool = True) -> ModelWithCheckpoint:
    # noinspection PyProtectedMember
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
        py as ctc_py,
        _train_experiments as ctc_train_experiments,
    )

    global _called_ctc_py_once

    if name in _model_cache_by_name:
        return _model_cache_by_name[name]

    if use_dependency_boundary:
        from i6_experiments.common.helpers.dependency_boundary import dependency_boundary

        model = dependency_boundary(
            functools.partial(_get_ctc_model, name=name, use_dependency_boundary=False),
            hash=_dep_bound_hash_by_ctc_model_name.get(name),
        )

    else:
        if not _called_ctc_py_once:
            from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

            with disable_register_output():
                ctc_py()
            _called_ctc_py_once = True

        exp = ctc_train_experiments[name]
        model = exp.get_last_fixed_epoch()

    _model_cache_by_name[name] = model
    return model


_lm_cache_by_name = {}
_called_lm_py_once = False
_called_lm_claix2023_py_once = False


def _get_lm_model(lm: _Lm) -> ModelWithCheckpoint:
    from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output
    from i6_experiments.users.zeyer.train_v3 import train_models_by_prefix as train_v3_models_by_prefix
    from i6_experiments.users.zeyer.train_v4 import train_models_by_prefix as train_v4_models_by_prefix

    global _called_lm_py_once, _called_lm_claix2023_py_once

    if lm.name in _lm_cache_by_name:
        return _lm_cache_by_name[lm.name]

    if lm.setup == "lm":
        if not _called_lm_py_once:
            from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import py as lm_py

            with disable_register_output():
                lm_py()
            _called_lm_py_once = True
    elif lm.setup == "lm_claix2023":
        from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm_claix2023 import py as lm_claix2023_py

        if not _called_lm_claix2023_py_once:
            with disable_register_output():
                lm_claix2023_py()
            _called_lm_claix2023_py_once = True
    else:
        raise ValueError(f"unknown setup {lm.setup!r}")

    train_models_by_prefix = {"v3": train_v3_models_by_prefix, "v4": train_v4_models_by_prefix}[lm.train_version]
    exp = train_models_by_prefix["lm/" + lm.name]
    model = exp.get_last_fixed_epoch()
    _lm_cache_by_name[lm.name] = model
    return model


def get_ctc_with_lm_and_framewise_prior(
    *,
    ctc_model: ModelWithCheckpoint,
    prior: Optional[tk.Path] = None,
    prior_type: str = "prob",
    prior_scale: Optional[Union[float, tk.Variable, DelayedBase]] = None,
    language_model: ModelWithCheckpoint,
    lm_scale: Union[float, tk.Variable],
) -> ModelWithCheckpoint:
    """Combined CTC model with LM and prior"""
    # Keep CTC model config as-is, extend below for prior and LM.
    ctc_model_def = ctc_model.definition
    if isinstance(ctc_model_def, ModelDefWithCfg):
        config: Dict[str, Any] = ctc_model_def.config.copy()
    else:
        config = {}

    # Add prior.
    # Then the CTC Model log_probs_wb_from_logits will include the prior.
    if prior is not None:
        assert prior_scale is not None
    if prior_scale is not None:
        assert prior is not None
        config.update(
            {
                "ctc_prior_type": "static",
                "ctc_prior_scale": prior_scale,
                "static_prior": {"type": prior_type, "file": prior},
            }
        )

    # Add LM.
    # LM has _model_def_dict in config. Put that as _lm_model_def_dict.
    config.update(
        {
            "_lm_model_def_dict": language_model.definition.config["_model_def_dict"],
            "lm_scale": lm_scale,
        }
    )
    config.setdefault("preload_from_files", {})["lm"] = {"prefix": "lm.", "filename": language_model.checkpoint}

    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(model_def=ctc_model_ext_def, config=config),
        checkpoint=ctc_model.checkpoint,
    )


def get_ctc_with_lm_and_labelwise_prior(
    *,
    ctc_model: ModelWithCheckpoint,
    prior: Optional[tk.Path] = None,
    prior_type: str = "prob",
    prior_scale: Optional[Union[float, tk.Variable, DelayedBase]] = None,
    language_model: ModelWithCheckpoint,
    lm_scale: Union[float, tk.Variable],
) -> ModelWithCheckpoint:
    """Combined CTC model with LM and prior"""
    # Keep CTC model config as-is, extend below for prior and LM.
    ctc_model_def_ = ctc_model.definition
    if isinstance(ctc_model_def_, ModelDefWithCfg):
        config: Dict[str, Any] = ctc_model_def_.config.copy()
        ctc_model_def_ = ctc_model_def_.model_def
    else:
        config = {}

    # Add prior.
    # Then the CTC Model log_probs_wb_from_logits will include the prior.
    if prior is not None:
        assert prior_scale is not None
    if prior_scale is not None:
        assert prior is not None
        config.update(
            {
                "labelwise_prior": {"type": prior_type, "file": prior, "scale": prior_scale},
            }
        )

    # Add LM.
    # LM has _model_def_dict in config. Put that as _lm_model_def_dict.
    lm_def = language_model.definition
    assert isinstance(lm_def, ModelDefWithCfg)
    lm_config = lm_def.config.copy()
    if lm_def.model_def is lm_model_def:
        config["_lm_model_def_dict"] = lm_config.pop("_model_def_dict")
    else:
        config["_lm_model_def"] = lm_def.model_def
    config["lm_scale"] = lm_scale
    config["preload_from_files"] = config["preload_from_files"].copy() if config.get("preload_from_files") else {}
    config["preload_from_files"]["lm"] = {"prefix": "lm.", "filename": language_model.checkpoint}
    lm_config_preload_from_files = lm_config.pop("preload_from_files", None)
    if lm_config_preload_from_files:
        config["preload_from_files"].update(lm_config_preload_from_files)
    assert not lm_config

    combined_model_def = ctc_model_ext_def
    if ctc_model_def_ is not ctc_model_def:
        # Also see: denoising_lm_2024.sis_recipe.tts_model.get_asr_with_tts_model_def
        # noinspection PyTypeChecker
        combined_model_def: ModelDef = functools.partial(ctc_model_ext_def, orig_ctc_model_def=ctc_model_def_)
        # Make it a proper ModelDef
        combined_model_def.behavior_version = max(ctc_model_ext_def.behavior_version, ctc_model_def_.behavior_version)
        combined_model_def.backend = ctc_model_def_.backend
        combined_model_def.batch_size_factor = ctc_model_def_.batch_size_factor
        # Need new recog serialization for the partial.
        config["__serialization_version"] = max(2, config.get("__serialization_version", 0))

    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(model_def=combined_model_def, config=config),
        checkpoint=ctc_model.checkpoint,
    )


def get_ctc_with_ngram_lm_and_framewise_prior(
    *,
    ctc_model: ModelWithCheckpoint,
    prior: Optional[tk.Path] = None,
    prior_type: str = "prob",
    prior_scale: Optional[Union[float, tk.Variable, DelayedBase]] = None,
    ngram_language_model: Optional[tk.Path] = None,
    lm_scale: Union[float, tk.Variable],
    ctc_decoder_opts: Dict[str, Any],  # for torchaudio.models.decoder.ctc_decoder, lexicon, nbest, beam_size, etc.
) -> ModelWithCheckpoint:
    """Combined CTC model with LM and prior"""
    # Keep CTC model config as-is, extend below for prior and LM.
    ctc_model_def_ = ctc_model.definition
    if isinstance(ctc_model_def_, ModelDefWithCfg):
        config: Dict[str, Any] = ctc_model_def_.config.copy()
    else:
        config = {}

    # Add prior.
    # Then the CTC Model log_probs_wb_from_logits will include the prior.
    if prior is not None:
        assert prior_scale is not None
    if prior_scale is not None:
        assert prior is not None
        config.update(
            {
                "ctc_prior_type": "static",
                "ctc_prior_scale": prior_scale,
                "static_prior": {"type": prior_type, "file": prior},
            }
        )

    # Add LM.
    if lm_scale:
        assert ngram_language_model is not None
    assert isinstance(ctc_decoder_opts, dict)
    config.update({"lm_scale": lm_scale, "lm_ngram_file": ngram_language_model, "ctc_decoder_opts": ctc_decoder_opts})

    # Use functools.partial to bind orig_ctc_model_def.
    # Need to make it a proper ModelDef.
    # noinspection PyTypeChecker
    ctc_model_ext_def_: ModelDef = functools.partial(ctc_model_ext_def, orig_ctc_model_def=ctc_model_def_)
    # Make it a proper ModelDef
    ctc_model_ext_def_.behavior_version = max(ctc_model_ext_def.behavior_version, ctc_model_def_.behavior_version)
    ctc_model_ext_def_.backend = ctc_model_def_.backend
    ctc_model_ext_def_.batch_size_factor = ctc_model_def_.batch_size_factor
    # Need new recog serialization for the partial.
    config["__serialization_version"] = max(2, config.get("__serialization_version", 0))

    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(model_def=ctc_model_ext_def_, config=config),
        checkpoint=ctc_model.checkpoint,
    )


def ctc_model_ext_def(
    *, epoch: int, in_dim: Dim, target_dim: Dim, orig_ctc_model_def: ModelDef = ctc_model_def
) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    import numpy

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa

    # (framewise) ctc_prior_type / static_prior handled by orig_ctc_model_def.
    model: Model = orig_ctc_model_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)

    lm_scale = config.typed_value("lm_scale", None)
    assert isinstance(lm_scale, (int, float))
    model.lm_scale = lm_scale
    lm_model_def_dict = config.typed_value("_lm_model_def_dict", None)
    lm_ngram_file = config.typed_value("lm_ngram_file", None)
    ctc_decoder_opts = config.typed_value("ctc_decoder_opts", None)
    if lm_model_def_dict:
        model.lm = rf.build_from_dict(lm_model_def_dict, vocab_dim=target_dim)
    elif lm_ngram_file or ctc_decoder_opts is not None:
        assert isinstance(ctc_decoder_opts, dict)  # e.g. lexicon, nbest, beam_size, etc, see below

        from torchaudio.models.decoder import ctc_decoder

        assert target_dim.vocab
        assert model.blank_idx is not None
        wb_target_dim = model.wb_target_dim
        assert wb_target_dim.vocab
        blank_token = wb_target_dim.vocab.labels[model.blank_idx]

        model.decoder = ctc_decoder(
            lm=lm_ngram_file,
            lm_weight=lm_scale,
            tokens=wb_target_dim.vocab.labels,
            blank_token=blank_token,
            sil_token=blank_token,
            **ctc_decoder_opts,
            # lexicon=self.lexicon,
            # nbest=self.n_best,
            # beam_size=self.beam_size,
            # beam_size_token=self.beam_size_token,
            # beam_threshold=self.beam_threshold,
            # sil_score=self.sil_score,
            # word_score=self.word_score,
        )

    else:
        raise ValueError("no LM specified")

    labelwise_prior = config.typed_value("labelwise_prior", None)
    if labelwise_prior:
        assert isinstance(labelwise_prior, dict) and set(labelwise_prior.keys()) == {"type", "file", "scale"}
        v = numpy.loadtxt(labelwise_prior["file"])
        assert v.shape == (target_dim.dimension,), (
            f"invalid shape {v.shape} for labelwise_prior {labelwise_prior['file']!r}, expected dim {target_dim}"
        )
        # The `type` is about what is stored in the file.
        # We always store it in log prob here, so we potentially need to convert it.
        if labelwise_prior["type"] == "log_prob":
            pass  # already log prob
        elif labelwise_prior["type"] == "prob":
            v = numpy.log(v)
        else:
            raise ValueError(f"invalid static_prior type {labelwise_prior['type']!r}")
        v *= labelwise_prior["scale"]  # can already apply now
        model.labelwise_prior = rf.Parameter(
            rf.convert_to_tensor(v, dims=[target_dim], dtype=rf.get_default_float_dtype()),
            auxiliary=True,
            non_critical_for_restore=True,
        )
    else:
        model.labelwise_prior = None

    return model


ctc_model_ext_def: ModelDef[Model]
ctc_model_ext_def.behavior_version = 21
ctc_model_ext_def.backend = "torch"
ctc_model_ext_def.batch_size_factor = _batch_size_factor


# Just an alias, but keep func here to not break hashes.
def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from .recog_ext.ctc import model_recog

    return model_recog(model=model, data=data, data_spatial_dim=data_spatial_dim)


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = True  # our models currently just are batch-size-dependent...


# Just an alias, but keep func here to not break hashes.
def model_recog_flashlight(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from .recog_ext.ctc_flashlight_neural_lm import model_recog_flashlight

    return model_recog_flashlight(model=model, data=data, data_spatial_dim=data_spatial_dim)


# RecogDef API
model_recog_flashlight: RecogDef[Model]
model_recog_flashlight.output_with_beam = True
model_recog_flashlight.output_blank_label = "<blank>"
model_recog_flashlight.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def get_ctc_prior(
    *,
    ctc_model: ModelWithCheckpoint,
    extra_config: Optional[Dict[str, Any]] = None,
    framewise_prior_dataset: DatasetConfig,
    vocab_w_blank_file: Optional[tk.Path] = None,
    task: Optional[Task] = None,
    blank_label: Optional[str] = None,
    blank_idx: Optional[int] = None,
) -> Prior:
    if blank_label is None:
        from .ctc import model_recog as model_recog_ctc_only

        blank_label = model_recog_ctc_only.output_blank_label

    if blank_idx is None:
        from .ctc import _ctc_model_def_blank_idx

        blank_idx = _ctc_model_def_blank_idx

    config = {
        "behavior_version": 24,  # should make it independent from batch size
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # OOM maybe otherwise
        "batch_size": 200_000 * ctc_model.definition.batch_size_factor,
        "max_seqs": 2000,
    }
    if extra_config:
        config = dict_update_deep(config, extra_config)

    prior = get_ctc_prior_probs(ctc_model, framewise_prior_dataset, config=config)

    if vocab_w_blank_file is None:
        assert task
        vocab_w_blank_file = get_vocab_w_blank_file_from_task(task, blank_label=blank_label, blank_idx=blank_idx)

    return Prior(file=prior, type="prob", vocab=vocab_w_blank_file)


def get_ctc_prior_probs(
    ctc_model: ModelWithCheckpoint, dataset: DatasetConfig, config: Optional[Dict[str, Any]] = None
) -> tk.Path:
    """
    :return: CTC prior, in prob space (not log prob)
    """
    # Note: there is also compute_model_softmax_prior_statistics,
    # which assumes a slightly different model API though,
    # and we must call log_probs_wb_from_logits to have it correct in any case.
    return collect_statistics(
        model=ctc_model, dataset=dataset, forward_def=_ctc_model_softmax_prior_returnn_forward, config=config
    ).mean


def get_prior_ngram(*, order: int, vocab: str) -> tk.Path:
    from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str, get_train_corpus_text
    from i6_experiments.users.zeyer.datasets.utils.vocab import ExtractVocabLabelsJob
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextLinesJob
    from i6_core.tools.git import CloneGitRepositoryJob
    from i6_core.lm.kenlm import KenLMplzJob, CompileKenLMJob

    kenlm_repo = CloneGitRepositoryJob(
        "https://github.com/kpu/kenlm", commit="f6c947dc943859e265fabce886232205d0fb2b37"
    ).out_repository.copy()
    kenlm_binary_path = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    # run it locally, and then make sure, and then make sure that necessary deps are installed:
    #   libeigen3-dev
    #   libboost-dev libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev
    kenlm_binary_path.creator.rqmt["engine"] = "short"
    kenlm_binary_path.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    # kenlm_binary_path = tk.Path(
    #     "/work/tools/asr/kenlm/2020-01-17/build/bin", hash_overwrite="LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    # )

    vocab_ = get_vocab_by_str(vocab)
    vocab_opts = vocab_.get_opts()
    vocab_file = ExtractVocabLabelsJob(vocab_opts).out_vocab

    # Get transcriptions text with applied SPM/BPE on it
    txt_with_vocab = ReturnnDatasetToTextLinesJob(
        returnn_dataset={
            "class": "LmDataset",
            "corpus_file": [get_train_corpus_text()],
            "use_cache_manager": True,
            "orth_vocab": vocab_opts,
            "seq_end_symbol": None,  # handled via orth_vocab
            "unknown_symbol": None,  # handled via orth_vocab
        },
        vocab=vocab_opts,
        data_key="data",
    ).out_txt

    kenlm_job = KenLMplzJob(
        text=[txt_with_vocab],
        order=order,
        interpolate_unigrams=True,
        pruning=[int(i >= 2) for i in range(order)],
        vocabulary=vocab_file,
        discount_fallback=(),
        kenlm_binary_folder=kenlm_binary_path,
        mem=12,
        time=4,
    )
    # Run locally, to have the locally installed deps available.
    kenlm_job.rqmt["engine"] = "short"
    prefix = f"{_sis_prefix}/ctc+lm"
    tk.register_output(f"{prefix}/prior_{order}gram.arpa.gz", kenlm_job.out_lm)
    return kenlm_job.out_lm


def _ctc_model_softmax_prior_returnn_forward(
    source: Tensor, /, in_spatial_dim: Dim, model: Model
) -> Tuple[Tensor, Dim]:
    """ForwardDef API"""
    log_probs, _, enc_spatial_dim = model.encode_and_get_ctc_log_probs(source, in_spatial_dim=in_spatial_dim)
    probs = rf.exp(log_probs)  # the statistics take the average over this, thus prob space, not log prob
    return probs, enc_spatial_dim


def ctc_model_rescore(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_beam_dim: Dim,
    targets_spatial_dim: Dim,
) -> Tensor:
    """RescoreDef API"""
    log_probs, _, enc_spatial_dim = model.encode_and_get_ctc_log_probs(data, in_spatial_dim=data_spatial_dim)

    batch_dims = targets.remaining_dims(targets_spatial_dim)

    # Note: Using ctc_loss directly requires quite a lot of memory,
    # as we would broadcast the log_probs over the beam dim.
    # Instead, we do a loop over the beam dim to avoid this.
    neg_log_prob_ = []
    for beam_idx in range(targets_beam_dim.get_dim_value()):
        targets_b = rf.gather(targets, axis=targets_beam_dim, indices=beam_idx)
        targets_b_seq_lens = rf.gather(targets_spatial_dim.dyn_size_ext, axis=targets_beam_dim, indices=beam_idx)
        targets_b_spatial_dim = Dim(targets_b_seq_lens, name=f"{targets_spatial_dim.name}_beam{beam_idx}")
        targets_b, _ = rf.replace_dim(targets_b, in_dim=targets_spatial_dim, out_dim=targets_b_spatial_dim)
        targets_b, _ = rf.slice(targets_b, axis=targets_b_spatial_dim, size=targets_b_spatial_dim)
        # Note: gradient does not matter (not used), thus no need use our ctc_loss_fixed_grad.
        neg_log_prob = rf.ctc_loss(
            logits=log_probs,
            logits_normalized=True,
            targets=targets_b,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=targets_b_spatial_dim,
            blank_index=model.blank_idx,
        )
        neg_log_prob_.append(neg_log_prob)
    neg_log_prob, _ = rf.stack(neg_log_prob_, out_dim=targets_beam_dim)
    log_prob_targets_seq = -neg_log_prob
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq


# just an alias now, but keep here to not break hash
def _ctc_model_rescore(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_beam_dim: Dim,
    targets_spatial_dim: Dim,
) -> Tensor:
    return ctc_model_rescore(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        targets=targets,
        targets_beam_dim=targets_beam_dim,
        targets_spatial_dim=targets_spatial_dim,
    )


# like lm_score
def ctc_best_path_score(
    recog_output: RecogOutput,
    *,
    dataset: DatasetConfig,  # for encoder inputs (e.g. audio)
    model: ModelWithCheckpoint,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param recog_output:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
        We ignore the scores here and just use the text of the hyps.
    :param dataset: the orig data which was used to generate recog_output
    :param model: AED model
    :param vocab: labels (line-based, maybe gzipped)
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param rescore_rqmt:
    """
    return rescore(
        recog_output=recog_output,
        dataset=dataset,
        model=model,
        vocab=vocab,
        vocab_opts_file=vocab_opts_file,
        rescore_def=ctc_best_path_model_rescore_def,
        forward_rqmt=rescore_rqmt,
    )


def ctc_best_path_model_rescore_def(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_beam_dim: Dim,
    targets_spatial_dim: Dim,
) -> Tensor:
    """
    RescoreDef API

    Also see :func:`ctc_model_rescore` above.
    """
    from i6_experiments.users.zeyer.nn_rf.fsa import best_path_ctc

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    data_batch_dims = data.remaining_dims(data_spatial_dim)

    log_probs, _, enc_spatial_dim = model.encode_and_get_ctc_log_probs(data, in_spatial_dim=data_spatial_dim)

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    assert set(batch_dims) == set(data_batch_dims).union({targets_beam_dim})

    _, neg_log_prob = best_path_ctc(
        logits=log_probs,
        logits_normalized=True,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    log_prob_targets_seq = -neg_log_prob
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq


ctc_best_path_model_rescore_def: RescoreDef


def ctc_prior_full_sum_rescore(
    *,
    model: Model,  # TODO doesnt need that. need blank_idx, downsample factor, prior
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_beam_dim: Dim,
    targets_spatial_dim: Dim,
) -> Tensor:
    """RescoreDef API"""
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim

    data  # noqa  # not used here

    enc_spatial_dim = ...  # TODO downsample factor from data_spatial_dim...
    log_probs = ...  # TODO...
    assert isinstance(log_probs, Tensor)

    batch_dims = targets.remaining_dims(targets_spatial_dim)

    # Note: Using ctc_loss directly requires quite a lot of memory,
    # as we would broadcast the log_probs over the beam dim.
    # Instead, we do a loop over the beam dim to avoid this.
    neg_log_prob_ = []
    for beam_idx in range(targets_beam_dim.get_dim_value()):
        targets_b = rf.gather(targets, axis=targets_beam_dim, indices=beam_idx)
        targets_b_seq_lens = rf.gather(targets_spatial_dim.dyn_size_ext, axis=targets_beam_dim, indices=beam_idx)
        targets_b_spatial_dim = Dim(targets_b_seq_lens, name=f"{targets_spatial_dim.name}_beam{beam_idx}")
        targets_b, _ = rf.replace_dim(targets_b, in_dim=targets_spatial_dim, out_dim=targets_b_spatial_dim)
        targets_b, _ = rf.slice(targets_b, axis=targets_b_spatial_dim, size=targets_b_spatial_dim)
        # Note: gradient does not matter (not used), thus no need use our ctc_loss_fixed_grad.
        neg_log_prob = rf.ctc_loss(
            logits=log_probs,
            logits_normalized=True,
            targets=targets_b,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=targets_b_spatial_dim,
            blank_index=blank_idx,
        )
        neg_log_prob_.append(neg_log_prob)
    neg_log_prob, _ = rf.stack(neg_log_prob_, out_dim=targets_beam_dim)
    log_prob_targets_seq = -neg_log_prob
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq


def ctc_recog_framewise_prior_auto_scale(
    *,
    prefix: str,
    task: Task,
    ctc_model: ModelWithCheckpoint,
    framewise_prior: Prior,
    lm: ModelWithCheckpoint,
    vocab_file: tk.Path,
    vocab_opts_file: tk.Path,
    n_best_list_size: int,
    first_pass_recog_beam_size: int,
) -> ScoreResultCollection:
    """
    Recog with ``model_recog_ctc_only`` to get N-best list on ``task.dev_dataset``,
    then calc scores with framewise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog``) with those scales.
    """
    from .ctc import model_recog as model_recog_ctc_only
    from i6_experiments.users.zeyer.decoding.lm_rescoring import (
        lm_framewise_prior_rescore,
        prior_score,
        lm_score,
    )

    # see recog_model, lm_labelwise_prior_rescore
    dataset = task.dev_dataset
    asr_scores = search_dataset(
        dataset=dataset,
        model=ctc_model,
        recog_def=model_recog_ctc_only,
        config={"beam_size": n_best_list_size},
        keep_alignment_frames=True,
        keep_beam=True,
    )
    prior_scores = prior_score(asr_scores, prior=framewise_prior)
    if model_recog_ctc_only.output_blank_label:
        asr_scores = ctc_alignment_to_label_seq(asr_scores, blank_label=model_recog_ctc_only.output_blank_label)
        prior_scores = ctc_alignment_to_label_seq(prior_scores, blank_label=model_recog_ctc_only.output_blank_label)
    lm_scores = lm_score(asr_scores, lm=lm, vocab=vocab_file, vocab_opts_file=vocab_opts_file)

    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        asr_scores = f(asr_scores)
        prior_scores = f(prior_scores)
        lm_scores = f(lm_scores)
        ref = f(ref)

    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob

    opt_scales_job = ScaleTuningJob(
        scores={"am": asr_scores.output, "prior": prior_scores.output, "lm": lm_scores.output},
        ref=ref.output,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    # But prior is still handled as negative in lm_framewise_prior_rescore and 1stpass model_recog below.
    # (The DelayedBase logic on the Sis Variable should handle this.)
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    # Rescore with optimal scales. Like recog_model with lm_framewise_prior_rescore.
    res = recog_model(
        task=task,
        model=ctc_model,
        recog_def=model_recog_ctc_only,
        config={"beam_size": n_best_list_size},
        recog_pre_post_proc_funcs_ext=[
            functools.partial(
                lm_framewise_prior_rescore,
                # framewise standard prior
                prior=framewise_prior,
                prior_scale=prior_scale,
                lm=lm,
                lm_scale=lm_scale,
                lm_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                vocab=vocab_file,
                vocab_opts_file=vocab_opts_file,
            )
        ],
    )
    tk.register_output(f"{prefix}/rescore-res.txt", res.output)

    model = get_ctc_with_lm_and_framewise_prior(
        ctc_model=ctc_model,
        prior=framewise_prior.file,
        prior_type=framewise_prior.type,
        prior_scale=prior_scale,
        language_model=lm,
        lm_scale=lm_scale,
    )
    res = recog_model(
        task=task,
        model=model,
        recog_def=model_recog,
        config={
            "beam_size": first_pass_recog_beam_size,
            "recog_version": 9,
            # Batch size was fitted on our small GPUs (1080) with 11GB for beam size 32.
            # So when the beam size is larger, reduce batch size.
            # (Linear is a bit wrong, because the encoder mem consumption is independent, but anyway...)
            "batch_size": int(5_000 * ctc_model.definition.batch_size_factor * min(32 / first_pass_recog_beam_size, 1)),
        },
        search_rqmt={"time": 24},
        name=f"{prefix}/recog-opt-1stpass",
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", res.output)
    return res


def ctc_recog_labelwise_prior_auto_scale(
    *,
    prefix: str,
    task: Task,
    ctc_model: ModelWithCheckpoint,
    labelwise_prior: Prior,
    lm: ModelWithCheckpoint,
    vocab_file: tk.Path,
    vocab_opts_file: tk.Path,
    n_best_list_size: int,
    first_pass_recog_beam_size: int,
    first_pass_search_rqmt: Optional[Dict[str, int]] = None,
) -> ScoreResultCollection:
    """
    Recog with ``model_recog_ctc_only`` to get N-best list on ``task.dev_dataset``,
    then calc scores with labelwise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog``) with those scales.
    """
    from .ctc import model_recog as model_recog_ctc_only
    from i6_experiments.users.zeyer.decoding.lm_rescoring import (
        lm_labelwise_prior_rescore,
        prior_score,
        lm_score,
    )

    # see recog_model, lm_labelwise_prior_rescore
    dataset = task.dev_dataset
    asr_scores = search_dataset(
        dataset=dataset,
        model=ctc_model,
        recog_def=model_recog_ctc_only,
        config={"beam_size": n_best_list_size},
        keep_beam=True,
    )
    prior_scores = prior_score(asr_scores, prior=labelwise_prior)
    lm_scores = lm_score(asr_scores, lm=lm, vocab=vocab_file, vocab_opts_file=vocab_opts_file)

    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        asr_scores = f(asr_scores)
        prior_scores = f(prior_scores)
        lm_scores = f(lm_scores)
        ref = f(ref)

    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob

    opt_scales_job = ScaleTuningJob(
        scores={"am": asr_scores.output, "prior": prior_scores.output, "lm": lm_scores.output},
        ref=ref.output,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    # But prior is still handled as negative in lm_framewise_prior_rescore and 1stpass model_recog below.
    # (The DelayedBase logic on the Sis Variable should handle this.)
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    # Rescore with optimal scales. Like recog_model with lm_framewise_prior_rescore.
    res = recog_model(
        task=task,
        model=ctc_model,
        recog_def=model_recog_ctc_only,
        config={"beam_size": n_best_list_size},
        recog_pre_post_proc_funcs_ext=[
            functools.partial(
                lm_labelwise_prior_rescore,
                # framewise standard prior
                prior=labelwise_prior,
                prior_scale=prior_scale,
                lm=lm,
                lm_scale=lm_scale,
                lm_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                vocab=vocab_file,
                vocab_opts_file=vocab_opts_file,
            )
        ],
    )
    tk.register_output(f"{prefix}/rescore-res.txt", res.output)

    model = get_ctc_with_lm_and_labelwise_prior(
        ctc_model=ctc_model,
        prior=labelwise_prior.file,
        prior_type=labelwise_prior.type,
        prior_scale=prior_scale,
        language_model=lm,
        lm_scale=lm_scale,
    )
    first_pass_search_rqmt = first_pass_search_rqmt.copy() if first_pass_search_rqmt else {}
    first_pass_search_rqmt.setdefault("time", 24)
    res = recog_model(
        task=task,
        model=model,
        recog_def=model_recog,
        config={
            "beam_size": first_pass_recog_beam_size,
            "recog_version": 9,
            # Batch size was fitted on our small GPUs (1080) with 11GB for beam size 32.
            # So when the beam size is larger, reduce batch size.
            # (Linear is a bit wrong, because the encoder mem consumption is independent, but anyway...)
            "batch_size": int(5_000 * ctc_model.definition.batch_size_factor * min(32 / first_pass_recog_beam_size, 1)),
        },
        search_rqmt=first_pass_search_rqmt,
        name=f"{prefix}/recog-opt-1stpass",
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", res.output)
    return res


def ctc_recog_recomb_labelwise_prior_auto_scale(
    *,
    prefix: str,
    task: Task,
    ctc_model: ModelWithCheckpoint,
    labelwise_prior: Optional[Prior] = None,
    prior_dataset: Optional[DatasetConfig] = None,
    lm: ModelWithCheckpoint,
    lm_rescore_config: Optional[Dict[str, Any]] = None,
    vocab_file: Optional[tk.Path] = None,
    vocab_opts_file: Optional[tk.Path] = None,
    n_best_list_size: int = 64,
    first_pass_recog_beam_size: int = 64,
    first_pass_extra_config: Optional[Dict[str, Any]] = None,
    first_pass_search_rqmt: Optional[Dict[str, int]] = None,
    recomb_type: str = "max",
    extra_config: Optional[Dict[str, Any]] = None,
    ctc_soft_collapse_threshold: Optional[float] = 0.8,
    recog_version: int = 10,
    recog_def: Optional[RecogDef[Model]] = None,
    ctc_only_recog_version: Optional[int] = None,
    ctc_only_recog_def: Optional[RecogDef[Model]] = None,
) -> ScoreResultCollection:
    """
    Recog with ``model_recog_with_recomb`` and recomb enabled to get N-best list on ``task.dev_dataset``,
    then calc scores with framewise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog_with_recomb``) with those scales.
    """
    from i6_experiments.users.zeyer.decoding.lm_rescoring import (
        lm_labelwise_prior_rescore,
        prior_score,
        lm_score,
    )
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from .ctc import _ctc_model_def_blank_idx

    if recog_def is None:
        from .recog_ext.ctc import model_recog_with_recomb as recog_def

    if ctc_only_recog_version is None:
        ctc_only_recog_version = recog_version

    if ctc_only_recog_def is None:
        ctc_only_recog_def = recog_def

    if vocab_file is None:
        from i6_experiments.users.zeyer.datasets.utils.vocab import get_vocab_file_from_task

        vocab_file = get_vocab_file_from_task(task)

    if vocab_opts_file is None:
        from i6_experiments.users.zeyer.datasets.utils.vocab import get_vocab_opts_file_from_task

        vocab_opts_file = get_vocab_opts_file_from_task(task)

    if labelwise_prior is None:
        from i6_experiments.users.zeyer.datasets.utils.vocab import ExtendVocabLabelsByNewLabelJob
        from i6_experiments.users.zeyer.decoding.prior_rescoring import PriorRemoveLabelRenormJob

        prior = get_ctc_prior_probs(
            ctc_model,
            prior_dataset or task.train_dataset.copy_train_as_static(),
            config={
                "behavior_version": 24,
                "batch_size": 200_000 * _batch_size_factor,
                "max_seqs": 2000,
                **(extra_config or {}),
            },
        )
        prior.creator.add_alias(f"{prefix}/prior")
        tk.register_output(f"{prefix}/prior.txt", prior)
        vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
            vocab=vocab_file, new_label=model_recog.output_blank_label, new_label_idx=_ctc_model_def_blank_idx
        ).out_vocab
        tk.register_output(f"{prefix}/vocab_w_blank.txt.gz", vocab_w_blank_file)
        log_prior_wo_blank = PriorRemoveLabelRenormJob(
            prior_file=prior,
            prior_type="prob",
            vocab=vocab_w_blank_file,
            remove_label=model_recog.output_blank_label,
            out_prior_type="log_prob",
        ).out_prior
        tk.register_output(f"{prefix}/log_prior_wo_blank.txt", log_prior_wo_blank)
        labelwise_prior = Prior(file=log_prior_wo_blank, type="log_prob", vocab=vocab_file)

    base_config = {
        "behavior_version": 24,  # should make it independent from batch size
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # OOM maybe otherwise
        "recog_recomb": recomb_type,
    }
    if extra_config:
        base_config = dict_update_deep(base_config, extra_config)
    if ctc_soft_collapse_threshold is not None:
        base_config.update(
            {
                "ctc_soft_collapse_threshold": ctc_soft_collapse_threshold,
                "ctc_soft_collapse_reduce_type": "max_renorm",
            }
        )

    # see recog_model, lm_labelwise_prior_rescore
    dataset = task.dev_dataset
    asr_scores = search_dataset(
        dataset=dataset,
        model=ctc_model,
        recog_def=ctc_only_recog_def,
        config={**base_config, "recog_version": ctc_only_recog_version, "beam_size": n_best_list_size},
        keep_beam=True,
    )
    prior_scores = prior_score(asr_scores, prior=labelwise_prior)
    lm_scores = lm_score(asr_scores, lm=lm, vocab=vocab_file, vocab_opts_file=vocab_opts_file, config=lm_rescore_config)

    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        asr_scores = f(asr_scores)
        prior_scores = f(prior_scores)
        lm_scores = f(lm_scores)
        ref = f(ref)

    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob

    opt_scales_job = ScaleTuningJob(
        scores={"am": asr_scores.output, "prior": prior_scores.output, "lm": lm_scores.output},
        ref=ref.output,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    # But prior is still handled as negative in lm_framewise_prior_rescore and 1stpass model_recog below.
    # (The DelayedBase logic on the Sis Variable should handle this.)
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    # Rescore with optimal scales. Like recog_model with lm_framewise_prior_rescore.
    res = recog_model(
        task=task,
        model=ctc_model,
        recog_def=ctc_only_recog_def,
        config={**base_config, "recog_version": ctc_only_recog_version, "beam_size": n_best_list_size},
        recog_pre_post_proc_funcs_ext=[
            functools.partial(
                lm_labelwise_prior_rescore,
                # framewise standard prior
                prior=labelwise_prior,
                prior_scale=prior_scale,
                lm=lm,
                lm_scale=lm_scale,
                lm_rescore_config=lm_rescore_config,
                lm_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 48},
                vocab=vocab_file,
                vocab_opts_file=vocab_opts_file,
            )
        ],
    )
    tk.register_output(f"{prefix}/rescore-res.txt", res.output)

    model = get_ctc_with_lm_and_labelwise_prior(
        ctc_model=ctc_model,
        prior=labelwise_prior.file,
        prior_type=labelwise_prior.type,
        prior_scale=prior_scale,
        language_model=lm,
        lm_scale=lm_scale,
    )
    first_pass_search_rqmt = first_pass_search_rqmt.copy() if first_pass_search_rqmt else {}
    first_pass_search_rqmt.setdefault("time", 24)
    first_pass_search_rqmt.setdefault("mem", 50)
    res = recog_model(
        task=task,
        model=model,
        recog_def=recog_def,
        config={
            **base_config,
            **(first_pass_extra_config or {}),
            "recog_version": recog_version,
            "beam_size": first_pass_recog_beam_size,
            # Batch size was fitted on our small GPUs (1080) with 11GB for beam size 32.
            # So when the beam size is larger, reduce batch size.
            # (Linear is a bit wrong, because the encoder mem consumption is independent, but anyway...)
            "batch_size": int(
                20_000 * ctc_model.definition.batch_size_factor * min(32 / first_pass_recog_beam_size, 1)
            ),
        },
        search_rqmt=first_pass_search_rqmt,
        name=f"{prefix}/recog-opt-1stpass",
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", res.output)
    return res


def ctc_labelwise_recog_auto_scale(
    *,
    prefix: str,
    task: Task,
    ctc_model: ModelWithCheckpoint,
    labelwise_prior: Optional[Prior] = None,
    prior_dataset: Optional[DatasetConfig] = None,
    lm: ModelWithCheckpoint,
    vocab_file: Optional[tk.Path] = None,
    vocab_opts_file: Optional[tk.Path] = None,
    n_best_list_size: int = 64,
    first_pass_recog_beam_size: int = 64,
    first_pass_search_rqmt: Optional[Dict[str, int]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    ctc_soft_collapse_threshold: Optional[float] = 0.8,
    extra_config_n_best_list: Optional[Dict[str, Any]] = None,
) -> ScoreResultCollection:
    """
    Recog with ``model_recog_label_sync_v2`` (without LM) to get N-best list on ``task.dev_dataset``,
    then calc scores with labelwise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog_label_sync_v2``) with those scales.
    """
    from i6_experiments.users.zeyer.decoding.lm_rescoring import (
        lm_labelwise_prior_rescore,
        prior_score,
        lm_score,
    )
    from .ctc import _ctc_model_def_blank_idx
    from .recog_ext.ctc_label_sync_espnet import model_recog_label_sync_v2
    from i6_core.returnn.search import SearchTakeBestJob
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput
    from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import sclite_score_recog_out_to_ref

    if vocab_file is None:
        from i6_experiments.users.zeyer.datasets.utils.vocab import get_vocab_file_from_task

        vocab_file = get_vocab_file_from_task(task)

    if vocab_opts_file is None:
        from i6_experiments.users.zeyer.datasets.utils.vocab import get_vocab_opts_file_from_task

        vocab_opts_file = get_vocab_opts_file_from_task(task)

    if labelwise_prior is None:
        from i6_experiments.users.zeyer.datasets.utils.vocab import ExtendVocabLabelsByNewLabelJob
        from i6_experiments.users.zeyer.decoding.prior_rescoring import PriorRemoveLabelRenormJob

        prior = get_ctc_prior_probs(
            ctc_model,
            prior_dataset or task.train_dataset.copy_train_as_static(),
            config={
                "behavior_version": 24,
                "batch_size": 200_000 * _batch_size_factor,
                "max_seqs": 2000,
                **(extra_config or {}),
            },
        )
        prior.creator.add_alias(f"{prefix}/prior")
        tk.register_output(f"{prefix}/prior.txt", prior)
        vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
            vocab=vocab_file, new_label=model_recog.output_blank_label, new_label_idx=_ctc_model_def_blank_idx
        ).out_vocab
        tk.register_output(f"{prefix}/vocab_w_blank.txt.gz", vocab_w_blank_file)
        log_prior_wo_blank = PriorRemoveLabelRenormJob(
            prior_file=prior,
            prior_type="prob",
            vocab=vocab_w_blank_file,
            remove_label=model_recog.output_blank_label,
            out_prior_type="log_prob",
        ).out_prior
        tk.register_output(f"{prefix}/log_prior_wo_blank.txt", log_prior_wo_blank)
        labelwise_prior = Prior(file=log_prior_wo_blank, type="log_prob", vocab=vocab_file)

    dataset = task.dev_dataset
    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )
    for f in task.recog_post_proc_funcs:  # BPE to words or so
        ref = f(ref)

    # Note: Still requires lots of memory. E.g. batch size 64, without LM, with ctc_soft_collapse_threshold,
    # takes more than 40GB.
    # Could maybe use forward_auto_split_batch_on_oom when we are sure that the batch size does not matter.
    base_config: Dict[str, Any] = {
        "behavior_version": 24,  # should make it independent from batch size
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # OOM maybe otherwise
    }
    if ctc_soft_collapse_threshold is not None:
        base_config.update(
            {
                "ctc_soft_collapse_threshold": ctc_soft_collapse_threshold,
                "ctc_soft_collapse_reduce_type": "max_renorm",
            }
        )
    if extra_config:
        config_dict_update_(base_config, extra_config)
    config_n_best = {**base_config, "beam_size": n_best_list_size}
    if extra_config_n_best_list:
        config_dict_update_(config_n_best, extra_config_n_best_list)

    # see recog_model, lm_labelwise_prior_rescore
    asr_scores = search_dataset(
        dataset=dataset,
        model=ctc_model,
        recog_def=model_recog_label_sync_v2,
        config=config_n_best,
        keep_beam=True,
    )
    asr_scores_labels = asr_scores
    for f in task.recog_post_proc_funcs:  # BPE to words or so
        asr_scores = f(asr_scores)

    # Calc WER as a sanity check.
    first_score = sclite_score_recog_out_to_ref(
        recog_output=RecogOutput(output=SearchTakeBestJob(asr_scores.output, output_gzip=True).out_best_search_results),
        ref=ref,
        corpus_name=dataset.get_main_name(),
    )
    tk.register_output(f"{prefix}/without-lm-{first_score.dataset_name}.txt", first_score.main_measure_value)

    prior_scores = prior_score(asr_scores_labels, prior=labelwise_prior)
    lm_scores = lm_score(asr_scores_labels, lm=lm, vocab=vocab_file, vocab_opts_file=vocab_opts_file)

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        prior_scores = f(prior_scores)
        lm_scores = f(lm_scores)

    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob

    opt_scales_job = ScaleTuningJob(
        scores={"am": asr_scores.output, "prior": prior_scores.output, "lm": lm_scores.output},
        ref=ref.output,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    # But prior is still handled as negative in lm_framewise_prior_rescore and 1stpass model_recog below.
    # (The DelayedBase logic on the Sis Variable should handle this.)
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    # Rescore with optimal scales. Like recog_model with lm_framewise_prior_rescore.
    res = recog_model(
        task=task,
        model=ctc_model,
        recog_def=model_recog_label_sync_v2,
        config=config_n_best,
        recog_pre_post_proc_funcs_ext=[
            functools.partial(
                lm_labelwise_prior_rescore,
                # framewise standard prior
                prior=labelwise_prior,
                prior_scale=prior_scale,
                lm=lm,
                lm_scale=lm_scale,
                lm_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                vocab=vocab_file,
                vocab_opts_file=vocab_opts_file,
            )
        ],
    )
    tk.register_output(f"{prefix}/rescore-res.txt", res.output)

    model = get_ctc_with_lm_and_labelwise_prior(
        ctc_model=ctc_model,
        prior=labelwise_prior.file,
        prior_type=labelwise_prior.type,
        prior_scale=prior_scale,
        language_model=lm,
        lm_scale=lm_scale,
    )
    first_pass_search_rqmt = first_pass_search_rqmt.copy() if first_pass_search_rqmt else {}
    first_pass_search_rqmt.setdefault("time", 24)
    res = recog_model(
        task=task,
        model=model,
        recog_def=model_recog_label_sync_v2,
        config={
            **base_config,
            "beam_size": first_pass_recog_beam_size,
            # TODO better batch size?
            # Batch size was fitted on our small GPUs (1080) with 11GB for beam size 32.
            # So when the beam size is larger, reduce batch size.
            # (Linear is a bit wrong, because the encoder mem consumption is independent, but anyway...)
            "batch_size": int(
                20_000 * ctc_model.definition.batch_size_factor * min(32 / first_pass_recog_beam_size, 1)
            ),
        },
        search_rqmt=first_pass_search_rqmt,
        name=f"{prefix}/recog-opt-1stpass",
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", res.output)
    return res


def _plot_scales(name: str, results: Dict[Tuple[float, float], tk.Path], x_axis_name: str = "prior_scale"):
    prefix = f"{_sis_prefix}/ctc+lm"
    plot_fn = PlotResults2DJob(x_axis_name=x_axis_name, y_axis_name="lm_scale", results=results).out_plot
    tk.register_output(f"{prefix}/{name}-plot-scales.pdf", plot_fn)


class PlotResults2DJob(Job):
    """
    Plot results
    """

    def __init__(self, *, x_axis_name: str, y_axis_name: str, results: Dict[Tuple[float, float], tk.Path]):
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name
        self.results = results

        self.out_plot = self.output_path("out-plot.pdf")

    def tasks(self):
        from sisyphus import Task

        yield Task("run", mini_task=True)

    def run(self):
        from ast import literal_eval
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        xs = sorted(set(x for x, _ in self.results.keys()))
        ys = sorted(set(y for _, y in self.results.keys()))
        results = {k: literal_eval(open(v).read()) for k, v in self.results.items()}
        first_res = results[next(iter(results.keys()))]
        assert isinstance(first_res, dict)

        plt.figure(figsize=(8, 8 * len(first_res)))

        for key_idx, key in enumerate(first_res.keys()):
            zs = np.zeros((len(ys), len(xs)))
            for y_idx, y in enumerate(ys):
                for x_idx, x in enumerate(xs):
                    zs[y_idx, x_idx] = results[(x, y)][key]

            best = np.min(zs.flatten())
            worst_limit = best * 1.3

            ax = plt.subplot(len(first_res), 1, 1 + key_idx)
            plt.contourf(xs, ys, zs, levels=np.geomspace(best, worst_limit, 30))

            ax.set_title(f"{key}")
            ax.set_ylabel(self.y_axis_name)
            ax.set_xlabel(self.x_axis_name)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

            cbar = plt.colorbar()
            cbar.set_label("WER [%]")

        plt.savefig(self.out_plot.get_path())
