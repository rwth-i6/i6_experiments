"""
CTC recognition with LM
"""

from typing import Optional, Any, Tuple, Dict
import functools
import numpy as np

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

from sisyphus import Job, Task, tk

from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, ModelWithCheckpoint

from i6_experiments.users.zeyer.recog import recog_model
from i6_experiments.users.zeyer.collect_model_dataset_stats import collect_statistics

from .ctc import Model, ctc_model_def, _batch_size_factor


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
# _lm_name = "trafo-n96-d512-gelu-drop0-b32_1k"
_lms = {
    "n24-d512": "trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k",
    "n96-d512": "trafo-n96-d512-gelu-drop0-b32_1k",
    "n32-d1024": "trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b32_1k",
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
        ]:
            model = get_ctc_with_lm(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            res = recog_model(
                task=task,
                model=model,
                recog_def=model_recog,
                config={
                    "beam_size": beam_size,
                    "recog_version": 8,
                    "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                    # "__trigger_hash_change": 1,
                },
                search_rqmt={"time": 24},
                name=f"{prefix}/recog-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}",
            )
            tk.register_output(
                f"{prefix}/recog-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}-res",
                res.output,
            )

        # Flashlight beam search implementation.
        # Play around with beam size here.
        for prior_scale, lm_scale in [
            (0.0, 1.0),
            # (0.2, 2.0),
            (0.3, 0.5),
        ]:
            model = get_ctc_with_lm(
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
            model = get_ctc_with_lm(
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
            model = get_ctc_with_lm(
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


def _get_lm_model(name: str) -> ModelWithCheckpoint:
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import py as lm_py
    from i6_experiments.users.zeyer.train_v3 import train_models_by_prefix

    global _called_lm_py_once

    if name in _lm_cache_by_name:
        return _lm_cache_by_name[name]

    if not _called_lm_py_once:
        from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

        with disable_register_output():
            lm_py()
        _called_lm_py_once = True

    exp = train_models_by_prefix["lm/" + name]
    model = exp.get_last_fixed_epoch()
    _lm_cache_by_name[name] = model
    return model


def get_ctc_with_lm(
    *,
    ctc_model: ModelWithCheckpoint,
    prior: Optional[tk.Path] = None,
    prior_type: str = "prob",
    prior_scale: Optional[float] = None,
    language_model: ModelWithCheckpoint,
    lm_scale: float,
) -> ModelWithCheckpoint:
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
    if prior_scale:
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


def ctc_model_ext_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    lm = rf.build_from_dict(config.typed_value("_lm_model_def_dict"), vocab_dim=target_dim)
    lm_scale = config.typed_value("lm_scale", None)
    assert isinstance(lm_scale, (int, float))
    model = ctc_model_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)
    model.lm = lm
    model.lm_scale = lm_scale
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
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim

    logits, enc, enc_spatial_dim = model(source, in_spatial_dim=in_spatial_dim)
    assert isinstance(logits, Tensor) and isinstance(enc_spatial_dim, Dim)
    assert logits.feature_dim  # we expect a feature dim
    assert enc_spatial_dim in logits.dims
    log_probs = model.log_probs_wb_from_logits(logits)
    assert isinstance(log_probs, Tensor)
    probs = rf.exp(log_probs)  # the statistics take the average over this, thus prob space, not log prob
    return probs, enc_spatial_dim


def _ctc_model_rescore(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_beam_dim: Dim,
    targets_spatial_dim: Dim,
):
    """RescoreDef API"""
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim

    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert isinstance(logits, Tensor) and isinstance(enc_spatial_dim, Dim)
    assert logits.feature_dim  # we expect a feature dim
    assert enc_spatial_dim in logits.dims
    log_probs = model.log_probs_wb_from_logits(logits)
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
            blank_index=model.blank_idx,
        )
        neg_log_prob_.append(neg_log_prob)
    neg_log_prob, _ = rf.stack(neg_log_prob_, out_dim=targets_beam_dim)
    log_prob_targets_seq = -neg_log_prob
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq


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
