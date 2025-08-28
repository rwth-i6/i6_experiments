import copy
from dataclasses import dataclass
from enum import Enum
from IPython import embed
from typing import Optional, Union, Tuple

from i6_core import corpus, discriminative_training, rasr, returnn
from sisyphus import Path
from sisyphus.delayed_ops import DelayedFormat

from i6_experiments.common.setups.rasr.config.lm_config import ArpaLmRasrConfig
from i6_experiments.users.raissi.setups.common.decoder.config import SearchParameters
from i6_experiments.users.raissi.setups.common.helpers.align.FSA import (
    create_rasrconfig_for_alignment_fsa,
)
from i6_experiments.users.raissi.setups.common.helpers.network import (
    add_label_prior_layer_to_network,
    add_fast_bw_layer_to_returnn_config,
    LogLinearScales,
)
from i6_experiments.users.raissi.setups.common.global_normalization.common import Criterion, CRF, FactorizationType




def augment_for_local_crf_loss(
    crp: rasr.CommonRasrParameters,
    denominator_wfst: str,
    log_linear_scales: LogLinearScales,
    returnn_config: returnn.ReturnnConfig,
    local_crf_params: CRF,
    output_layer: str = "output",
    local_crf_layer_name: str = "local_crf",
    fw_ce_smoothing: float = 0.0,
    fs_ce_smoothing: float = 0.0,
    scale_offset: float = 0.0,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,

):
    """
    :param crp: common rasr parameters
    :param denominator_wfst: it should be generated offline by WFST composition
    :param log_linear_scales: declare the scales of different elements, label, prior, transition probabilities
    :param returnn_config: of type ReturnnConfig
    :param local_crf_params: CRF datacalss
    :param output_layer: the layer fed into the loss layer where the gradient is computed with respect to the logits
    :param local_crf_layer_name: loss layer
    :param fw_ce_smoothing: regularization with cross-entropy with an alignment
    :param fs_ce_smoothing: regularization with cross-entropy with fullsum loss
    :param scale_offset: if you want all scales not sum up to 1
    :param extra_rasr_config:
    :param extra_rasr_post_config:
    :return:
    """

    assert local_crf_params.criterion == Criterion.LOCAL_CRF
    assert 1 - scale_offset - fw_ce_smoothing - fs_ce_smoothing > 0, "Your scaling is causing reveresed loss"
    returnn_config.config.pop("chunking", None)

    if fw_ce_smoothing > 0:
        loss_info = {"loss": "ce",
                     "loss_scale": fw_ce_smoothing,
                     "target": "classes",
                     "loss_opts": {"focal_loss_factor": 2.0},  # this can be hardcoded by now
                     }
        returnn_config.config["network"][output_layer].update(**loss_info)

    if fs_ce_smoothing > 0:
        returnn_config = add_fast_bw_layer_to_returnn_config(
            crp=crp,
            returnn_config=returnn_config,
            reference_layer=output_layer,
            log_linear_scales=LogLinearScales(
                transition_scale=log_linear_scales.transition_scale,
                label_posterior_scale=log_linear_scales.label_posterior_scale,
            ),
        )
        returnn_config.config["network"]["output_bw"]["loss_scale"] = fs_ce_smoothing

    automaton_config = create_rasrconfig_for_alignment_fsa(
        crp=crp,
        log_file_name="local_crf.log",
        extra_rasr_config=extra_rasr_config,
        extra_rasr_post_config=extra_rasr_post_config,

    )

    prior_layer_name = None
    if local_crf_params.factorization_type == FactorizationType.POST_LAB:
        returnn_config.config["network"], prior_layer_name = add_label_prior_layer_to_network(
            network=returnn_config.config["network"],
            reference_layer=output_layer,
            label_prior_type=local_crf_params.label_prior_type,
            label_prior=local_crf_params.label_prior,
            label_prior_estimation_axes=local_crf_params.label_prior_estimation_axes,

        )
    crf_loss_opts = {
        "numerator_sprint_opts": {
            "sprintConfigStr": DelayedFormat("--config={}", automaton_config),
            "sprintControlConfig": {"verbose": True},
            "sprintExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
            "usePythonSegmentOrder": False,  # must be false with the nn_trainer_exe, otherwise it gets stuck!
            "numInstances": 1,
        },
        "denominator_wfst_file": denominator_wfst,
        "factorization_type": local_crf_params.factorization_type.value,
        "tdp_scale": log_linear_scales.transition_scale,
        "am_scale": log_linear_scales.label_posterior_scale,
        "label_prior_scale": log_linear_scales.lm_scale, #note that this is not frame-level priors but label level
    }
    if prior_layer_name is not None:
        crf_loss_opts["label_prior_layer"] = prior_layer_name

    returnn_config.config["network"][local_crf_layer_name] = {
        "class": "copy",
        "from": output_layer,
        "loss": local_crf_params.criterion.value,
        "loss_opts": crf_loss_opts,
        "loss_scale": 1 - scale_offset - fw_ce_smoothing - fs_ce_smoothing,

    }

    return returnn_config