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
from i6_experiments.users.raissi.setups.common.discrimininative_training.common import Criterion
from i6_experiments.users.raissi.setups.common.helpers.align.FSA import (
    create_rasrconfig_for_alignment_fsa,
)
from i6_experiments.users.raissi.setups.common.helpers.network import (
    add_fast_bw_layer_to_returnn_config,
    LogLinearScales,
)

@dataclass(frozen=True, eq=True)
class LFparameters:
    num_classes: int
    num_data_dim: int
    criterion: Criterion

    loss_like_ce: Optional[bool] = False
    frame_rejection_threshold: Optional[float] = 0.000001
    silence_weight: Optional[float] = 1.0
    discard_silence: Optional[bool] = False
    efficient: Optional[bool] = True
    margin: float = 0.0


def augment_for_lfmmi(
    *,
    crp: rasr.CommonRasrParameters,
    denominator_wfst: str,
    log_linear_scales: LogLinearScales,
    returnn_config: returnn.ReturnnConfig,
    lfmmi_params: LFparameters,
    output_layer: str = "output",
    lfmmi_layer_name: str = "output_lfmmi",
    fw_ce_smoothing: float = 0.0,
    fs_ce_smoothing: float = 0.0,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,

):
    automaton_config = create_rasrconfig_for_alignment_fsa(
        crp=crp,
        extra_rasr_config=extra_rasr_config,
        extra_rasr_post_config=extra_rasr_post_config,

    )

    returnn_config.config.pop("chunking", None)
    returnn_config.config["network"][lfmmi_layer_name] = {
        "class": "copy",
        "from": output_layer,
        "loss": lfmmi_params.criterion.value,
        "loss_scale": 1 - fw_ce_smoothing - fs_ce_smoothing,
        "loss_opts": {
            "sprint_opts": {
                "sprintConfigStr": DelayedFormat("--config={}", automaton_config),
                "sprintControlConfig": {"verbose": True},
                "sprintExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
                "usePythonSegmentOrder": True,  # must be true!
                "numInstances": num_rasr_instances,
            },
            "fast_bw_opts": {
                "den_fsa_file": denominator_wfst,
            },
            "frame_rejection_threshold": lfmmi_params.frame_rejection_threshold,
            "efficient": lfmmi_params.efficient,

        },
    }
    if lfmmi_params.loss_like_ce:
        returnn_config.config["network"][lfmmi_layer_name]["loss_like_ce"] = lfmmi_params.loss_like_ce
    if lfmmi_params.discard_silence:
        returnn_config.config["network"][lfmmi_layer_name]["loss_opts"]["discard_silence"] = lfmmi_params.discard_silence
    if lfmmi_params.margin != 0.0:
        returnn_config.config["network"][lfmmi_layer_name]["loss_opts"]["margin"] = lfmmi_params.margin


    if log_linear_scales.label_posterior_scale != 1.:
        returnn_config.config["network"][lfmmi_layer_name]["loss_opts"]["am_scaling_factor"] = log_linear_scales.label_posterior_scale
    if log_linear_scales.tdp_scale != 1.:
        crp.acoustic_model_config.tdp.scale = log_linear_scales.tdp_scale



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
                transition_scale=log_linear_scales.tdp_scale,
                label_posterior_scale=log_linear_scales.label_posterior_scale,
            ),
        )

        returnn_config.config["network"]["output_bw"]["loss_scale"] = fs_ce_smoothing

    return returnn_config
