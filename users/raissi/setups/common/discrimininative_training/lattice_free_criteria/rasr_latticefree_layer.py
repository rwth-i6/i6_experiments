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
    criterion: Criterion

    loss_like_ce: Optional[bool] = False
    frame_rejection_threshold: Optional[float] = 0.000001
    silence_weight: Optional[float] = 1.0
    discard_silence: Optional[bool] = False
    efficient: Optional[bool] = True
    margin: float = 0.0


def augment_for_lfmmi(
    crp: rasr.CommonRasrParameters,
    denominator_wfst: str,
    log_linear_scales: LogLinearScales,
    returnn_config: returnn.ReturnnConfig,
    lfmmi_params: LFparameters,
    output_layer: str = "output",
    lfmmi_layer_name: str = "lf_mmi",
    fw_ce_smoothing: float = 0.0,
    fs_ce_smoothing: float = 0.0,
    scale_offset: float = 0.0,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,

):
    assert lfmmi_params.criterion == Criterion.LFMMI
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
        scaled_output_layer = returnn_config.config["network"]["fast_bw"]["from"]
    else:
        out_denot = output_layer.split("-")[0]
        scaled_output_layer = ("_").join(["multiply-scale", out_denot])
        returnn_config.config["network"][scaled_output_layer] = {
            "class": "combine",
            "kind": "eval",
            "eval": "am_scale*(safe_log(source(0)))",
            "eval_locals": {"am_scale": log_linear_scales.label_posterior_scale},
            "from": [output_layer],
        }

    automaton_config = create_rasrconfig_for_alignment_fsa(
        crp=crp,
        log_file_name="lfmmi.log",
        extra_rasr_config=extra_rasr_config,
        extra_rasr_post_config=extra_rasr_post_config,

    )
    returnn_config.config["network"][lfmmi_layer_name] = {
        "class": lfmmi_params.criterion.value,
        "from": scaled_output_layer,
        "numerator_sprint_opts": {
            "sprintConfigStr": DelayedFormat("--config={}", automaton_config),
            "sprintControlConfig": {"verbose": True},
            "sprintExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
            "usePythonSegmentOrder": False,  # must be false with the nn_trainer_exe, otherwise it gets stuck!
            "numInstances": 1,
        },
        "denominator_wfst_file": denominator_wfst,
        "tdp_scale": log_linear_scales.transition_scale,
        #"frame_rejection_threshold": lfmmi_params.frame_rejection_threshold,
        #"efficient": lfmmi_params.efficient,
    }
    # if lfmmi_params.loss_like_ce:
    #    returnn_config.config["network"][lfmmi_layer_name]["loss_like_ce"] = lfmmi_params.loss_like_ce
    # if lfmmi_params.discard_silence:
    #    returnn_config.config["network"][lfmmi_layer_name]["loss_opts"]["discard_silence"] = lfmmi_params.discard_silence
    # if log_linear_scales.label_posterior_scale != 1.:
    #    returnn_config.config["network"][lfmmi_layer_name]["loss_opts"]["am_scaling_factor"] = log_linear_scales.label_posterior_scale

    if lfmmi_params.margin != 0.0:
        returnn_config.config["network"][lfmmi_layer_name]["loss_opts"]["margin"] = lfmmi_params.margin

    returnn_config.config["network"]["output_lfmmi"] = {
        "class": "copy",
        "from": output_layer,
        "loss": "via_layer",
        "loss_opts": {"align_layer": lfmmi_layer_name, "loss_wrt_to_act_in": "softmax"},
        "loss_scale": 1 - scale_offset - fw_ce_smoothing - fs_ce_smoothing,
    }

    return returnn_config
