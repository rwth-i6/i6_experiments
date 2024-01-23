__all__ = ["augment_for_fast_bw", "BwScales"]

import copy
from dataclasses import dataclass
import typing

from sisyphus.delayed_ops import DelayedFormat

from i6_core import rasr, returnn


@dataclass(frozen=True, eq=True)
class BwScales:
    label_posterior_scale: float
    transition_scale: float

    label_prior_scale: typing.Optional[float] = None

    @classmethod
    def default(cls) -> "BwScales":
        return cls(label_posterior_scale=0.3, label_prior_scale=None, transition_scale=0.3)


def get_bw_crp(
    base_crp: rasr.CommonRasrParameters,
    extra_rasr_config: typing.Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: typing.Optional[rasr.RasrConfig] = None,
) -> typing.Tuple[rasr.RasrConfig, rasr.RasrConfig]:
    crp = copy.deepcopy(base_crp)

    crp.acoustic_model_config.tdp.applicator_type = "corrected"
    transition_types = ["*", "silence"]
    if crp.acoustic_model_config.tdp.tying_type == "global-and-nonword":
        transition_types.extend([f"nonword-{i}" for i in [0, 1]])
    for t in transition_types:
        crp.acoustic_model_config.tdp[t].exit = 0.0

    # Create additional Rasr config file for the automaton
    mapping = {
        "corpus": "neural-network-trainer.corpus",
        "lexicon": ["neural-network-trainer.alignment-fsa-exporter.model-combination.lexicon"],
        "acoustic_model": ["neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model"],
    }
    config, post_config = rasr.build_config_from_mapping(crp, mapping)
    post_config["*"].output_channel.file = "fastbw.log"

    # Define action
    config.neural_network_trainer.action = "python-control"

    # neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions = (
        False
    )
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores = (
        False
    )
    # neural_network_trainer.alignment_fsa_exporter
    config.neural_network_trainer.alignment_fsa_exporter.model_combination.acoustic_model.fix_allophone_context_at_word_boundaries = (
        True
    )
    config.neural_network_trainer.alignment_fsa_exporter.model_combination.acoustic_model.transducer_builder_filter_out_invalid_allophones = (
        True
    )

    # additional config
    config._update(extra_rasr_config)
    post_config._update(extra_rasr_post_config)

    return config, post_config


def augment_for_fast_bw(
    *,
    crp: rasr.CommonRasrParameters,
    returnn_config: returnn.ReturnnConfig,
    log_linear_scales: BwScales = None,
    from_output_layer: str = "center-output",
    label_prior: typing.Optional[returnn.CodeWrapper] = None,
    extra_rasr_config: typing.Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: typing.Optional[rasr.RasrConfig] = None,
    layer_prefix: str = "bw-",
    bw_output_layer_name: str = "loss",
    bw_calculation_layer_name: str = "fast-score-calc",
    bw_loss_scale: float = 1.0,
    remove_aux_losses: bool = True,
) -> returnn.ReturnnConfig:
    crp = copy.deepcopy(crp)
    returnn_config = copy.deepcopy(returnn_config)

    if log_linear_scales is None:
        log_linear_scales = BwScales.default()

    network = returnn_config.config["network"]

    if remove_aux_losses:
        to_pop = [k for k in network.keys() if k.startswith("aux")]
        for k in to_pop:
            network.pop(k)
    for layer in network.values():
        layer.pop("target", None)
        layer.pop("loss", None)
        layer.pop("loss_scale", None)
        layer.pop("loss_opts", None)

    if log_linear_scales.label_prior_scale:
        # We are creating a standard hybrid HMM w/ priors in training

        assert label_prior is not None, "Hybrid HMM needs a transcription based prior for fullsum training"

        # Here we are creating a standard hybrid HMM, without prior we have a posterior HMM
        prior_name = f"{layer_prefix}prior-{from_output_layer}"
        network[prior_name] = {
            "class": "constant",
            "dtype": "float32",
            "value": label_prior,
        }

        comb_name = f"{layer_prefix}comb-prior-{from_output_layer}"
        network[comb_name] = {
            "class": "combine",
            "kind": "eval",
            "eval": "am_scale * (safe_log(source(0)) - (safe_log(source(1)) * prior_scale))",
            "eval_locals": {
                "am_scale": log_linear_scales.label_posterior_scale,
                "prior_scale": log_linear_scales.label_prior_scale,
            },
            "from": [from_output_layer, prior_name],
        }

        inputs = comb_name
    else:
        # We are creating a posterior HMM (P-HMM)

        comb_name = f"{layer_prefix}multiply-scale-{from_output_layer}"
        network[comb_name] = {
            "class": "combine",
            "kind": "eval",
            "eval": "am_scale * (safe_log(source(0)))",
            "eval_locals": {"am_scale": log_linear_scales.label_posterior_scale},
            "from": from_output_layer,
        }

        inputs = comb_name

    bw_calc_layer_name = f"{layer_prefix}{bw_calculation_layer_name}"
    network[f"{layer_prefix}{bw_output_layer_name}"] = {
        "class": "copy",
        "from": from_output_layer,
        "loss": "via_layer",
        "loss_opts": {
            "align_layer": bw_calc_layer_name,
            "loss_wrt_to_act_in": "softmax",
        },
        "loss_scale": bw_loss_scale,
    }
    # network["fast_bw"] defined further below...

    config, post_config = get_bw_crp(crp, extra_rasr_config, extra_rasr_post_config)
    rasr_cfg_job = rasr.WriteRasrConfigJob(config, post_config)

    network[bw_calc_layer_name] = {
        "class": "fast_bw",
        "align_target": "sprint",
        "from": inputs,
        "sprint_opts": {
            "sprintExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
            "sprintConfigStr": DelayedFormat("--config={}", rasr_cfg_job.out_config),
            "sprintControlConfig": {"verbose": True},
            "usePythonSegmentOrder": False,
            "numInstances": 1,
        },
        "tdp_scale": log_linear_scales.transition_scale,
    }

    returnn_config.config.pop("chunking", None)
    returnn_config.config.pop("pretrain", None)

    return returnn_config
