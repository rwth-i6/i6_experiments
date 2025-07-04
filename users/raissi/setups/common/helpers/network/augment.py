import copy
from dataclasses import dataclass
from typing import Any, Iterable, Dict, List, Optional, Tuple, Union

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhonemeStateClasses,
    PhoneticContext,
    RasrStateTying,
)

from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import PriorType

from i6_experiments.users.raissi.setups.common.helpers.network.frame_rate import FrameRateReductionRatioinfo
from i6_experiments.users.raissi.setups.common.helpers.align.FSA import (
    correct_rasr_FSA_bug,
    create_rasrconfig_for_alignment_fsa,
)

DEFAULT_INIT = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)"


@dataclass(frozen=True, eq=True)
class LogLinearScales:
    label_posterior_scale: float
    transition_scale: float
    context_label_posterior_scale: float = 1.0
    label_prior_scale: Optional[float] = None
    lm_scale: Optional[float] = None

    @classmethod
    def default(cls) -> "LogLinearScales":
        return cls(
            label_posterior_scale=0.3, transition_scale=0.3, label_prior_scale=None, context_label_posterior_scale=1.0
        )


@dataclass(frozen=True, eq=True)
class LossScales:
    center_scale: int = 1.0
    right_scale: int = 1.0
    left_scale: int = 1.0

    def get_scale(self, label_name: str):
        if "center" in label_name:
            return self.center_scale
        elif "right" in label_name:
            return self.right_scale
        elif "left" in label_name:
            return self.left_scale
        else:
            raise NotImplemented("Not recognized label name for output loss scale")


Layer = Dict[str, Any]
Network = Dict[str, Layer]


def add_mlp(
    network: Network,
    layer_name: str,
    size: int,
    *,
    source_layer: Union[str, List[str]] = "encoder-output",
    prefix: str = "",
    l2: Optional[float] = None,
    init: str = DEFAULT_INIT,
    n_layers: int = 2,
) -> str:

    assert n_layers > 0, "set a number of layers > 0"

    for i in range(1, n_layers + 1):
        l_name = f"{prefix}linear{i}-{layer_name}"
        network[l_name] = {
            "class": "linear",
            "activation": "relu",
            "from": source_layer,
            "n_out": size,
            "forward_weights_init": init,
        }
        if l2 is not None:
            network[l_name]["L2"] = l2
        source_layer = l_name

    return l_name


def get_embedding_layer(source: Union[str, List[str]], dim: int, l2=0.01):
    return {
        "with_bias": False,
        "L2": l2,
        "class": "linear",
        "activation": None,
        "from": source,
        "n_out": dim,
    }


def pop_phoneme_state_classes(
    label_info: LabelInfo,
    network: Network,
    labeling_input: str,
    remaining_classes: int,
    prefix: str = "",
) -> Tuple[Network, str, int]:
    if label_info.phoneme_state_classes == PhonemeStateClasses.boundary:
        class_layer_name = f"{prefix}boundaryClass"
        labeling_output = f"{prefix}popBoundary"

        # continues below
    elif label_info.phoneme_state_classes == PhonemeStateClasses.word_end:
        class_layer_name = f"{prefix}wordEndClass"
        labeling_output = f"{prefix}popWordEnd"

        # continues below
    elif label_info.phoneme_state_classes == PhonemeStateClasses.none:
        rem_dim = remaining_classes
        labeling_output = "data:classes"

        return network, labeling_output, rem_dim
    else:
        raise NotImplemented(f"unknown phoneme state class {label_info.phoneme_state_classes}")

    factor = label_info.phoneme_state_classes.factor()
    network[class_layer_name] = {
        "class": "eval",
        "from": labeling_input,
        "eval": f"tf.math.floormod(source(0), {factor})",
        "out_type": {"dim": factor, "dtype": "int32", "sparse": True},
    }
    rem_dim = remaining_classes // factor
    network[labeling_output] = {
        "class": "eval",
        "from": labeling_input,
        "eval": f"tf.math.floordiv(source(0), {factor})",
        "out_type": {"dim": rem_dim, "dtype": "int32", "sparse": True},
    }

    return network, labeling_output, rem_dim


def augment_net_with_label_pops(
    network: Network,
    label_info: LabelInfo,
    frame_rate_reduction_ratio_info: FrameRateReductionRatioinfo,
    prefix: str = "",
    labeling_input: str = "data:classes",
) -> Network:
    assert label_info.state_tying in [RasrStateTying.diphone, RasrStateTying.triphone]

    remaining_label_dim = label_info.get_n_of_dense_classes()

    network = copy.deepcopy(network)

    if frame_rate_reduction_ratio_info.factor > 1:
        # This layer sets the time step ratio between the input and the output of the NN.

        frr_factors = (
            [frame_rate_reduction_ratio_info.factor]
            if isinstance(frame_rate_reduction_ratio_info.factor, int)
            else frame_rate_reduction_ratio_info.factor
        )
        t_tag = f"{frame_rate_reduction_ratio_info.time_tag_name}"
        for factor in frr_factors:
            t_tag += f".ceildiv_right({factor})"

        network[f"{prefix}classes_"] = {
            "class": "reinterpret_data",
            "set_dim_tags": {"T": returnn.CodeWrapper(t_tag)},
            "from": labeling_input,
        }
        labeling_input = f"{prefix}classes_"

    if label_info.state_tying == RasrStateTying.triphone:
        network[f"{prefix}futureLabel"] = {
            "class": "eval",
            "from": labeling_input,
            "eval": f"tf.math.floormod(source(0), {label_info.n_contexts})",
            "register_as_extern_data": f"{prefix}futureLabel",
            "out_type": {"dim": label_info.n_contexts, "dtype": "int32", "sparse": True},
        }
        remaining_label_dim //= label_info.n_contexts
        network[f"{prefix}popFutureLabel"] = {
            "class": "eval",
            "from": labeling_input,
            "eval": f"tf.math.floordiv(source(0), {label_info.n_contexts})",
            "out_type": {"dim": remaining_label_dim, "dtype": "int32", "sparse": True},
        }
        labeling_input = f"{prefix}popFutureLabel"

    network[f"{prefix}pastLabel"] = {
        "class": "eval",
        "from": labeling_input,
        "eval": f"tf.math.floormod(source(0), {label_info.n_contexts})",
        "register_as_extern_data": f"{prefix}pastLabel",
        "out_type": {"dim": label_info.n_contexts, "dtype": "int32", "sparse": True},
    }

    remaining_label_dim //= label_info.n_contexts
    assert remaining_label_dim == label_info.get_n_state_classes()

    # popPastLabel in disguise, the label order makes it so that this is directly the center state
    network[f"{prefix}centerState"] = {
        "class": "eval",
        "from": labeling_input,
        "eval": f"tf.math.floordiv(source(0), {label_info.n_contexts})",
        "register_as_extern_data": f"{prefix}centerState",
        "out_type": {"dim": remaining_label_dim, "dtype": "int32", "sparse": True},
    }
    labeling_input = f"{prefix}centerState"

    network, labeling_input, remaining_label_dim = pop_phoneme_state_classes(
        label_info,
        network,
        labeling_input,
        remaining_label_dim,
        prefix=prefix,
    )

    network[f"{prefix}stateId"] = {
        "class": "eval",
        "from": labeling_input,
        "eval": f"tf.math.floormod(source(0), {label_info.n_states_per_phone})",
        "out_type": {
            "dim": label_info.n_states_per_phone,
            "dtype": "int32",
            "sparse": True,
        },
    }

    remaining_label_dim //= label_info.n_states_per_phone
    assert remaining_label_dim == label_info.n_contexts

    network[f"{prefix}centerPhoneme"] = {
        "class": "eval",
        "from": labeling_input,
        "eval": f"tf.math.floordiv(source(0), {label_info.n_states_per_phone})",
        "out_type": {
            "dim": remaining_label_dim,
            "dtype": "int32",
            "sparse": True,
        },
    }

    if frame_rate_reduction_ratio_info.single_state_alignment:
        network[f"{prefix}singleStateCenter"] = {
            "class": "eval",
            "from": [f"{prefix}centerPhoneme", f"{prefix}wordEndClass"],
            "eval": f"(source(0)*{label_info.phoneme_state_classes.factor()})+source(1)",
            "out_type": {
                "dim": label_info.get_n_single_state_classes(),
                "dtype": "int32",
                "sparse": True,
            },
            "register_as_extern_data": f"singleStateCenter",
        }

    return network


def augment_net_with_monophone_outputs(
    shared_network: Network,
    encoder_output_len: int,
    label_info: LabelInfo,
    frame_rate_reduction_ratio_info: FrameRateReductionRatioinfo,
    *,
    add_mlps=True,
    use_multi_task=True,
    final_ctx_type: Optional[PhoneticContext] = None,
    focal_loss_factor=2.0,
    label_smoothing=0.0,
    l2=None,
    encoder_output_layer: str = "encoder-output",
    prefix: str = "",
    loss_scale=1.0,
    shared_delta_encoder=False,
    weights_init: str = DEFAULT_INIT,
) -> Network:
    assert (
        encoder_output_layer in shared_network
    ), f"net needs output layer '{encoder_output_layer}' layer to be predefined"
    assert not add_mlps or final_ctx_type is not None

    network = copy.copy(shared_network)
    center_target, center_dim = (
        ("singleStateCenter", label_info.get_n_single_state_classes())
        if frame_rate_reduction_ratio_info.single_state_alignment
        else ("centerState", label_info.get_n_state_classes())
    )

    loss_opts = {}
    if focal_loss_factor > 0.0:
        loss_opts["focal_loss_factor"] = focal_loss_factor
    if label_smoothing > 0.0:
        loss_opts["label_smoothing"] = label_smoothing

    if add_mlps:
        if final_ctx_type == PhoneticContext.triphone_symmetric:
            tri_out = encoder_output_len + (2 * label_info.ph_emb_size)
            tri_mlp = add_mlp(
                network,
                "triphone",
                tri_out,
                prefix=prefix,
                source_layer=encoder_output_layer,
                l2=l2,
                init=weights_init,
            )

            network[f"{prefix}center-output"] = {
                "class": "softmax",
                "from": tri_mlp,
                "target": center_target,
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }

            if use_multi_task:
                context_mlp = add_mlp(
                    network,
                    "contexts",
                    encoder_output_len,
                    prefix=prefix,
                    source_layer=encoder_output_layer,
                    l2=l2,
                    init=weights_init,
                )
                network[f"{prefix}right-output"] = {
                    "class": "softmax",
                    "from": context_mlp,
                    "target": "futureLabel",
                    "n_out": label_info.n_contexts,
                    "loss": "ce",
                    "loss_opts": copy.copy(loss_opts),
                }
                network[f"{prefix}left-output"] = {
                    "class": "softmax",
                    "from": context_mlp,
                    "target": "pastLabel",
                    "n_out": label_info.n_contexts,
                    "loss": "ce",
                    "loss_opts": copy.copy(loss_opts),
                }
                network[f"{prefix}center-output"]["target"] = center_target
                network[f"{prefix}center-output"]["n_out"] = label_info.get_n_state_classes()

        elif final_ctx_type == PhoneticContext.triphone_forward:
            di_mlp = add_mlp(
                network,
                "diphone",
                encoder_output_len + label_info.ph_emb_size,
                prefix=prefix,
                source_layer=encoder_output_layer,
                l2=l2,
                init=weights_init,
            )

            network[f"{prefix}center-output"] = {
                "class": "softmax",
                "from": di_mlp,
                "target": center_target,
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }

            if use_multi_task:
                tri_out = encoder_output_len + label_info.ph_emb_size + label_info.st_emb_size
                left_ctx_mlp = add_mlp(
                    network,
                    "leftContext",
                    encoder_output_len,
                    prefix=prefix,
                    source_layer=encoder_output_layer,
                    l2=l2,
                    init=weights_init,
                )
                tri_mlp = add_mlp(
                    network,
                    "triphone",
                    tri_out,
                    prefix=prefix,
                    source_layer=encoder_output_layer,
                    l2=l2,
                    init=weights_init,
                )

                network[f"{prefix}left-output"] = {
                    "class": "softmax",
                    "from": left_ctx_mlp,
                    "target": "pastLabel",
                    "n_out": label_info.n_contexts,
                    "loss": "ce",
                    "loss_opts": copy.copy(loss_opts),
                }
                network[f"{prefix}right-output"] = {
                    "class": "softmax",
                    "from": tri_mlp,
                    "target": "futureLabel",
                    "n_out": label_info.n_contexts,
                    "loss": "ce",
                    "loss_opts": copy.copy(loss_opts),
                }
                network[f"{prefix}center-output"]["target"] = center_target
                network[f"{prefix}center-output"]["n_out"] = center_dim

        elif final_ctx_type == PhoneticContext.triphone_backward:
            assert use_multi_task, "it is not possible to have a monophone backward without multitask"

            center_mlp = add_mlp(
                network,
                center_target,
                encoder_output_len,
                prefix=prefix,
                source_layer=encoder_output_layer,
                l2=l2,
                init=weights_init,
            )
            di_mlp = add_mlp(
                network,
                "diphone",
                1030,
                prefix=prefix,
                source_layer=encoder_output_layer,
                l2=l2,
                init=weights_init,
            )
            tri_mlp = add_mlp(
                network,
                "triphone",
                1040,
                prefix=prefix,
                source_layer=encoder_output_layer,
                l2=l2,
                init=weights_init,
            )

            network[f"{prefix}center-output"] = {
                "class": "softmax",
                "from": center_mlp,
                "target": center_target,
                "n_out": center_dim,
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }
            network[f"{prefix}left-output"] = {
                "class": "softmax",
                "from": tri_mlp,
                "target": "pastLabel",
                "n_out": label_info.n_contexts,
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }
            network[f"{prefix}right-output"] = {
                "class": "softmax",
                "from": di_mlp,
                "target": "futureLabel",
                "n_out": label_info.n_contexts,
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }

        elif final_ctx_type == PhoneticContext.tri_state_transition:
            raise "this is not tested yet"

            delta_blstm_n = f"{prefix}deltaEncoder-output"
            di_out = encoder_output_len + ph_emb_size

            if shared_delta_encoder:
                add_delta_blstm_(
                    network,
                    name=delta_blstm_n,
                    l2=l2,
                    source_layer=encoder_output_layer,
                )
                di_mlp = add_mlp(
                    network,
                    "diphone",
                    di_out,
                    source_layer=delta_blstm_n,
                    l2=l2,
                    init=weights_init,
                )
            else:
                add_delta_blstm_(network, name=delta_blstm_n, l2=l2, prefix=prefix)
                di_mlp = add_mlp(
                    network,
                    "diphone",
                    di_out,
                    l2=l2,
                    prefix=prefix,
                    init=weights_init,
                )

            network[f"{prefix}center-output"] = {
                "class": "softmax",
                "from": di_mlp,
                "target": center_target,
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }

            if use_multi_task:
                tri_out = encoder_output_len + ph_emb_size + st_emb_size
                left_ctx_mlp = add_mlp(
                    network,
                    "leftContext",
                    encoder_output_len,
                    prefix=prefix,
                    source_layer=encoder_output_layer,
                    l2=l2,
                    init=weights_init,
                )

                if shared_delta_encoder:
                    tri_mlp = add_mlp(
                        network,
                        "triphone",
                        tri_out,
                        prefix=prefix,
                        source_layer=delta_blstm_n,
                        l2=l2,
                        init=weights_init,
                    )
                else:
                    tri_mlp = add_mlp(
                        network,
                        "triphone",
                        tri_out,
                        prefix=prefix,
                        source_layer=delta_blstm_n,
                        l2=l2,
                        init=weights_init,
                    )

                network[f"{prefix}left-output"] = {
                    "class": "softmax",
                    "from": left_ctx_mlp,
                    "target": "pastLabel",
                    "n_out": label_info.n_contexts,
                    "loss": "ce",
                    "loss_opts": copy.copy(loss_opts),
                }
                network[f"{prefix}right-output"] = {
                    "class": "softmax",
                    "from": tri_mlp,
                    "target": "futureLabel",
                    "n_out": label_info.n_contexts,
                    "loss": "ce",
                    "loss_opts": copy.copy(loss_opts),
                }
                network[f"{prefix}center-output"]["target"] = center_target
                network[f"{prefix}center-output"]["n_out"] = center_dim
    else:
        network[f"{prefix}center-output"] = {
            "class": "softmax",
            "from": encoder_output_layer,
            "target": center_target,
            "loss": "ce",
            "loss_opts": copy.copy(loss_opts),
        }

        if use_multi_task:
            network[f"{prefix}left-output"] = {
                "class": "softmax",
                "from": encoder_output_layer,
                "target": "pastLabel",
                "n_out": label_info.n_contexts,
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }
            network[f"{prefix}right-output"] = {
                "class": "softmax",
                "from": encoder_output_layer,
                "target": "futureLabel",
                "n_out": label_info.n_contexts,
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }
            network[f"{prefix}center-output"]["target"] = center_target
            network[f"{prefix}center-output"]["n_out"] = center_dim

    if loss_scale != 1.0:
        network[f"{prefix}center-output"]["loss_scale"] = loss_scale
        if use_multi_task:
            network[f"{prefix}left-output"]["loss_scale"] = loss_scale
            network[f"{prefix}right-output"]["loss_scale"] = loss_scale

    return network


def augment_net_with_diphone_outputs(
    shared_network: Network,
    label_info: LabelInfo,
    use_multi_task: bool,
    encoder_output_len: int,
    frame_rate_reduction_ratio_info: FrameRateReductionRatioinfo,
    l2: float = 0.0,
    label_smoothing=0.2,
    encoder_output_layer: str = "encoder-output",
    prefix: str = "",
    weights_init: str = DEFAULT_INIT,
) -> Network:
    assert (
        encoder_output_layer in shared_network
    ), f"net needs output layer '{encoder_output_layer}' layer to be predefined"

    network = copy.deepcopy(shared_network)
    center_target = "singleStateCenter" if frame_rate_reduction_ratio_info.single_state_alignment else "centerState"

    network["pastEmbed"] = get_embedding_layer(source="pastLabel", dim=label_info.ph_emb_size, l2=l2)
    network[f"{prefix}linear1-diphone"]["from"] = [encoder_output_layer, "pastEmbed"]

    if use_multi_task:
        network["currentState"] = get_embedding_layer(source=center_target, dim=label_info.st_emb_size, l2=l2)
        network[f"{prefix}linear1-triphone"]["from"] = [encoder_output_layer, "currentState"]
    else:
        loss_opts = copy.deepcopy(network[f"{prefix}center-output"]["loss_opts"])
        loss_opts["label_smoothing"] = label_smoothing
        left_ctx_mlp = add_mlp(network, "leftContext", encoder_output_len, l2=l2, prefix=prefix, init=weights_init)
        network[f"{prefix}left-output"] = {
            "class": "softmax",
            "from": left_ctx_mlp,
            "target": "pastLabel",
            "loss": "ce",
            "loss_opts": loss_opts,
        }

    network[f"{prefix}center-output"]["loss_opts"].pop("label_smoothing", None)
    network[f"{prefix}center-output"]["target"] = center_target

    return network


def augment_with_triphone_embeds(
    shared_network: Network,
    frame_rate_reduction_ratio_info: FrameRateReductionRatioinfo,
    label_info: LabelInfo,
    l2: float,
    copy_net=True,
) -> Network:
    network = copy.deepcopy(shared_network) if copy_net else shared_network
    center_target, center_dim = (
        ("singleStateCenter", label_info.get_n_single_state_classes())
        if frame_rate_reduction_ratio_info.single_state_alignment
        else ("centerState", label_info.get_n_state_classes())
    )

    network["pastEmbed"] = get_embedding_layer(source="pastLabel", dim=label_info.ph_emb_size, l2=l2)
    network["currentState"] = get_embedding_layer(source=center_target, dim=label_info.st_emb_size, l2=l2)
    return network


def augment_net_with_triphone_outputs(
    shared_network: Network,
    variant: PhoneticContext,
    frame_rate_reduction_ratio_info: FrameRateReductionRatioinfo,
    label_info: LabelInfo,
    l2: float = 0.0,
    encoder_output_layer: str = "encoder-output",
    prefix: str = "",
) -> Network:
    assert (
        encoder_output_layer in shared_network
    ), f"net needs output layer '{encoder_output_layer}' layer to be predefined"
    assert variant == PhoneticContext.triphone_forward, "only triphone forward is implemented"

    network = copy.deepcopy(shared_network)

    network = augment_with_triphone_embeds(
        network,
        frame_rate_reduction_ratio_info=frame_rate_reduction_ratio_info,
        label_info=label_info,
        l2=l2,
        copy_net=False,
    )

    network[f"{prefix}linear1-diphone"]["from"] = [encoder_output_layer, "pastEmbed"]
    network[f"{prefix}linear1-triphone"]["from"] = [
        encoder_output_layer,
        "currentState",
        "pastEmbed",
    ]

    if "loss_opts" in network[f"{prefix}center-output"]:
        network[f"{prefix}center-output"]["loss_opts"].pop("label_smoothing", None)

    return network


def remove_label_pops_and_losses(network: Network, except_layers: Optional[Iterable[str]] = None) -> Network:
    network = copy.copy(network)
    except_layers = [] if except_layers is None else except_layers

    layers_to_pop = {
        "centerPhoneme",
        "stateId",
        "pastLabel",
        "popFutureLabel",
        "futureLabel",
        "classes_",
    } - set(except_layers or [])
    for k in layers_to_pop:
        network.pop(k, None)

    for center_target in ["centerState", "singleStateCenter"]:
        if center_target in network and center_target not in except_layers:
            network.pop(center_target, None)

    for layer in network.values():
        layer.pop("target", None)
        layer.pop("loss", None)
        layer.pop("loss_scale", None)
        layer.pop("loss_opts", None)

    return network


def remove_label_pops_and_losses_from_returnn_config(
    cfg: returnn.ReturnnConfig, except_layers: Optional[Iterable[str]] = None, except_extern_data: Optional[Iterable[str]] = None, modify_chunking: bool = True
) -> returnn.ReturnnConfig:
    cfg = copy.deepcopy(cfg)
    except_layers = [] if except_layers is None else except_layers
    except_extern_data = [] if except_extern_data is None else except_extern_data
    cfg.config["network"] = remove_label_pops_and_losses(cfg.config["network"], except_layers)

    for k in ["centerState", "singleStateCenter", "classes", "futureLabel", "pastLabel"]:
        if k in cfg.config["extern_data"] and k not in except_extern_data:
            cfg.config["extern_data"].pop(k, None)


    chk_cfg = cfg.config.get("chunking", None)
    if modify_chunking:
        if isinstance(chk_cfg, tuple):
            cfg.config["chunking"] = f"{chk_cfg[0]['data']}:{chk_cfg[1]['data']}"

    return cfg


def add_label_prior_layer_to_network(
    network: Network,
    reference_layer: str = "center-output",
    label_prior_type: Optional[PriorType] = None,
    label_prior: Optional[returnn.CodeWrapper] = None,
    label_prior_estimation_axes: str = None,
):
    out_denot = reference_layer.split("-")[0]
    prior_name = ("_").join(["label_prior", out_denot])

    if label_prior_type == PriorType.TRANSCRIPT:
        assert label_prior is not None, "You forgot to provide the prior values"
        network[prior_name] = {"class": "constant", "dtype": "float32", "value": label_prior}
    elif label_prior_type == PriorType.AVERAGE:
        network[prior_name] = {
            "class": "accumulate_mean",
            "exp_average": 0.001,
            "from": reference_layer,
            "is_prob_distribution": True,
        }
    elif label_prior_type == PriorType.ONTHEFLY:
        assert (
            label_prior_estimation_axes is not None
        ), "You forgot to set one which axis you want to average the prior, eg. bt"
        network[prior_name] = {
            "class": "reduce",
            "mode": "mean",
            "from": reference_layer,
            "axis": label_prior_estimation_axes,
        }
    else:
        raise NotImplementedError("Unknown PriorType")

    return network, prior_name

def add_fast_bw_layer_to_network(
    crp: rasr.CommonRasrParameters,
    network: Network,
    log_linear_scales: LogLinearScales,
    reference_layer: str = "center-output",
    label_prior_type: Optional[PriorType] = None,
    label_prior: Optional[returnn.CodeWrapper] = None,
    label_prior_estimation_axes: str = None,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,
) -> Network:

    crp = correct_rasr_FSA_bug(crp)

    if label_prior_type is not None:
        assert (
            log_linear_scales.label_prior_scale is not None
        ), "If you plan to use the prior, please set the scale for it"
        if label_prior_type == PriorType.TRANSCRIPT:
            assert label_prior is not None, "You forgot to set the label prior file"

    for attribute in ["loss", "loss_opts", "target"]:
        if reference_layer in network:
            network[reference_layer].pop(attribute, None)

    inputs = []
    out_denot = reference_layer.split("-")[0]
    # prior calculation

    if label_prior_type is not None:
        prior_name = ("_").join(["label_prior", out_denot])
        comb_name = ("_").join(["comb-prior", out_denot])
        prior_eval_string = "(safe_log(source(1)) * prior_scale)"
        inputs.append(comb_name)
        if label_prior_type == PriorType.TRANSCRIPT:
            network[prior_name] = {"class": "constant", "dtype": "float32", "value": label_prior}
        elif label_prior_type == PriorType.AVERAGE:
            network[prior_name] = {
                "class": "accumulate_mean",
                "exp_average": 0.001,
                "from": reference_layer,
                "is_prob_distribution": True,
            }
        elif label_prior_type == PriorType.ONTHEFLY:
            assert (
                label_prior_estimation_axes is not None
            ), "You forgot to set one which axis you want to average the prior, eg. bt"
            network[prior_name] = {
                "class": "reduce",
                "mode": "mean",
                "from": reference_layer,
                "axis": label_prior_estimation_axes,
            }
            prior_eval_string = "tf.stop_gradient((safe_log(source(1)) * prior_scale))"
        else:
            raise NotImplementedError("Unknown PriorType")

        network[comb_name] = {
            "class": "combine",
            "kind": "eval",
            "eval": f"am_scale*(safe_log(source(0)) - {prior_eval_string})",
            "eval_locals": {
                "am_scale": log_linear_scales.label_posterior_scale,
                "prior_scale": log_linear_scales.label_prior_scale,
            },
            "from": [reference_layer, prior_name],
        }

    else:
        comb_name = ("_").join(["multiply-scale", out_denot])
        inputs.append(comb_name)
        network[comb_name] = {
            "class": "combine",
            "kind": "eval",
            "eval": "am_scale*(safe_log(source(0)))",
            "eval_locals": {"am_scale": log_linear_scales.label_posterior_scale},
            "from": [reference_layer],
        }

    network["output_bw"] = {
        "class": "copy",
        "from": reference_layer,
        "loss": "via_layer",
        "loss_opts": {"align_layer": "fast_bw", "loss_wrt_to_act_in": "softmax"},
        "loss_scale": 1.0,
    }

    network["fast_bw"] = {
        "class": "fast_bw",
        "align_target": "sprint",
        "from": inputs,
        "tdp_scale": log_linear_scales.transition_scale,
    }

    automaton_config = create_rasrconfig_for_alignment_fsa(
        crp=crp,
        extra_rasr_config=extra_rasr_config,
        extra_rasr_post_config=extra_rasr_post_config,

    )

    network["fast_bw"]["sprint_opts"] = {
        "sprintExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
        "sprintConfigStr": DelayedFormat("--config={}", automaton_config),
        "sprintControlConfig": {"verbose": True},
        "usePythonSegmentOrder": False,
        "numInstances": 1,
    }

    return network


def add_fast_bw_layer_to_returnn_config(
    crp: rasr.CommonRasrParameters,
    returnn_config: returnn.ReturnnConfig,
    log_linear_scales: LogLinearScales,
    import_model: [tk.Path, str] = None,
    reference_layer: str = "center-output",
    label_prior_type: Optional[PriorType] = None,
    label_prior: Optional[returnn.CodeWrapper] = None,
    label_prior_estimation_axes: str = None,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,
) -> returnn.ReturnnConfig:

    returnn_config.config["network"] = add_fast_bw_layer_to_network(
        crp=crp,
        network=returnn_config.config["network"],
        log_linear_scales=log_linear_scales,
        reference_layer=reference_layer,
        label_prior_type=label_prior_type,
        label_prior=label_prior,
        label_prior_estimation_axes=label_prior_estimation_axes,
        extra_rasr_config=extra_rasr_config,
        extra_rasr_post_config=extra_rasr_post_config,
    )

    if "chunking" in returnn_config.config:
        del returnn_config.config["chunking"]
    if "pretrain" in returnn_config.config and import_model is not None:
        del returnn_config.config["pretrain"]

    # ToDo: handel the import model part

    return returnn_config


def add_fast_bw_factored_layer_to_network(
    crp: rasr.CommonRasrParameters,
    network: Network,
    log_linear_scales: LogLinearScales,
    loss_scales: LossScales,
    label_info: LabelInfo,
    reference_layers: [str] = ["left-output", "center-output" "right-output"],
    label_prior_type: Optional[PriorType] = None,
    label_prior: Optional[returnn.CodeWrapper] = None,
    label_prior_estimation_axes: str = None,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,
) -> Network:

    crp = correct_rasr_FSA_bug(crp)

    if label_prior_type is not None:
        assert (
            log_linear_scales.label_prior_scale is not None
        ), "If you plan to use the prior, please set the scale for it"
        if label_prior_type == PriorType.TRANSCRIPT:
            assert label_prior is not None, "You forgot to set the label prior file"

    inputs = []
    for reference_layer in reference_layers:
        for attribute in ["loss", "loss_opts", "target"]:
            if reference_layer in network:
                network[reference_layer].pop(attribute, None)

        out_denot = reference_layer.split("-")[0]
        am_scale = (
            log_linear_scales.label_posterior_scale
            if "center" in reference_layer
            else log_linear_scales.context_label_posterior_scale
        )
        # prior calculation
        if label_prior_type is not None:
            prior_name = ("_").join(["label_prior", out_denot])
            comb_name = ("_").join(["comb-prior", out_denot])
            prior_eval_string = "(safe_log(source(1)) * prior_scale)"
            inputs.append(comb_name)
            if label_prior_type == PriorType.TRANSCRIPT:
                network[prior_name] = {"class": "constant", "dtype": "float32", "value": label_prior}
            elif label_prior_type == PriorType.AVERAGE:
                network[prior_name] = {
                    "class": "accumulate_mean",
                    "exp_average": 0.001,
                    "from": reference_layer,
                    "is_prob_distribution": True,
                }
            elif label_prior_type == PriorType.ONTHEFLY:
                assert (
                    label_prior_estimation_axes is not None
                ), "You forgot to set one which axis you want to average the prior, eg. bt"
                network[prior_name] = {
                    "class": "reduce",
                    "mode": "mean",
                    "from": reference_layer,
                    "axis": label_prior_estimation_axes,
                }
                prior_eval_string = "tf.stop_gradient((safe_log(source(1)) * prior_scale))"
            else:
                raise NotImplementedError("Unknown PriorType")

            network[comb_name] = {
                "class": "combine",
                "kind": "eval",
                "eval": f"am_scale*(safe_log(source(0)) - {prior_eval_string})",
                "eval_locals": {
                    "am_scale": am_scale,
                    "prior_scale": log_linear_scales.label_prior_scale,
                },
                "from": [reference_layer, prior_name],
            }

        else:
            comb_name = ("_").join(["multiply-scale", out_denot])
            inputs.append(comb_name)
            network[comb_name] = {
                "class": "combine",
                "kind": "eval",
                "eval": "am_scale*(safe_log(source(0)))",
                "eval_locals": {"am_scale": am_scale},
                "from": [reference_layer],
            }

        bw_out = ("_").join(["output-bw", out_denot])
        network[bw_out] = {
            "class": "copy",
            "from": reference_layer,
            "loss": "via_layer",
            "loss_opts": {
                "align_layer": ("/").join(["fast_bw", out_denot]),
                "loss_wrt_to_act_in": "softmax",
            },
            "loss_scale": loss_scales.get_scale(reference_layer),
        }

    network["fast_bw"] = {
        "class": "fast_bw_factored",
        "align_target": "hmm-monophone",
        "hmm_opts": {"num_contexts": label_info.n_contexts},
        "from": inputs,
        "tdp_scale": log_linear_scales.transition_scale,
        "n_out": label_info.n_contexts * 2 + label_info.get_n_state_classes(),
    }

    automaton_config = create_rasrconfig_for_alignment_fsa(
        crp=crp,
        extra_rasr_config=extra_rasr_config,
        extra_rasr_post_config=extra_rasr_post_config,

    )

    network["fast_bw"]["sprint_opts"] = {
        "sprintExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
        "sprintConfigStr": DelayedFormat("--config={}", automaton_config),
        "sprintControlConfig": {"verbose": True},
        "usePythonSegmentOrder": False,
        "numInstances": 1,
    }

    return network


def add_fast_bw_factored_layer_to_returnn_config(
    crp: rasr.CommonRasrParameters,
    returnn_config: returnn.ReturnnConfig,
    log_linear_scales: LogLinearScales,
    loss_scales: LossScales,
    label_info: LabelInfo,
    import_model: [tk.Path, str] = None,
    reference_layers: [str] = ["left-output", "center-output", "right-output"],
    label_prior_type: Optional[PriorType] = None,
    label_prior: Optional[returnn.CodeWrapper] = None,
    label_prior_estimation_axes: str = None,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,
) -> returnn.ReturnnConfig:

    returnn_config.config["network"] = add_fast_bw_factored_layer_to_network(
        crp=crp,
        network=returnn_config.config["network"],
        log_linear_scales=log_linear_scales,
        loss_scales=loss_scales,
        label_info=label_info,
        reference_layers=reference_layers,
        label_prior_type=label_prior_type,
        label_prior=label_prior,
        label_prior_estimation_axes=label_prior_estimation_axes,
        extra_rasr_config=extra_rasr_config,
        extra_rasr_post_config=extra_rasr_post_config,
    )

    if "chunking" in returnn_config.config:
        del returnn_config.config["chunking"]
    if "pretrain" in returnn_config.config and import_model is not None:
        del returnn_config.config["pretrain"]

    return returnn_config
