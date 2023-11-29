import copy
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

from i6_experiments.users.raissi.setups.common.helpers.network.frame_rate import FrameRateReductionRatioinfo

DEFAULT_INIT = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)"

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
) -> str:
    l_one_name = f"{prefix}linear1-{layer_name}"
    l_two_name = f"{prefix}linear2-{layer_name}"

    network[l_one_name] = {
        "class": "linear",
        "activation": "relu",
        "from": source_layer,
        "n_out": size,
        "forward_weights_init": init,
    }

    network[l_two_name] = {
        "class": "linear",
        "activation": "relu",
        "from": l_one_name,
        "n_out": size,
        "forward_weights_init": init,
    }

    if l2 is not None:
        network[l_one_name]["L2"] = l2
        network[l_two_name]["L2"] = l2

    return l_two_name


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
    center_target, center_dim = ("singleStateCenter", label_info.get_n_single_state_classes()) if frame_rate_reduction_ratio_info.single_state_alignment else ("centerState", label_info.get_n_state_classes())

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
    use_multi_task: bool,
    encoder_output_len: int,
    frame_rate_reduction_ratio_info: FrameRateReductionRatioinfo,
    l2: float = 0.0,
    label_smoothing=0.2,
    ph_emb_size=64,
    st_emb_size=256,
    encoder_output_layer: str = "encoder-output",
    prefix: str = "",
    weights_init: str = DEFAULT_INIT,
) -> Network:
    assert (
        encoder_output_layer in shared_network
    ), f"net needs output layer '{encoder_output_layer}' layer to be predefined"

    network = copy.deepcopy(shared_network)
    center_target = "singleStateCenter" if frame_rate_reduction_ratio_info.single_state_alignment else "centerState"

    network["pastEmbed"] = get_embedding_layer(source="pastLabel", dim=ph_emb_size, l2=l2)
    network[f"{prefix}linear1-diphone"]["from"] = [encoder_output_layer, "pastEmbed"]

    if use_multi_task:
        network["currentState"] = get_embedding_layer(source=center_target, dim=st_emb_size, l2=l2)
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
    ph_emb_size: int,
    st_emb_size: int,
    l2: float,
    copy_net=True,
) -> Network:
    network = copy.deepcopy(shared_network) if copy_net else shared_network
    center_target, center_dim = ("singleStateCenter",
                                 label_info.get_n_single_state_classes()) if frame_rate_reduction_ratio_info.single_state_alignment else (
    "centerState", label_info.get_n_state_classes())

    network["pastEmbed"] = get_embedding_layer(source="pastLabel", dim=ph_emb_size, l2=l2)
    network["currentState"] = get_embedding_layer(source=center_target, dim=st_emb_size, l2=l2)
    return network


def augment_net_with_triphone_outputs(
    shared_network: Network,
    variant: PhoneticContext,
    l2: float = 0.0,
    ph_emb_size=64,
    st_emb_size=256,
    encoder_output_layer: str = "encoder-output",
    prefix: str = "",
) -> Network:
    assert (
        encoder_output_layer in shared_network
    ), f"net needs output layer '{encoder_output_layer}' layer to be predefined"
    assert variant == PhoneticContext.triphone_forward, "only triphone forward is implemented"

    network = copy.deepcopy(shared_network)

    network = augment_with_triphone_embeds(
        network, ph_emb_size=ph_emb_size, st_emb_size=st_emb_size, l2=l2, copy_net=False
    )

    network[f"{prefix}linear1-diphone"]["from"] = [encoder_output_layer, "pastEmbed"]
    network[f"{prefix}linear1-triphone"]["from"] = [
        encoder_output_layer,
        "currentState",
        "pastEmbed",
    ]

    network[f"{prefix}center-output"]["loss_opts"].pop("label_smoothing", None)

    return network


def remove_label_pops_and_losses(network: Network, except_layers: Optional[Iterable[str]] = None) -> Network:
    network = copy.copy(network)

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
        if center_target in network:
            network.pop(center_target, None)


    for layer in network.values():
        layer.pop("target", None)
        layer.pop("loss", None)
        layer.pop("loss_scale", None)
        layer.pop("loss_opts", None)

    return network


def remove_label_pops_and_losses_from_returnn_config(
    cfg: returnn.ReturnnConfig, except_layers: Optional[Iterable[str]] = None
) -> returnn.ReturnnConfig:
    cfg = copy.deepcopy(cfg)
    cfg.config["network"] = remove_label_pops_and_losses(cfg.config["network"], except_layers)

    for k in ["centerState", "singleStateCenter", "classes", "futureLabel", "pastLabel"]:
        if k in cfg.config["extern_data"]:
            cfg.config["extern_data"].pop(k, None)

    chk_cfg = cfg.config.get("chunking", None)
    if isinstance(chk_cfg, tuple):
        cfg.config["chunking"] = f"{chk_cfg[0]['data']}:{chk_cfg[1]['data']}"

    return cfg


def add_fast_bw_layer(
    crp: rasr.CommonRasrParameters,
    returnn_config: returnn.ReturnnConfig,
    log_linear_scales: Dict = None,
    import_model: [tk.Path, str] = None,
    reference_layer: str = "center-output",
    label_prior: Optional[returnn.CodeWrapper] = None,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,
) -> returnn.ReturnnConfig:

    crp.acoustic_model_config.tdp.applicator_type = "corrected"
    transition_types = ["*", "silence"]
    if crp.acoustic_model_config.tdp.tying_type == "global-and-nonword":
        for nw in [0, 1]:
            transition_types.append(f"nonword-{nw}")
    for t in transition_types:
        crp.acoustic_model_config.tdp[t].exit = 0.0

    if log_linear_scales is None:
        log_linear_scales = {"label_posterior_scale": 0.3, "transition_scale": 0.3}
    if "label_prior_scale" in log_linear_scales:
        assert prior is not None, "Hybrid HMM needs a transcription based prior for fullsum training"

    for attribute in ["loss", "loss_opts", "target"]:
        del returnn_config.config["network"][reference_layer][attribute]

    inputs = []
    out_denot = reference_layer.split("-")[0]
    # prior calculation

    if label_prior is not None:
        # Here we are creating a standard hybrid HMM, without prior we have a posterior HMM
        prior_name = ("_").join(["label_prior", reference_layer_denot])
        returnn_config.config["network"][prior_name] = {"class": "constant", "dtype": "float32", "value": label_prior}
        comb_name = ("_").join(["comb-prior", out_denot])
        inputs.append(comb_name)
        returnn_config.config["network"][comb_name] = {
            "class": "combine",
            "kind": "eval",
            "eval": "am_scale*( safe_log(source(0)) - (safe_log(source(1)) * prior_scale) )",
            "eval_locals": {
                "am_scale": log_linear_scales["label_posterior_scale"],
                "prior_scale": log_linear_scales["label_prior_scale"],
            },
            "from": [reference_layer, prior_name],
        }
    else:
        comb_name = ("_").join(["multiply-scale", out_denot])
        inputs.append(comb_name)
        returnn_config.config["network"][comb_name] = {
            "class": "combine",
            "kind": "eval",
            "eval": "am_scale*(safe_log(source(0)))",
            "eval_locals": {"am_scale": log_linear_scales["label_posterior_scale"]},
            "from": [reference_layer],
        }

    returnn_config.config["network"]["output_bw"] = {
        "class": "copy",
        "from": reference_layer,
        "loss": "via_layer",
        "loss_opts": {"align_layer": "fast_bw", "loss_wrt_to_act_in": "softmax"},
        "loss_scale": 1.0,
    }

    returnn_config.config["network"]["fast_bw"] = {
        "class": "fast_bw",
        "align_target": "sprint",
        "from": inputs,
        "tdp_scale": log_linear_scales["transition_scale"],
    }

    if "chunking" in returnn_config.config:
        del returnn_config.config["chunking"]
    if "pretrain" in returnn_config.config and import_model is not None:
        del returnn_config.config["pretrain"]

    # start training from existing model
    if import_model is not None:
        returnn_config.config["import_model_train_epoch1"] = import_model

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

    automaton_config = rasr.WriteRasrConfigJob(config, post_config).out_config
    tk.register_output("train/bw.config", automaton_config)

    returnn_config.config["network"]["fast_bw"]["sprint_opts"] = {
        "sprintExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
        "sprintConfigStr": DelayedFormat("--config={}", automaton_config),
        "sprintControlConfig": {"verbose": True},
        "usePythonSegmentOrder": False,
        "numInstances": 1,
    }

    return returnn_config
