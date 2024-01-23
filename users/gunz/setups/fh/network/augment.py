import copy
from dataclasses import dataclass
import typing

from i6_core import returnn

from ..factored import LabelInfo, PhonemeStateClasses, PhoneticContext, RasrStateTying

DEFAULT_INIT = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)"

Layer = typing.Dict[str, typing.Any]
Network = typing.Dict[str, Layer]


def add_mlp(
    network: Network,
    layer_name: str,
    size: int,
    *,
    source_layer: typing.Union[str, typing.List[str]] = "encoder-output",
    prefix: str = "",
    l2: typing.Optional[float] = None,
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


def get_embedding_layer(source: typing.Union[str, typing.List[str]], dim: int, l2=0.01):
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
) -> typing.Tuple[Network, str, int]:
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


@dataclass(frozen=True, eq=True)
class SubsamplingInfo:
    factor: typing.Union[int, typing.List[int]]
    time_tag_name: str


def augment_net_with_label_pops(
    network: Network,
    label_info: LabelInfo,
    classes_subsampling_info: typing.Optional[SubsamplingInfo] = None,
    prefix: str = "",
    labeling_input: str = "data:classes",
) -> Network:
    assert label_info.state_tying in [RasrStateTying.diphone, RasrStateTying.triphone]

    remaining_label_dim = label_info.get_n_of_dense_classes()

    network = copy.deepcopy(network)

    if classes_subsampling_info is not None:
        # This layer sets the time step ratio between the input and the output of the NN.

        ss_factors = (
            [classes_subsampling_info.factor]
            if isinstance(classes_subsampling_info.factor, int)
            else classes_subsampling_info.factor
        )
        t_tag = f"{classes_subsampling_info.time_tag_name}"
        for factor in ss_factors:
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

    return network


def augment_net_with_monophone_outputs(
    shared_network: Network,
    encoder_output_len: int,
    label_info: LabelInfo,
    *,
    add_mlps=True,
    use_multi_task=True,
    final_ctx_type: typing.Optional[PhoneticContext] = None,
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
    encoder_out_len = encoder_output_len

    loss_opts = {}
    if focal_loss_factor > 0.0:
        loss_opts["focal_loss_factor"] = focal_loss_factor
    if label_smoothing > 0.0:
        loss_opts["label_smoothing"] = label_smoothing

    if add_mlps:
        if final_ctx_type == PhoneticContext.triphone_symmetric:
            tri_out = encoder_out_len + (2 * label_info.ph_emb_size)
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
                "target": "centerState",
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }

            if use_multi_task:
                context_mlp = add_mlp(
                    network,
                    "contexts",
                    encoder_out_len,
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
                network[f"{prefix}center-output"]["target"] = "centerState"
                network[f"{prefix}center-output"]["n_out"] = label_info.get_n_state_classes()

        elif final_ctx_type == PhoneticContext.triphone_forward:
            di_mlp = add_mlp(
                network,
                "diphone",
                encoder_out_len + label_info.ph_emb_size,
                prefix=prefix,
                source_layer=encoder_output_layer,
                l2=l2,
                init=weights_init,
            )

            network[f"{prefix}center-output"] = {
                "class": "softmax",
                "from": di_mlp,
                "target": "centerState",
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }

            if use_multi_task:
                tri_out = encoder_out_len + label_info.ph_emb_size + label_info.st_emb_size
                left_ctx_mlp = add_mlp(
                    network,
                    "leftContext",
                    encoder_out_len,
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
                network[f"{prefix}center-output"]["target"] = "centerState"
                network[f"{prefix}center-output"]["n_out"] = label_info.get_n_state_classes()

        elif final_ctx_type == PhoneticContext.triphone_backward:
            assert use_multi_task, "it is not possible to have a monophone backward without multitask"

            center_mlp = add_mlp(
                network,
                "centerState",
                encoder_out_len,
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
                "target": "centerState",
                "n_out": label_info.get_n_state_classes(),
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
            raise "this is not implemented yet"

            delta_blstm_n = f"{prefix}deltaEncoder-output"
            di_out = encoder_out_len + ph_emb_size

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
                "target": "centerState",
                "loss": "ce",
                "loss_opts": copy.copy(loss_opts),
            }

            if use_multi_task:
                tri_out = encoder_out_len + ph_emb_size + st_emb_size
                left_ctx_mlp = add_mlp(
                    network,
                    "leftContext",
                    encoder_out_len,
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
                network[f"{prefix}center-output"]["target"] = "centerState"
                network[f"{prefix}center-output"]["n_out"] = label_info.get_n_state_classes()
    else:
        network[f"{prefix}center-output"] = {
            "class": "softmax",
            "from": encoder_output_layer,
            "target": "centerState",
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
            network[f"{prefix}center-output"]["target"] = "centerState"
            network[f"{prefix}center-output"]["n_out"] = label_info.get_n_state_classes()

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
    network["pastEmbed"] = get_embedding_layer(source="pastLabel", dim=ph_emb_size, l2=l2)
    network[f"{prefix}linear1-diphone"]["from"] = [encoder_output_layer, "pastEmbed"]

    if use_multi_task:
        network["currentState"] = get_embedding_layer(source="centerState", dim=st_emb_size, l2=l2)
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
    network[f"{prefix}center-output"]["target"] = "centerState"

    return network


def augment_with_triphone_embeds(
    shared_network: Network,
    ph_emb_size: int,
    st_emb_size: int,
    l2: float,
    copy_net=True,
) -> Network:
    network = copy.deepcopy(shared_network) if copy_net else shared_network
    network["pastEmbed"] = get_embedding_layer(source="pastLabel", dim=ph_emb_size, l2=l2)
    network["currentState"] = get_embedding_layer(source="centerState", dim=st_emb_size, l2=l2)
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


def remove_label_pops_and_losses(
    network: Network, except_layers: typing.Optional[typing.Iterable[str]] = None
) -> Network:
    network = copy.copy(network)

    layers_to_pop = {
        "centerPhoneme",
        "stateId",
        "centerState",
        "pastLabel",
        "popFutureLabel",
        "futureLabel",
        "classes_",
    } - set(except_layers or [])
    for k in layers_to_pop:
        network.pop(k, None)

    for layer in network.values():
        layer.pop("target", None)
        layer.pop("loss", None)
        layer.pop("loss_scale", None)
        layer.pop("loss_opts", None)

    return network


def remove_label_pops_and_losses_from_returnn_config(
    cfg: returnn.ReturnnConfig, except_layers: typing.Optional[typing.Iterable[str]] = None
) -> returnn.ReturnnConfig:
    cfg = copy.deepcopy(cfg)
    cfg.config["network"] = remove_label_pops_and_losses(cfg.config["network"], except_layers)

    for k in ["centerState", "classes", "futureLabel", "pastLabel"]:
        cfg.config["extern_data"].pop(k, None)

    chk_cfg = cfg.config.get("chunking", None)
    if isinstance(chk_cfg, tuple):
        cfg.config["chunking"] = f"{chk_cfg[0]['data']}:{chk_cfg[1]['data']}"

    return cfg
