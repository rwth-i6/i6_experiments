__all__ = ["get_prolog_augment_network_to_joint_triphone_softmax",
           "augment_returnn_config_to_joint_triphone_softmax"]

import copy
from textwrap import dedent

from i6_core import returnn

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo
from i6_experiments.users.raissi.setups.common.helpers.network.augment import Network

from i6_experiments.users.raissi.setups.common.helpers.train.returnn_time_tag import get_context_dim_tag_prolog


def get_prolog_augment_network_to_joint_triphone_softmax(
    network: Network,
    label_info: LabelInfo,
    out_joint_score_layer: str,
    log_softmax: bool,
    left_context_softmax_layer: str = "left-output",
    center_state_softmax_layer: str = "center-output",
    right_context_softmax_layer: str = "right-output",
    encoder_output_layer: str = "encoder-output",
    prepare_for_train: bool = False,
) -> Network:
    dim_prolog_di, c_spatial_dim, l_range_dim = get_context_dim_tag_prolog(
        spatial_size=label_info.get_n_state_classes(),
        feature_size=label_info.n_contexts,
        context_type="L",
        spatial_dim_variable_name="__center_state_spatial",
        feature_dim_variable_name="__left_feature",
    )
    dim_prolog_tri, dense_spatial_dim, r_range_dim = get_context_dim_tag_prolog(
        spatial_size=label_info.get_n_of_dense_classes(),
        feature_size=label_info.n_contexts,
        context_type="R",
        spatial_dim_variable_name="__dense_spatial",
        feature_dim_variable_name="__right_feature",
    )
    dim_prolog = dim_prolog_di+dim_prolog_tri

    for layer in network.values():
        layer.pop("target", None)
        layer.pop("loss", None)
        layer.pop("loss_scale", None)
        layer.pop("loss_opts", None)

    for softmax_layer in [right_context_softmax_layer, center_state_softmax_layer, left_context_softmax_layer]:
        network[softmax_layer] = {
            **network[softmax_layer],
            "class": "linear",
            "activation": "log_softmax" if log_softmax else "softmax",
        }

    #Repeating the left context
    network["pastLabelRange"] = {
        "class": "range",
        "dtype": "int32",
        "start": 0,
        "limit": label_info.n_contexts,
        "sparse": True,
        "out_spatial_dim": l_range_dim,
    }
    network["pastLabel"] = {
        "class": "reinterpret_data",
        "from": "pastLabelRange",
        "set_sparse_dim": l_range_dim,
    }
    network["pastEmbed"]["in_dim"] = l_range_dim
    network["pastEmbed"]["from"] = "pastLabel"

    # Repeating the right context
    network["futureLabelRange"] = {
        "class": "range",
        "dtype": "int32",
        "start": 0,
        "limit": label_info.get_n_state_classes(),
        "sparse": True,
        "out_spatial_dim": c_spatial_dim,
    }
    network["futureLabel"] = {
        "class": "reinterpret_data",
        "from": "futureLabelRange",
        "set_sparse_dim": c_spatial_dim,
    }
    network["currentState"]["in_dim"] = c_spatial_dim
    network["currentState"]["from"] = "futureLabel"


    network["left-output"].pop("n_out", None)
    network["left-output"]["out_dim"] = l_range_dim

    network["right-output"].pop("n_out", None)
    network["right-output"]["out_dim"] = r_range_dim


    network["linear1-triphone"]["from"] = [f"{encoder_output_layer}", "currentState", "pastEmbed"]

    # Compute scores and flatten output
    network[f"{out_joint_score_layer}_scores"] = {
        "class": "combine",
        "from": [right_context_softmax_layer, center_state_softmax_layer, left_context_softmax_layer],
        "kind": "add" if log_softmax else "mul",
    }
    network[out_joint_score_layer] = {
        "class": "merge_dims",
        "axes": [c_spatial_dim, l_range_dim, r_range_dim],
        "keep_order": True,
        "from": f"{out_joint_score_layer}_scores",
        "out_dim": dense_spatial_dim,
    }

    if prepare_for_train:
        # To train numerically stable RETURNN needs a softmax activation at the end.
        #
        # Here we're using the fact that softmax(log_softmax(x)) = x to add a softmax
        # layer w/o actually running two softmaxes on top of each other.

        assert log_softmax

        network[f"{out_joint_score_layer}_merged"] = network[out_joint_score_layer]
        network[out_joint_score_layer] = {
            "class": "activation",
            "activation": "softmax",
            "from": f"{out_joint_score_layer}_merged",
        }

    network[out_joint_score_layer]["register_as_extern_data"] = out_joint_score_layer

    return dim_prolog, network


def augment_returnn_config_to_joint_triphone_softmax(
    returnn_config: returnn.ReturnnConfig,
    label_info: LabelInfo,
    out_joint_score_layer: str,
    log_softmax: bool,
    left_context_softmax_layer: str = "left-output",
    center_state_softmax_layer: str = "center-output",
    right_context_softmax_layer: str = "right-output",
    encoder_output_layer: str = "encoder-output",
    prepare_for_train: bool = False,
) -> returnn.ReturnnConfig:
    """
    Assumes a diphone FH model and expands the model to calculate the scores for the joint
    posteriors `p(c, l)` instead of having two outputs `p(c | l)` and `p(l)`.

    The output layer contains normalized (log) acoustic scores.
    """

    assert not prepare_for_train or log_softmax, "training preparation implies log-softmax"

    returnn_config = copy.deepcopy(returnn_config)

    returnn_config.config["forward_output_layer"] = out_joint_score_layer

    extern_data = returnn_config.config["extern_data"]
    extern_data[out_joint_score_layer] = {
        "available_for_inference": True,
        "dim": label_info.get_n_state_classes() * label_info.n_contexts *label_info.n_contexts,
        "same_dim_tags_as": extern_data["data"]["same_dim_tags_as"],
    }

    dim_prolog, network = get_prolog_augment_network_to_joint_triphone_softmax(
        network=returnn_config.config["network"],
        label_info=label_info,
        out_joint_score_layer=out_joint_score_layer,
        log_softmax=log_softmax,
        center_state_softmax_layer=center_state_softmax_layer,
        left_context_softmax_layer=left_context_softmax_layer,
        right_context_softmax_layer=right_context_softmax_layer,
        encoder_output_layer=encoder_output_layer,
        prepare_for_train=prepare_for_train,
    )

    update_cfg = returnn.ReturnnConfig({}, python_prolog=dim_prolog)
    returnn_config.update(update_cfg)

    return returnn_config