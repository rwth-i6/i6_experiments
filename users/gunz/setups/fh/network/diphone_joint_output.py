__all__ = ["augment_to_joint_diphone_softmax"]

import copy
import math
from textwrap import dedent

from i6_core import returnn

from ..factored import LabelInfo


def augment_to_joint_diphone_softmax(
    returnn_config: returnn.ReturnnConfig,
    label_info: LabelInfo,
    out_joint_score_layer: str,
    log_softmax: bool,
    center_state_softmax_layer: str = "center-output",
    left_context_softmax_layer: str = "left-output",
    encoder_output_layer: str = "encoder-output",
) -> returnn.ReturnnConfig:
    """
    Assumes a diphone FH model and expands the model output a softmax over the joint
    output probability `p(c, l)` instead of having two outputs `p(c | l)` and `p(l)`.

    The output layer contains normalized log acoustic scores.
    """

    returnn_config = copy.deepcopy(returnn_config)

    returnn_config.config["forward_output_layer"] = out_joint_score_layer

    extern_data = returnn_config.config["extern_data"]
    for k in ["centerState", "futureLabel", "pastLabel"]:
        extern_data.pop(k, None)
    extern_data["classes"].pop("same_dim_tags_as", None)
    extern_data[out_joint_score_layer] = {
        "available_for_inference": True,
        "dim": label_info.get_n_state_classes() * label_info.n_contexts,
        "same_dim_tags_as": extern_data["data"]["same_dim_tags_as"],
    }

    center_state_spatial_dim_variable_name = "__center_state_spatial"
    center_state_feature_dim_variable_name = "__center_state_feature"
    left_context_repetition_spatial_dim_variable_name = "__left_context_spatial"

    dim_prolog = dedent(
        f"""
        from returnn.tf.util.data import FeatureDim

        {center_state_spatial_dim_variable_name} = FeatureDim("contexts-L", {label_info.n_contexts})
        {center_state_feature_dim_variable_name} = FeatureDim("L", {label_info.n_contexts})
        {left_context_repetition_spatial_dim_variable_name} = FeatureDim("repeat-L", {label_info.get_n_state_classes()})
        """
    )
    c_spatial_dim = returnn.CodeWrapper(center_state_spatial_dim_variable_name)
    c_range_dim = returnn.CodeWrapper(center_state_feature_dim_variable_name)
    l_spatial_dim = returnn.CodeWrapper(left_context_repetition_spatial_dim_variable_name)

    network = returnn_config.config["network"]

    for k in ["linear1-triphone", "linear2-triphone", "right-output"]:
        # reduce error surface, remove all triphone-related stuff
        network.pop(k, None)
    for layer in network.values():
        layer.pop("target", None)
        layer.pop("loss", None)
        layer.pop("loss_scale", None)
        layer.pop("loss_opts", None)

    for softmax_layer in [center_state_softmax_layer, left_context_softmax_layer]:
        network[softmax_layer] = {
            **network[softmax_layer],
            "class": "linear",
            "activation": "log_softmax",
        }

    # Preparation of expanded center-state
    network[f"{encoder_output_layer}_expanded"] = {
        "class": "expand_dims",
        "from": encoder_output_layer,
        "axis": "spatial",
        "dim": c_spatial_dim,
    }
    network["pastLabelRange"] = {
        "class": "range",
        "dtype": "int32",
        "start": 0,
        "limit": label_info.n_contexts,
        "sparse": True,
        "out_spatial_dim": c_spatial_dim,
    }
    network["pastLabel"] = {
        "class": "reinterpret_data",
        "from": "pastLabelRange",
        "set_sparse_dim": c_range_dim,
    }
    network["pastEmbed"]["in_dim"] = c_range_dim
    network["pastEmbed"]["from"] = "pastLabel"
    network["linear1-diphone"]["from"] = [f"{encoder_output_layer}_expanded", "pastEmbed"]

    # Left context just needs to be repeated |center_state| number of times
    network[f"{left_context_softmax_layer}_expanded"] = {
        "class": "expand_dims",
        "from": left_context_softmax_layer,
        "axis": "feature",
        "dim": l_spatial_dim,
    }

    # Compute scores and flatten output
    network[f"{out_joint_score_layer}_scores"] = {
        "class": "combine",
        "from": [center_state_softmax_layer, f"{left_context_softmax_layer}_expanded"],
        "kind": "add",  # log space
    }
    network[out_joint_score_layer] = {
        "class": "merge_dims",
        "axes": [c_spatial_dim, "F"],
        "keep_order": True,
        "from": f"{out_joint_score_layer}_scores",
    }

    if not log_softmax:
        network[f"{out_joint_score_layer}_normalized"] = network[out_joint_score_layer]
        network[out_joint_score_layer] = {
            "class": "activation",
            "activation": "exp",
            "from": f"{out_joint_score_layer}_normalized",
        }

    network[out_joint_score_layer]["register_as_extern_data"] = out_joint_score_layer

    update_cfg = returnn.ReturnnConfig({}, python_prolog=dim_prolog)
    returnn_config.update(update_cfg)

    return returnn_config
