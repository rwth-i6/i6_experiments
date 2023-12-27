__all__ = ["augment_returnn_config_to_joint_diphone_softmax",
           "get_prolog_augment_network_to_joint_diphone_softmax"]

import copy
from textwrap import dedent

from i6_core import returnn

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo
from i6_experiments.users.raissi.setups.common.helpers.network.augment import Network


def augment_returnn_config_to_joint_diphone_softmax(
    returnn_config: returnn.ReturnnConfig,
    label_info: LabelInfo,
    out_joint_score_layer: str,
    log_softmax: bool,
    center_state_softmax_layer: str = "center-output",
    left_context_softmax_layer: str = "left-output",
    encoder_output_layer: str = "encoder-output",
    prepare_for_train: bool = False,
    keep_right_context: bool = False
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
    for k in ["centerState", "singleStateCenter", "futureLabel", "pastLabel"]:
        if k in extern_data:
            extern_data.pop(k, None)
    if "classes" in extern_data:
        extern_data["classes"].pop("same_dim_tags_as", None)
    extern_data[out_joint_score_layer] = {
        "available_for_inference": True,
        "dim": label_info.get_n_state_classes() * label_info.n_contexts,
        "same_dim_tags_as": extern_data["data"]["same_dim_tags_as"],
    }

    dim_prolog, network = get_prolog_augment_network_to_joint_diphone_softmax(
        network= returnn_config.config["network"],
        label_info = label_info,
        out_joint_score_layer=out_joint_score_layer,
        log_softmax=log_softmax,
        center_state_softmax_layer=center_state_softmax_layer,
        left_context_softmax_layer=left_context_softmax_layer,
        encoder_output_layer=encoder_output_layer,
        prepare_for_train=prepare_for_train
)



    update_cfg = returnn.ReturnnConfig({}, python_prolog=dim_prolog)
    returnn_config.update(update_cfg)

    return returnn_config



def get_prolog_augment_network_to_joint_diphone_softmax(
    network: Network,
    label_info: LabelInfo,
    out_joint_score_layer: str,
    log_softmax: bool,
    center_state_softmax_layer: str = "center-output",
    left_context_softmax_layer: str = "left-output",
    encoder_output_layer: str = "encoder-output",
    prepare_for_train: bool = False,
) -> Network:

    dim_prolog, c_spatial_dim, c_range_dim = get_context_dim_tag_prolog(
        spatial_size=label_info.n_contexts,
        feature_size=label_info.n_contexts,
        context_type="L",
        spatial_dim_variable_name="__center_state_spatial",
        feature_dim_variable_name="__center_state_feature")

    if not keep_right_context:
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
            "activation": "log_softmax" if log_softmax else "softmax",
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
    network[f"{center_state_softmax_layer}_transposed"] = {
        # Transpose the center output because in diphone-no-tying-dense, the left context is
        # the trailing index. So we have for every center state all left contexts next to each
        # other, not for every left context all center states.
        "class": "swap_axes",
        "axis1": 2,
        "axis2": 3,
        "from": center_state_softmax_layer,
    }

    # Left context just needs to be repeated |center_state| number of times
    network[f"{left_context_softmax_layer}_expanded"] = {
        "class": "expand_dims",
        "from": left_context_softmax_layer,
        "axis": "spatial",
    }

    # Compute scores and flatten output
    network[f"{out_joint_score_layer}_scores"] = {
        "class": "combine",
        "from": [f"{center_state_softmax_layer}_transposed", f"{left_context_softmax_layer}_expanded"],
        "kind": "add" if log_softmax else "mul",
    }
    network[out_joint_score_layer] = {
        "class": "merge_dims",
        "axes": [f"dim:{label_info.get_n_state_classes()}", f"dim:{label_info.n_contexts}"],
        "keep_order": True,
        "from": f"{out_joint_score_layer}_scores",
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




