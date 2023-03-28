import copy
from textwrap import dedent
import typing

from i6_core import returnn

from ..factored import LabelInfo


def get_context_dim_tag_prolog(
    spatial_size: int,
    feature_size: int,
    context_type: str,
    spatial_dim_variable_name: str,
    feature_dim_variable_name: str,
) -> typing.Tuple[str, returnn.CodeWrapper, returnn.CodeWrapper]:
    code = dedent(
        f"""
        from returnn.tf.util.data import FeatureDim, SpatialDim
        {spatial_dim_variable_name} = SpatialDim("contexts-{context_type}", {spatial_size})
        {feature_dim_variable_name} = FeatureDim("{context_type}", {feature_size})
        """
    )
    return (
        code,
        returnn.CodeWrapper(spatial_dim_variable_name),
        returnn.CodeWrapper(feature_dim_variable_name),
    )


def augment_for_center_state(
    center_state_config: returnn.ReturnnConfig,
    label_info: LabelInfo,
    center_state_softmax_layer: str,
    out_center_state_layer: str,
    center_state_batch_size: typing.Optional[int],
    left_context_range: typing.Optional[typing.Tuple[int, int]],
) -> typing.Tuple[returnn.CodeWrapper, returnn.CodeWrapper]:
    from_l, to_l = 0, label_info.n_contexts
    if left_context_range is not None:
        from_l, to_l = left_context_range
    num_ctx = to_l - from_l

    center_state_config.config["forward_output_layer"] = out_center_state_layer
    if center_state_batch_size is not None:
        center_state_config.config["batch_size"] = center_state_batch_size

    extern_data = center_state_config.config["extern_data"]
    extern_data[out_center_state_layer] = {
        "available_for_inference": True,
        "dim": label_info.get_n_state_classes() * num_ctx,
        "sparse": True,
        "same_dim_tags_as": extern_data["classes"]["same_dim_tags_as"],
        "dtype": "int32",
    }

    network = center_state_config.config["network"]
    dim_prolog, spatial_dim, range_dim = get_context_dim_tag_prolog(
        num_ctx,
        label_info.n_contexts,
        spatial_dim_variable_name="__left_context_spatial",
        feature_dim_variable_name="__left_context_feature",
        context_type="L",
    )
    network["expanded-encoder-output-left"] = {
        "class": "expand_dims",
        "from": network["encoder-output"]["from"],
        "axis": "spatial",
        "dim": spatial_dim,
    }
    network["encoder-output"]["from"] = "expanded-encoder-output-left"
    network["pastLabelRange"] = {
        "class": "range",
        "dtype": "int32",
        "limit": to_l,
        "start": from_l,
        "sparse": True,
        "out_spatial_dim": spatial_dim,
    }
    network["pastLabel"] = {
        "class": "reinterpret_data",
        "from": "pastLabelRange",
        "set_sparse_dim": range_dim,
    }
    network["pastEmbed"]["in_dim"] = range_dim
    network["pastEmbed"]["from"] = "pastLabel"

    network[f"{out_center_state_layer}_merged"] = {
        "class": "merge_dims",
        "axes": [spatial_dim, "F"],
        "keep_order": True,
        "from": center_state_softmax_layer,
    }
    # Ensure priors sum to one
    network[out_center_state_layer] = {
        "class": "eval",
        "eval": f"tf.math.divide(source(0), {num_ctx}.0)",
        "from": f"{out_center_state_layer}_merged",
        "register_as_extern_data": out_center_state_layer,
    }

    update_cfg = returnn.ReturnnConfig({}, python_prolog=dim_prolog)
    center_state_config.update(update_cfg)

    return spatial_dim, range_dim


def augment_for_right_context(
    right_context_config: returnn.ReturnnConfig,
    label_info: LabelInfo,
    right_context_softmax_layer: str,
    out_right_context_layer: str,
    center_state_spatial_dim: returnn.CodeWrapper,
    right_context_batch_size: typing.Optional[int],
) -> typing.Tuple[returnn.CodeWrapper, returnn.CodeWrapper]:
    right_context_config.config["forward_output_layer"] = out_right_context_layer
    if right_context_batch_size is not None:
        right_context_config.config["batch_size"] = right_context_batch_size

    extern_data = right_context_config.config["extern_data"]
    extern_data[out_right_context_layer] = {
        "available_for_inference": True,
        "dim": label_info.get_n_of_dense_classes(),
        "sparse": True,
        "same_dim_tags_as": extern_data["classes"]["same_dim_tags_as"],
        "dtype": "int32",
    }

    network = right_context_config.config["network"]
    (dim_prolog, right_context_spatial_dim, right_context_range_dim,) = get_context_dim_tag_prolog(
        label_info.get_n_state_classes(),
        label_info.get_n_state_classes(),
        spatial_dim_variable_name="__center_state_spatial",
        feature_dim_variable_name="__center_state_feature",
        context_type="C",
    )
    network["expanded-encoder-output-center"] = {
        "class": "expand_dims",
        "from": "expanded-encoder-output-left",
        "axis": "spatial",
        "dim": right_context_spatial_dim,
    }
    network["encoder-output"]["from"] = "expanded-encoder-output-center"
    network["centerStateRange"] = {
        "class": "range",
        "dtype": "int32",
        "limit": label_info.get_n_state_classes(),
        "start": 0,
        "sparse": True,
        "out_spatial_dim": right_context_spatial_dim,
    }
    network["centerState"] = {
        "class": "reinterpret_data",
        "from": "centerStateRange",
        "set_sparse_dim": right_context_range_dim,
    }
    network["currentState"]["in_dim"] = right_context_range_dim
    network["currentState"]["from"] = "centerState"

    network[f"{out_right_context_layer}_merged"] = {
        "class": "merge_dims",
        "axes": [center_state_spatial_dim, right_context_spatial_dim, "F"],
        "keep_order": True,
        "from": right_context_softmax_layer,
    }

    # Ensure priors sum to one
    total_num_contexts_scored = label_info.n_contexts * label_info.get_n_state_classes()
    network[out_right_context_layer] = {
        "class": "eval",
        "eval": f"tf.math.divide(source(0), {total_num_contexts_scored}.0)",
        "from": f"{out_right_context_layer}_merged",
        "register_as_extern_data": out_right_context_layer,
    }

    update_cfg = returnn.ReturnnConfig({}, python_prolog=dim_prolog)
    right_context_config.update(update_cfg)

    return right_context_spatial_dim, right_context_range_dim


def get_returnn_config_for_left_context_prior_estimation(
    config_in: returnn.ReturnnConfig,
    *,
    left_context_softmax_layer="left-output",
    left_context_batch_size: typing.Optional[int] = None,
) -> returnn.ReturnnConfig:

    """
    Assumes forward decomposition and augments the given returnn config to be
    compatible with ReturnnRasrComputePriorV2Job.

    Returns a config for left context.

    The output layers (as given by function parameters) are reshaped such that
    averaging their outputs directly yields the priors.

    The user needs to run ReturnnRasrComputePriorV2Job with the returned config.
    """

    # Left Context does not need any network modifications
    left_context_config = copy.deepcopy(config_in)
    left_context_config.config["forward_output_layer"] = left_context_softmax_layer
    if left_context_batch_size is not None:
        left_context_config.config["batch_size"] = left_context_batch_size

    return left_context_config


def get_returnn_config_for_center_state_prior_estimation(
    config_in: returnn.ReturnnConfig,
    *,
    label_info: LabelInfo,
    center_state_softmax_layer="center-output",
    out_center_state_layer="center-state-outputs",
    center_state_batch_size: typing.Optional[int] = None,
) -> returnn.ReturnnConfig:
    """
    Assumes forward decomposition and augments the given returnn config to be
    compatible with ReturnnRasrComputePriorV2Job.

    Returns a config for center state.

    The output layers (as given by function parameters) are reshaped such that
    averaging their outputs directly yields the priors.

    The user needs to run ReturnnRasrComputePriorV2Job with the returned config.
    """

    # Center State
    center_state_config = copy.deepcopy(config_in)
    augment_for_center_state(
        center_state_config,
        label_info,
        center_state_batch_size=center_state_batch_size,
        center_state_softmax_layer=center_state_softmax_layer,
        out_center_state_layer=out_center_state_layer,
        left_context_range=None,
    )

    return center_state_config


def get_returnn_configs_for_right_context_prior_estimation(
    config_in: returnn.ReturnnConfig,
    *,
    label_info: LabelInfo,
    right_context_softmax_layer="right-output",
    out_right_context_layer="right-context-outputs",
    right_context_batch_size: typing.Optional[int] = None,
) -> typing.List[returnn.ReturnnConfig]:
    """
    Assumes forward decomposition and augments the given returnn config to be
    compatible with ReturnnRasrComputePriorV2Job.

    Returns a config for the right context.

    The output layers (as given by function parameters) are reshaped such that
    averaging their outputs directly yields the priors.

    The user needs to run ReturnnRasrComputePriorV2Job. Due to the size of the output,
    for the right context, multiple prior computation jobs need to be run. Their
    output then is joined into the final list of priors via `JoinRightContextPriorsJob`.
    """

    # Right Context takes modified config for center state and augments it further
    right_configs = []
    for left_context in range(label_info.n_contexts):
        right_context_config = copy.deepcopy(config_in)

        l_from, l_to = left_context, left_context + 1
        center_state_spatial, center_state_range = augment_for_center_state(
            right_context_config,
            label_info,
            center_state_batch_size=1,
            center_state_softmax_layer="center-output",
            out_center_state_layer="center-state-outputs",
            left_context_range=(l_from, l_to),
        )

        center_output_layer = right_context_config.config["forward_output_layer"]
        right_context_config.config["extern_data"].pop(center_output_layer)
        right_context_config.config["network"].pop(center_output_layer)

        augment_for_right_context(
            right_context_config,
            label_info,
            right_context_batch_size=right_context_batch_size,
            right_context_softmax_layer=right_context_softmax_layer,
            out_right_context_layer=out_right_context_layer,
            center_state_spatial_dim=center_state_spatial,
        )

        right_configs.append(right_context_config)

    return right_configs
