__all__ = [
    "add_acoustic_scoring",
]

import copy
import typing

import i6_core.returnn as returnn

from ..factored import LabelInfo

if typing.TYPE_CHECKING:
    import tensorflow as tf


def compute_acoustic_scores(
    n_contexts: typing.Union[int, "tf.Tensor"],
    in_classes: "tf.Tensor",
    out_center_state: "tf.Tensor",
    out_left_context: "tf.Tensor",
    out_right_context: "tf.Tensor",
    center_state_priors: "tf.Tensor",  # (L, C)
    left_context_priors: "tf.Tensor",  # (L,)
    right_context_priors: "tf.Tensor",  # (L * C, R)
    center_state_prior_scale: typing.Union[float, "tf.Tensor"],
    left_context_prior_scale: typing.Union[float, "tf.Tensor"],
    right_context_prior_scale: typing.Union[float, "tf.Tensor"],
    center_state_scale: typing.Union[float, "tf.Tensor"] = 1.0,
    left_context_scale: typing.Union[float, "tf.Tensor"] = 1.0,
    right_context_scale: typing.Union[float, "tf.Tensor"] = 1.0,
) -> "tf.Tensor":
    """
    Given the softmax output and classes to score, computes the acoustic scores
    using TensorFlow.

    Assumes the labels that have been fed into the model to compute the left/center/right
    posteriors have their right context factored out and have been deduplicated the
    way `tf.unique` would do it.
    """

    import tensorflow as tf

    in_classes = tf.squeeze(in_classes)
    out_center_state = tf.squeeze(out_center_state)
    out_left_context = tf.squeeze(out_left_context)
    out_right_context = tf.squeeze(out_right_context)

    with tf.name_scope("Labels"):
        right_contexts = tf.math.floormod(in_classes, n_contexts, name="RightContexts")
        center_left_labels = tf.math.floordiv(in_classes, n_contexts, name="CenterLeft")
        # _unique_center_left is what's forwarded
        _unique_center_left, center_left_remap_indices = tf.unique(center_left_labels, name="UniqueCenterLeft")
        left_contexts = tf.math.floormod(center_left_labels, n_contexts, name="LeftContexts")
        center_states = tf.math.floordiv(center_left_labels, n_contexts, name="CenterStates")

    with tf.name_scope("LeftContext"):
        with tf.name_scope("Stack"):
            left_context_posterior_indexes = tf.stack([center_left_remap_indices, left_contexts], axis=1)
            left_context_prior_indexes = tf.expand_dims(left_contexts, axis=-1)

        with tf.name_scope("Gather"):
            selected_left_context_posteriors = tf.gather_nd(
                out_left_context, left_context_posterior_indexes, name="Posterior"
            )
            selected_left_context_priors = tf.gather_nd(left_context_priors, left_context_prior_indexes, name="Prior")

        with tf.name_scope("Score"):
            left_score: tf.Tensor = -(tf.math.log(selected_left_context_posteriors) * left_context_scale) + (
                selected_left_context_priors * left_context_prior_scale
            )

    with tf.name_scope("CenterState"):
        with tf.name_scope("Stack"):
            center_state_posterior_indexes = tf.stack([center_left_remap_indices, center_states], axis=1)
            center_state_prior_indexes = tf.stack([left_contexts, center_states], axis=1)

        with tf.name_scope("Gather"):
            selected_center_state_posteriors = tf.gather_nd(
                out_center_state, center_state_posterior_indexes, name="Posterior"
            )
            selected_center_state_priors = tf.gather_nd(center_state_priors, center_state_prior_indexes, name="Prior")

        with tf.name_scope("Score"):
            center_score: tf.Tensor = -(tf.math.log(selected_center_state_posteriors) * center_state_scale) + (
                selected_center_state_priors * center_state_prior_scale
            )

    with tf.name_scope("RightContext"):
        with tf.name_scope("Stack"):
            right_context_posterior_indexes = tf.stack(
                [center_left_remap_indices, right_contexts],
                axis=1,
            )
            right_context_prior_indexes = tf.stack([center_left_labels, right_contexts], axis=1)

        with tf.name_scope("Gather"):
            selected_right_context_posteriors = tf.gather_nd(
                out_right_context, right_context_posterior_indexes, name="Posterior"
            )
            selected_right_context_priors = tf.gather_nd(
                right_context_priors, right_context_prior_indexes, name="Prior"
            )

        with tf.name_scope("Score"):
            right_score: tf.Tensor = -(tf.math.log(selected_right_context_posteriors) * right_context_scale) + (
                selected_right_context_priors * right_context_prior_scale
            )

    return tf.add_n([left_score, center_score, right_score], name="AcousticScores")


def compute_acoustic_scores_with_placeholders(
    n_contexts: int,
    classes: "tf.Tensor",
    left_contexts: "tf.Tensor",
    center_states: "tf.Tensor",
    right_contexts: "tf.Tensor",
    out_left_ctx: "tf.Tensor",
    out_center_state: "tf.Tensor",
    out_right_ctx: "tf.Tensor",
    left_remap_indices: "tf.Tensor",
    center_left_remap_indices: "tf.Tensor",
) -> "tf.Tensor":
    import tensorflow as tf

    return compute_acoustic_scores(
        n_contexts=tf.compat.v1.placeholder_with_default(
            tf.constant(n_contexts, dtype=tf.int32),
            shape=(),
            name="NContexts",
        ),
        in_classes=classes,
        out_center_state=out_center_state,
        out_left_context=out_left_ctx,
        out_right_context=out_right_ctx,
        center_state_priors=tf.compat.v1.placeholder(tf.float32, name="CenterStatePriors"),
        left_context_priors=tf.compat.v1.placeholder(tf.float32, name="LeftContextPriors"),
        right_context_priors=tf.compat.v1.placeholder(tf.float32, name="RightContextPriors"),
        center_state_prior_scale=tf.compat.v1.placeholder(tf.float32, shape=(), name="CenterStatePriorScale"),
        left_context_prior_scale=tf.compat.v1.placeholder(tf.float32, shape=(), name="LeftContextPriorScale"),
        right_context_prior_scale=tf.compat.v1.placeholder(tf.float32, shape=(), name="RightContextPriorScale"),
        center_state_scale=tf.compat.v1.placeholder_with_default(
            tf.constant(1.0, dtype=tf.float32),
            shape=(),
            name="CenterStateScale",
        ),
        left_context_scale=tf.compat.v1.placeholder_with_default(
            tf.constant(1.0, dtype=tf.float32),
            shape=(),
            name="LeftContextScale",
        ),
        right_context_scale=tf.compat.v1.placeholder_with_default(
            tf.constant(1.0, dtype=tf.float32),
            shape=(),
            name="RightContextScale",
        ),
    )


def unique_nd(t: "tf.Tensor") -> "typing.Tuple[tf.Tensor, tf.Tensor]":
    """
    Computes unique scalars in an n-D tensor.

    Returns the unique scalars and a tensor of the shape of the input tensor that
    contains indexes to remap the scalars back into their original shape.
    """

    import tensorflow as tf

    t1d = tf.reshape(t, shape=(-1,))
    uniq, idx = tf.unique(t1d)
    return uniq, tf.reshape(idx, shape=tf.shape(t))


def add_acoustic_scoring(
    config: returnn.ReturnnConfig,
    label_info: LabelInfo,
    left_ctx_mlp_layers: typing.List[str],
    center_state_mlp_layers: typing.List[str],
    right_ctx_mlp_layers: typing.List[str],
    encoder_output_layer: str,
) -> returnn.ReturnnConfig:
    """
    Extends the given RETURNN config with triphone factored-hybrid acoustic score
    computation operations.

    The code assumes BTF-order and assumes the classes to score are given across the
    batch axis. It will deduplicate across the batch axis as necessary to improve
    performance.

    The MLP layer lists must be in dependency order. The first entry should be the
    first linear layer and the last layer the softmax.
    """

    assert label_info.n_contexts > 0
    assert len(left_ctx_mlp_layers) > 0
    assert len(center_state_mlp_layers) > 0
    assert len(right_ctx_mlp_layers) > 0

    config = copy.deepcopy(config)
    net = config.config["network"]

    # Duplicate MLPs
    mlp_layers = [*left_ctx_mlp_layers, *center_state_mlp_layers, *right_ctx_mlp_layers]
    for layer in mlp_layers:
        net[f"score_{layer}"] = {**net[layer], "reuse_params": layer}

    # Left Context
    left_linear1 = left_ctx_mlp_layers[0]
    left_softmax = left_ctx_mlp_layers[-1]
    net["score_l_slice_batch_1"] = {
        "class": "slice",
        "from": net[left_linear1]["from"],
        "axis": "B",
        "slice_start": 0,
        "slice_end": 1,
    }
    net[f"score_{left_linear1}"]["from"] = "score_l_slice_batch_1"

    # Center State
    center_linear1 = center_state_mlp_layers[0]
    center_softmax = center_state_mlp_layers[-1]

    # Safety assertions to ensure we don't break weights
    assert list(net[center_linear1]["from"]) == [encoder_output_layer, "pastEmbed"]

    net["score_c_unique_left"] = {
        "class": "eval",
        "from": "pastLabel",
        "eval": "self.network.get_config().typed_value('unique_nd')(source(0))",
    }
    net["score_c_unique_left_unique"] = {
        "class": "eval",
        "from": "score_c_unique_left",
        "eval": "source(0)[0]",
    }
    net["score_c_unique_left_remap"] = {
        "class": "eval",
        "from": "score_c_unique_left",
        "eval": "source(0)[1]",
    }
    net["score_c_pastEmbed"] = {
        **net["pastEmbed"],
        "from": "score_c_unique_left_unique",
        "reuse_params": "pastEmbed",
    }

    net["score_c_left_broadcast_encoder_output"] = {
        "class": "eval",
        "from": [encoder_output_layer, "score_c_pastEmbed"],
        "eval": "tf.broadcast_to(source(0), shape=tf.shape(source(1)))",
    }
    net[f"score_{center_linear1}"]["from"] = [
        encoder_output_layer,
        "score_c_pastEmbed",
    ]

    # Right Context
    right_linear1 = right_ctx_mlp_layers[0]
    right_softmax = right_ctx_mlp_layers[-1]

    # Safety assertions to ensure we don't break weights
    assert list(net[right_linear1]["from"]) == [
        encoder_output_layer,
        "currentState",
        "pastEmbed",
    ]

    net["score_r_unique_center_left"] = {
        "class": "eval",
        "from": "popFutureLabel",
        "eval": "self.network.get_config().typed_value('unique_nd')(source(0))",
    }
    net["score_r_unique_center_left_unique"] = {
        "class": "eval",
        "from": "score_r_unique_center_left",
        "eval": "source(0)[0]",
    }
    net["score_r_unique_center_left_remap"] = {
        "class": "eval",
        "from": "score_r_unique_center_left",
        "eval": "source(0)[1]",
    }
    net["score_r_pastLabel"] = {
        **net["pastLabel"],
        "from": "score_r_unique_center_left_unique",
        "register_as_extern_data": None,
    }
    net["score_r_pastEmbed"] = {
        **net["pastEmbed"],
        "from": "score_r_pastLabel",
        "reuse_params": "pastEmbed",
    }
    net["score_r_centerState"] = {
        **net["centerState"],
        "from": "score_r_unique_center_left_unique",
        "register_as_extern_data": None,
    }
    net["score_r_currentState"] = {
        **net["currentState"],
        "from": "score_r_centerState",
        "reuse_params": "currentState",
    }

    net["score_r_broadcast_encoder_output"] = {
        "class": "eval",
        "from": [encoder_output_layer, "score_r_unique_center_left_unique"],
        "eval": "tf.broadcast_to(source(0), shape=tf.shape(source(1)))",
    }
    net[f"score_{right_linear1}"]["from"] = [
        encoder_output_layer,
        "score_r_currentState",
        "score_r_pastEmbed",
    ]

    net["out_acoustic_scores"] = {
        "class": "eval",
        "eval": f"self.network.get_config().typed_value('compute_acoustic_scores_with_placeholders')({label_info.n_contexts}, *(source(i) for i in range(9)))",
        "from": [
            "data:classes",
            "data:pastLabel",
            "data:centerState",
            "data:futureLabel",
            f"score_{left_softmax}",
            f"score_{center_softmax}",
            f"score_{right_softmax}",
            "score_c_unique_left_remap",
            "score_r_unique_center_left_remap",
        ],
    }

    config.python_epilog = [
        config.python_epilog,
        compute_acoustic_scores,
        compute_acoustic_scores_with_placeholders,
        unique_nd,
    ]

    raise NotImplementedError()

    return config
