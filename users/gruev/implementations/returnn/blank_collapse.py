import math
import i6_experiments.users.gruev.returnn_helpers as layers


def blank_collapse(network, softmax_layer, blank_threshold, blank_idx, apply_log=False):
    """
    :param network: network on which to perform blank collapse
    :param softmax_layer: softmax-normalized probabilities, dims=[B, T, F]
    :param blank_threshold:
    :param blank_idx:
    :param apply_log: adjust for ctc-softmax output
    """

    if is_logspace:
        blank_threshold = math.log(blank_threshold)

    ### INITIAL BLANK THRESHOLDING
    # computes softmax_layer_output[:, :, blank_idx], output_dims=[B, T]
    network, blank_axis = layers.add_gather_layer(network, "blank_axis", softmax_layer, axis="F", position=0)
    # computes a mask, softmax_layer_output[:, :, blank_idx] > blank_threshold, output_dims=[B, T]
    network, blank_mask = layers.add_combine_layer(network, "blank_mask", blank_axis, kind="greater")

    ### INTEGRATION OF AUDIO LENGTHS
    # computes audio lengths in batch, output_dims=[B,]
    network, audio_lengths = layers.add_length_layer(network, "audio_lengths", blank_mask, axis="T")
    # computes a range [0, ..., max(audio_lengths)-1], output_dims=[T,]
    network, audio_lengths_range = layers.add_range_from_length_layer(network, "audio_lengths_range", audio_lengths)
    # computes a mask for individual audio lengths, output_dims=[B, T]
    network, audio_lengths_mask = layers.add_compare_layer(
        network, "audio_lengths_mask", [audio_lengths, audio_lengths_range], kind="less_equal"
    )
    # computes a mask, takes individual audio lengths into account, output_dims=[B, T]
    network, blank_mask_w_audio_lengths = layers.add_combine_layer(
        network, "blank_mask_w_audio_lengths", [blank_mask, audio_lengths_mask], kind="logical_or"
    )

    ### OMISSION OF INITIAL BLANKS
    # computes a map, True is mapped to +inf, False is mapped to [0, 1, 2, ...], output_dims=[B, T]
    network, mapping = layers.add_switch_layer(
        network, "mapping", condition=blank_mask_w_audio_lengths, true_from=int(1e4), false_from=audio_lengths_range
    )
    # computes indices of first non-blank elements, output_dims=[B,]
    network, start_indices = layers.add_reduce_layer(network, "start_indices", mapping, axis="T", mode="argmin")
    # computes a mask, masks initial blanks for each sequence, output_dims=[B, T]
    network, start_indices_mask = layers.add_compare_layer(
        network, "start_indices_mask", [audio_lengths_range, start_indices], kind="less"
    )

    ### COMBINATION OF BLANK MASK, AUDIO LENGTHS MASK, AND INITIAL BLANKS MASK
    # computes a left-shifted mask, output_dims=[B, T]
    network, blank_mask_w_audio_lengths_shifted = layers.add_shift_layer(
        network,
        "blank_mask_w_audio_lengths_shifted",
        blank_mask_w_audio_lengths,
        axis="T",
        amount=-1,
        pad=True,
        pad_value=True,
        adjust_size=False,
    )
    # computes a mask, blank_mask_w_audio_lengths & blank_mask_w_audio_lengths_shifted, output_dims=[B, T]
    network, blank_mask_w_shift = layers.add_combine_layer(
        network,
        "blank_mask_w_shift",
        [blank_mask_w_audio_lengths, blank_mask_w_audio_lengths_shifted],
        kind="logical_and",
    )
    # computes a mask, blank_mask_w_shift | start_indices_mask, output_dims=[B, T]
    network, blank_mask_inverted = layers.add_combine_layer(
        network, "blank_mask_inverted", [blank_mask_w_shift, start_indices_mask], kind="logical_or"
    )
    # computes a mask, ~blank_mask_inverted, output_dims=[B, T]
    network, blank_mask_final = layers.add_eval_layer(
        network, "blank_mask_final", [blank_mask_inverted], eval_str="tf.math.logical_not(source(0))"
    )

    ### BLANK COLLAPSE
    network, blank_collapse = layers.add_mask_layer(network, "blank_collapse", softmax_layer, mask=blank_mask_final)
