import tensorflow as tf


def tensorflow_unique_consecutive_counts(x: tf.constant):
    """ TF2 implementation of torch.unique_consecutive(..., return_counts=True) """
    _, unique_indices = tf.unique(x)

    inversion_indices = tf.concat(
        values=[
            tf.constant([True]),
            tf.not_equal(unique_indices[1:], unique_indices[:-1]),
        ],
        axis=0,
    )

    inversion_indices = tf.experimental.numpy.nonzero(inversion_indices)
    counts = tf.experimental.numpy.diff(inversion_indices)
    return counts


def tensorflow_blank_collapse(logprobs, blank_threshold, blank_idx):
    """
    :param logprobs: softmax-normalized probabilities in log-space, [1, T, V+1]
    :param blank_threshold: collapse threshold probability in log-space
    :param blank_idx: index of blank label, i.e. V+1
    """

    blanks = logprobs[:blank_idx] > tf.math.log(tf.constant([blank_threshold]))
    counts = tensorflow_unique_consecutive_counts(blanks)

    blank_begin, blank_end = blanks[0].numpy(), blanks[-1].numpy()
    initial_blank_cnt = counts[0][0].numpy() if blank_begin else 0
    final_blank_cnt = counts[0][-1].numpy() if blank_end else 0

    initial_slice = initial_blank_cnt
    final_slice = len(blanks) - final_blank_cnt

    blanks = blanks[initial_slice:final_slice]
    blanks_shift = tf.roll(blanks, shift=-1, axis=0)

    collapsed_logrpobs = logprobs[initial_slice:final_slice]
    collapsed_logprobs = collapsed_logrpobs[~(blanks & blanks_shift)]

    return collapsed_logprobs


def tensorflow_blank_collapse_batched(logprobs, audio_features_len, blank_threshold, blank_idx):
    """
    :param logprobs: softmax-normalized probabilities in log-space, [B, T, V+1]
    :param audio_features_len: length of T as [B]
    :param blank_threshold: collapse threshold probability in log-space
    :param blank_idx: index of blank label, i.e. V+1
    """
    batch_dim, time_dim = logprobs.shape[0], logprobs.shape[1]

    # Global candidates for blank collapse pruning
    blanks = logprobs[:, :, blank_idx] > blank_threshold  # [B, T]

    # For batches, adjust individual lengths by mapping paddings to True values in mask
    audio_lens_mask = (
        tf.expand_dims(tf.range(time_dim), axis=0) >= tf.expand_dims(audio_features_len, axis=1)
    )  # [B, T]
    blanks = blanks | audio_lens_mask  # [B, T]

    # Obtain counts on initial and final blank frames
    sequence = tf.where(blanks == True)
    sequence_mask, sequence_indices = sequence[:, 0], sequence[:, 1]  # [T',]
    _, _, sequence_bounds = tf.unique_with_counts(sequence_mask)  # [B, ]

    sequence_bounds = tf.concat(
        [tf.constant([0]), tf.math.cumsum(sequence_bounds)], axis=0
    )  # [B+1, ]

    initial_blank_idx = tf.gather(sequence_indices, sequence_bounds[:-1])  # [B, ]
    final_blank_idx = tf.gather(sequence_indices, (sequence_bounds - 1)[1:])  # [B, ]

    # Logical-and between "blanks" and "blanks_shift" to account for label-blank-label case
    blanks_shift = tf.roll(blanks, shift=-1, axis=1)

    # Logical-or between "(blanks & blanks_shift)" and "bounds_mask" to restore proper lengths
    bounds_mask = tf.tile(
        tf.expand_dims(tf.range(time_dim), axis=0), tf.constant([batch_dim, 1])
    )  # [B, T]
    bounds_mask_initial = bounds_mask < tf.expand_dims(initial_blank_idx, axis=1)  # [B, T]
    bounds_mask_final = bounds_mask > tf.expand_dims(final_blank_idx, axis=1)  # [B, T]
    bounds_mask = bounds_mask_initial | bounds_mask_final  # [B, T]

    # Logical-not to assign True to frames kept
    blanks = ~((blanks & blanks_shift) | bounds_mask)  # [B, T]

    # De-batchify and re-arrange based on changed lengths
    collapsed_sequence = tf.cast(tf.where(blanks == True), tf.int32)
    collapsed_mask, collapsed_indices = collapsed_sequence[:, 0], collapsed_sequence[:, 1]
    _, _, collapsed_audio_features_len = tf.unique_with_counts(collapsed_mask)  # [B, ]

    # Compute new time dimension to restore batching
    collapsed_time_dim = tf.reduce_max(collapsed_lengths).numpy()  # T''

    # Align mask and indices to match the collapsed audio lengths in sorted order
    collapsed_mask = tf.tile(
        tf.expand_dims(tf.range(batch_dim), axis=1),
        tf.constant([1, collapsed_time_dim])
    )  # [B, T'']

    # TODO: how to pad varying-length sequences effectively?
    collapsed_indices = tf.split(collapsed_indices, collapsed_lengths)  # tuple (B, )

    # Added in later TF2 versions
    # tf.keras.utils.pad_sequences(collapsed_indices, padding='post')

    batch_indices = torch.nn.utils.rnn.pad_sequence(
        batch_indices, batch_first=True
    )  # [B, T'']

    # Restore original order within the batch
    collapsed_logprobs = logprobs[collapsed_mask, collapsed_indices]  # [B, T'', V+1]

    return collapsed_logprobs, collapsed_audio_features_len