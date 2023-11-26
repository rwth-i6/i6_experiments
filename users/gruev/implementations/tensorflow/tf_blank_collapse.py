import tensorflow


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