import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import contextlib
import ast
from collections import Counter

from i6_experiments.users.schmitt.hdf import load_hdf_data
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob

from sisyphus import Path


def shift_alignment_boundaries_v1(data, blank_idx, max_shift, network):
  """
  We take as input a batch of alignments (B, T) and output a batch of new alignments by randomly moving the alignment
  boundaries with a maximum shift of `max_shift`.
  This was my first implementation, which led to errors in RETURNN and which had a bias.
  :param data:
  :param blank_idx:
  :param max_shift:
  :param network:
  :return:
  """
  import tensorflow as tf

  x = data.get_placeholder_as_batch_major()
  seq_lens = data.get_sequence_lengths()
  max_shift = tf.constant(max_shift)

  @tf.function
  def get_augmented_alignment():
    # the batch with the new alignments
    new_data = tf.TensorArray(
      tf.int32,
      size=tf.shape(x)[0],
      dynamic_size=True,
      clear_after_read=True
    )

    i = 0

    # go over each alignment
    for alignment in x:
      # positions of non-blank labels in the original alignment
      non_blank_pos = tf.cast(tf.where(tf.not_equal(alignment, blank_idx)), dtype=tf.dtypes.int32)
      nb_labels = alignment[tf.not_equal(alignment, blank_idx)]
      # prepend -1 and append seq_len of alignment to the positions
      non_blank_pos_ext = tf.concat((tf.constant([[-1]]), non_blank_pos, [[seq_lens[i]]]), axis=0)
      # from these extended positions, we get the amount of left and right space of each label
      right_spaces = non_blank_pos_ext[2:] - non_blank_pos_ext[1:-1] - 1
      left_spaces = non_blank_pos_ext[1:-1] - non_blank_pos_ext[:-2] - 1
      # the maximum shifting amount is then determined by the amount of space and the max_shift parameter
      max_left_shift = tf.where(tf.greater(left_spaces, max_shift), max_shift, left_spaces)
      max_right_shift = tf.where(tf.greater(right_spaces, max_shift), max_shift, right_spaces)

      # concat the tensors in order to loop over them
      nb_datas = tf.concat(
        (
          non_blank_pos,
          max_left_shift,
          max_right_shift
        ),
        axis=-1)

      # create an array of new positions
      new_non_blank_pos = tf.TensorArray(
        tf.int32,
        size=tf.shape(non_blank_pos)[0],
        dynamic_size=True,
        clear_after_read=True
      )

      # loop index
      j = 0
      # store prev pos
      prev_pos = tf.TensorArray(
        tf.int32,
        size=1,
        dynamic_size=True,
        clear_after_read=True
      )
      for nb_data in nb_datas:
        # the min and max value of the new position is given by the current position and the maximum left and right shift
        minval = nb_data[0] - nb_data[1]
        maxval = nb_data[0] + nb_data[2]

        # if we are past the first element, the left boundary is additionally constrained by the new position of
        # the previous label
        if j > 0:
          minval = tf.maximum(minval, prev_pos.read(0) + 1)
        # the new position is uniformly sampled within the possible boundaries
        new_pos = tf.random.uniform((1,), minval=minval, maxval=maxval + 1, dtype=tf.dtypes.int32)
        prev_pos = prev_pos.write(0, tf.squeeze(new_pos))
        new_non_blank_pos = new_non_blank_pos.write(j, new_pos)
        j += 1

      # stack the new positions into an array
      new_non_blank_pos = new_non_blank_pos.stack()

      # scatter the labels into the new alignment according to the new positions
      output_shape = tf.expand_dims(tf.shape(alignment)[0], axis=0)
      # add 1 to the labels here because a label might have value 0 which is the same as the default value in scatter
      new_alignment = tf.scatter_nd(new_non_blank_pos, nb_labels + 1, output_shape)
      # subtract 1 again from the labels in the final alignment and replace the 0's with the blank idx
      new_alignment = tf.where(tf.equal(new_alignment, 0), blank_idx, new_alignment - 1)

      new_data = new_data.write(i, new_alignment)
      i += 1

    return new_data.stack()

  return get_augmented_alignment()


def shift_alignment_boundaries_v2(alignment: np.ndarray, blank_idx: int, max_shift: int):
  alignment = tf.convert_to_tensor(alignment)
  labels = alignment[tf.not_equal(alignment, blank_idx)]
  label_positions = tf.cast(tf.where(tf.not_equal(alignment, blank_idx)), tf.int32)[:, 0]  # [S]
  # prepend -1 and append seq_len of alignment to the positions to calculate the distances to the boundaries
  label_positions_ext = tf.concat(([-1], label_positions, [alignment.shape[0] - 1]), axis=0)  # [S+2]
  # for each label, the distance to the next label on the right and left (or to the sequence boundaries)
  distance_right = label_positions_ext[2:] - label_positions - 1  # [S]
  distance_left = label_positions - label_positions_ext[:-2] - 1  # [S]
  # find the uneven distances
  distance_right_uneven = distance_right[:-1] % 2 != 0  # [S-1]
  distance_left_uneven = distance_left[1:] % 2 != 0  # [S-1]
  num_uneven = tf.reduce_sum(tf.cast(distance_right_uneven, tf.int32))
  assert tf.math.reduce_all(distance_right_uneven == distance_left_uneven), "Uneven distances are not equal."
  # randomly choose whether to ceil or floor the uneven distances
  random_ceil = tf.random.uniform((num_uneven,), minval=0, maxval=2, dtype=tf.int32)
  random_ceil_right = tf.tensor_scatter_nd_update(
    tf.zeros_like(distance_right[1:], dtype=tf.int32), tf.where(distance_right_uneven), random_ceil)
  random_ceil_left = tf.tensor_scatter_nd_update(
    tf.ones_like(distance_left[:-1], dtype=tf.int32), tf.where(distance_left_uneven), random_ceil)
  # can at most use half of the space to the right except for the last (or first) label
  # use floor division in both cases and then apply random ceil
  distance_right = tf.concat((distance_right[:-1] // 2, [distance_right[-1]]), axis=0)  # [S]
  distance_left = tf.concat(([distance_left[0]], distance_left[1:] // 2), axis=0)  # [S]
  distance_right += tf.concat((random_ceil_right, [0]), axis=0)
  distance_left += tf.concat(([0], 1 - random_ceil_left), axis=0)
  # random shift is either max_shift or the available space to the right or left
  random_shift = tf.random.uniform(
    (tf.shape(label_positions)[0],), minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32)  # [S]
  random_shift = tf.clip_by_value(random_shift, -distance_left, distance_right)  # TODO bias problem to boundaries
  # new positions are the old positions plus the random shift
  new_positions = label_positions + random_shift
  assert tf.reduce_all(new_positions[1:] > new_positions[:-1]), "New positions are not sorted anymore."
  # scatter the labels into the new alignment according to the new positions
  new_alignment = tf.fill(alignment.shape, blank_idx)
  new_alignment = tf.tensor_scatter_nd_update(new_alignment, tf.expand_dims(new_positions, axis=1), labels)
  return new_alignment.numpy()


shift_alignment_boundaries_func_str = """
def shift_alignment_boundaries(data, network):
  import tensorflow as tf

  blank_idx = {blank_idx}
  max_shift = {max_shift}

  x = data.get_placeholder_as_batch_major()
  seq_lens = data.get_sequence_lengths()
  max_shift = tf.constant(max_shift)

  @tf.function
  def get_augmented_alignment():
    # the batch with the new alignments
    new_data = tf.TensorArray(
      tf.int32,
      size=tf.shape(x)[0],
      dynamic_size=True,
      clear_after_read=True
    )

    i = 0

    # go over each alignment
    for alignment in x:
      # positions of non-blank labels in the original alignment
      non_blank_pos = tf.cast(tf.where(tf.not_equal(alignment, blank_idx)), dtype=tf.dtypes.int32)
      nb_labels = alignment[tf.not_equal(alignment, blank_idx)]
      # prepend -1 and append seq_len of alignment to the positions
      non_blank_pos_ext = tf.concat((tf.constant([[-1]]), non_blank_pos, [[seq_lens[i]]]), axis=0)
      # from these extended positions, we get the amount of left and right space of each label
      right_spaces = non_blank_pos_ext[2:] - non_blank_pos_ext[1:-1] - 1
      left_spaces = non_blank_pos_ext[1:-1] - non_blank_pos_ext[:-2] - 1
      # the maximum shifting amount is then determined by the amount of space and the max_shift parameter
      max_left_shift = tf.where(tf.greater(left_spaces, max_shift), max_shift, left_spaces)
      max_right_shift = tf.where(tf.greater(right_spaces, max_shift), max_shift, right_spaces)

      # concat the tensors in order to loop over them
      nb_datas = tf.concat(
        (
          non_blank_pos,
          max_left_shift,
          max_right_shift
        ),
        axis=-1)

      # create an array of new positions
      new_non_blank_pos = tf.TensorArray(
        tf.int32,
        size=tf.shape(non_blank_pos)[0],
        dynamic_size=True,
        clear_after_read=True
      )

      # loop index
      j = 0
      # store prev pos
      prev_pos = tf.TensorArray(
        tf.int32,
        size=1,
        dynamic_size=True,
        clear_after_read=True
      )
      for nb_data in nb_datas:
        # the min and max value of the new position is given by the current position and the maximum left and right shift
        minval = nb_data[0] - nb_data[1]
        maxval = nb_data[0] + nb_data[2]

        # if we are past the first element, the left boundary is additionally constrained by the new position of
        # the previous label
        if j > 0:
          minval = tf.maximum(minval, prev_pos.read(0) + 1)
        # the new position is uniformly sampled within the possible boundaries
        new_pos = tf.random.uniform((1,), minval=minval, maxval=maxval + 1, dtype=tf.dtypes.int32)
        prev_pos = prev_pos.write(0, tf.squeeze(new_pos))
        new_non_blank_pos = new_non_blank_pos.write(j, new_pos)
        j += 1

      # stack the new positions into an array
      new_non_blank_pos = new_non_blank_pos.stack()

      # scatter the labels into the new alignment according to the new positions
      output_shape = tf.expand_dims(tf.shape(alignment)[0], axis=0)
      # add 1 to the labels here because a label might have value 0 which is the same as the default value in scatter
      new_alignment = tf.scatter_nd(new_non_blank_pos, nb_labels + 1, output_shape)
      # subtract 1 again from the labels in the final alignment and replace the 0's with the blank idx
      new_alignment = tf.where(tf.equal(new_alignment, 0), blank_idx, new_alignment - 1)

      new_data = new_data.write(i, new_alignment)
      i += 1

    return new_data.stack()

  return get_augmented_alignment()
"""


@contextlib.contextmanager
def make_scope():
  """
  :rtype: tf.compat.v1.Session
  """

  import returnn.tf.compat as tf_compat
  with tf.Graph().as_default() as graph:
    with tf_compat.v1.Session(graph=graph) as session:
      yield session


def test_alignment_augmentation_in_returnn():
  from returnn.config import Config
  from returnn.tf.network import TFNetwork
  import returnn.tf.compat as tf_compat

  with make_scope() as session:
    config = Config()
    config.update({
      "shift_alignment_boundaries": shift_alignment_boundaries_v1,
      "extern_data": {
        "data": {
          "dim": 10,
          "sparse": True
        }
      },
      "network": {
        "augmented_align": {
          "class": "eval",
          "from": "data:data",
          "eval": "self.network.get_config().typed_value('shift_alignment_boundaries')(source(0, as_data=True), blank_idx=0, max_shift=4, network=self.network)"
        },
        "output": {"class": "copy", "from": ["augmented_align"]},
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))

    session.run(tf_compat.v1.global_variables_initializer())
    out = network.layers["output"].output.placeholder
    n_batch = 2
    seq_len = 21
    input_data = np.array([
      [0, 0, 0, 4, 0, 0, 8, 0, 2, 0, 0, 0, 0, 6, 0, 0, 0, 0, 3, 0, 0],
      [0, 7, 0, 6, 2, 0, 0, 7, 0, 6, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0]
    ],
      dtype="int32")
    seq_lens = np.array([seq_len, seq_len], dtype="int32")
    assert input_data.shape == (n_batch, seq_lens[0])
    feed = {network.extern_data.data["data"].placeholder: input_data,
            network.extern_data.data["data"].size_placeholder[0]: seq_lens,
            # network.extern_data.data["seq_tag"].placeholder: input_tags
            }

    print("input: ", input_data)
    out = session.run([out, network.get_post_control_dependencies()], feed_dict=feed)

    print("output: ", out)

    network.call_graph_reset_callbacks()


def calculate_shift_statistics_over_corpus(max_shift, num_iterations, alignment_data, blank_idx):
  label_position_diff_counter = Counter()
  for i, (seq_tag, alignment_old) in enumerate(alignment_data.items()):
    if i % 100 == 0:
      print(i)

    alignment_new = alignment_old.copy()
    for _ in range(num_iterations):
      alignment_new = shift_alignment_boundaries_v2(alignment_new, blank_idx=blank_idx, max_shift=max_shift)

    label_positions_old = np.argwhere(alignment_old != blank_idx)[:, 0]
    label_positions_new = np.argwhere(alignment_new != blank_idx)[:, 0]
    label_position_diffs = label_positions_new - label_positions_old
    label_position_diff_counter.update(label_position_diffs)

  plt.bar(label_position_diff_counter.keys(), label_position_diff_counter.values())
  plt.title(f"Label position differences for \nmax_shift={max_shift} and {num_iterations} iterations.")
  plt.show()
  plt.close()


def compare_alignments(seq_tag, alignment_data, vocab, blank_idx, max_shift, num_iterations):
  alignment_old = alignment_data[seq_tag]
  alignment_new = alignment_old.copy()
  for _ in range(num_iterations):
    alignment_new = shift_alignment_boundaries_v2(alignment_new, blank_idx=blank_idx, max_shift=max_shift)

  labels_old = alignment_old[alignment_old != blank_idx]
  labels_new = alignment_new[alignment_new != blank_idx]

  fig, ax_old = PlotAlignmentJob._get_fig_ax(alignment_old)
  PlotAlignmentJob._set_ticks(ax_old, alignment_old, labels_old, vocab, blank_idx, ymin=0.5)
  ax_new = ax_old.twiny()
  PlotAlignmentJob._set_ticks(ax_new, alignment_new, labels_new, vocab, blank_idx, ymax=0.5, color="g")
  plt.title(f"Comparison of alignment for {seq_tag} \nwith max_shift={max_shift} and {num_iterations} iterations.")
  plt.show()
  plt.close()


def test_alignment_augmentation(hdf_path: str):
  # load vocabulary as dictionary
  with open("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab", "r") as f:
    json_data = f.read()
    vocab = ast.literal_eval(json_data)  # label -> idx
    vocab = {v: k for k, v in vocab.items()}  # idx -> label
  blank_idx = 10025
  alignment_data = load_hdf_data(Path(hdf_path))

  max_shift = 5
  num_iterations = 5
  calculate_shift_statistics_over_corpus(
    max_shift=max_shift, num_iterations=num_iterations, alignment_data=alignment_data, blank_idx=blank_idx)
  compare_alignments(
    seq_tag="dev-other/4570-56594-0002/4570-56594-0002",
    alignment_data=alignment_data,
    vocab=vocab,
    blank_idx=blank_idx,
    max_shift=max_shift,
    num_iterations=num_iterations
  )


if __name__ == "__main__":
  test_alignment_augmentation("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2021_22/i6_core/returnn/forward/ReturnnForwardJob.1fohfY7LLczN/output/alignments.hdf")
