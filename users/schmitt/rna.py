

def rna_loss(source, **kwargs):
  """
  Computes the RNA loss function.

  :param log_prob:
  :return:
  """
  # acts: (B, T, U, V)
  # targets: (B, U-1)
  # input_lengths (B,)
  # label_lengths (B,)
  import sys, os
  sys.path.insert(0, "/u/schmitt/experiments/transducer/config/returnn_config/functions")
  import tensorflow as tf
  log_probs = source(0, as_data=True, auto_convert=False)
  targets = source(1, as_data=True, auto_convert=False)
  encoder = source(2, as_data=True, auto_convert=False)

  enc_lens = encoder.get_sequence_lengths()
  dec_lens = targets.get_sequence_lengths()

  from rna_tf_impl import tf_forward_shifted_rna
  costs = -tf_forward_shifted_rna(log_probs.get_placeholder_as_batch_major(), targets.get_placeholder_as_batch_major(),
                                  enc_lens, dec_lens, blank_index=eval("targetb_blank_idx"), debug=False)
  costs = tf.where(tf.math.is_finite(costs), costs, tf.zeros_like(costs))
  return costs


def rna_alignment(source, **kwargs):
  """
  Computes the RNA loss function.

  :param log_prob:
  :return:
  """
  # acts: (B, T, U, V)
  # targets: (B, U-1)
  # input_lengths (B,)
  # label_lengths (B,)
  import sys, os
  sys.path.insert(0, "/u/schmitt/experiments/transducer/config/returnn_config/functions")
  log_probs = source(0, as_data=True, auto_convert=False).get_placeholder_as_batch_major()
  targets = source(1, as_data=True, auto_convert=False)
  encoder = source(2, as_data=True, auto_convert=False)

  enc_lens = encoder.get_sequence_lengths()
  dec_lens = targets.get_sequence_lengths()

  # target_len = TFUtil.get_shape_dim(targets.get_placeholder_as_batch_major(), 1)
  # log_probs = TFUtil.check_input_dim(log_probs, 2, target_len+1)
  # enc_lens = tf.Print(enc_lens, ["enc_lens:", enc_lens,
  # "dec_lens:", dec_lens,
  # "targets:", tf.shape(targets.get_placeholder_as_batch_major()), "log-probs:", tf.shape(log_probs.get_placeholder_as_batch_major())], summarize=-1)

  from rna_tf_impl import tf_forward_shifted_rna
  costs, alignment = tf_forward_shifted_rna(log_probs, targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
                                            blank_index=eval("targetb_blank_idx"), debug=False, with_alignment=True)
  return alignment  # (B, T)


def rna_alignment_out(sources, **kwargs):
  from returnn.tf.util.data import Data

  log_probs = sources[0].output
  targets = sources[1].output
  encoder = sources[2].output
  enc_lens = encoder.get_sequence_lengths()
  return Data(name="rna_alignment", sparse=True, dim=eval("targetb_num_labels"), size_placeholder={0: enc_lens})


def rna_loss_out(sources, **kwargs):
  from returnn.tf.util.data  import Data
  return Data(name="rna_loss", shape=())
