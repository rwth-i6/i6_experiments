
compressed_concat_code = """
import tensorflow as tf
from returnn.tf.layers.basic import _ConcatInputLayer, register_layer_class
from returnn.tf.util.data import Data

class CompressedConcatLayer(_ConcatInputLayer):
  layer_class = "compressed_concat"

  def __init__(self, sources, **kwargs):
    super().__init__(**kwargs)

    enc = sources[0].output
    dec = sources[1].output
    enc_placeholder = enc.get_placeholder_as_batch_major()
    dec_placeholder = dec.get_placeholder_as_batch_major()
    enc_lens = enc.get_sequence_lengths()
    dec_lens = dec.get_sequence_lengths()

    B = tf.shape(enc_placeholder)[0]
    F1 = enc_placeholder.shape[-1]
    F2 = dec_placeholder.shape[-1]
    
    b = tf.constant(0)
    out = tf.TensorArray(dtype=tf.float32, size=B, dynamic_size=False, element_shape=[None, F1+F2], infer_shape=False, clear_after_read=True)

    cond = lambda b, _: tf.less(b, B)
    def body(b, out):
      enc_padded = tf.pad(enc_placeholder[b, :enc_lens[b]], [(0, 0), (0, F2)], mode='CONSTANT', constant_values=0)  # (T, F1+F2)
      dec_padded = tf.pad(dec_placeholder[b, :dec_lens[b]], [(0, 0), (F1, 0)], mode='CONSTANT', constant_values=0)  # (U, F1+F2)
      enc_expand = tf.expand_dims(enc_padded, 1)  # (T, 1, F1+F2)
      dec_expand = tf.expand_dims(dec_padded, 0)  # (1, U, F2+F2)
      enc_dec_concat = enc_expand + dec_expand  # (T, U, F1+F2)
      enc_dec_reshaped = tf.reshape(enc_dec_concat, [enc_lens[b] * dec_lens[b], F1 + F2])  # (T*U, F1+F2)

      return b+1, out.write(b, enc_dec_reshaped)

    b, out = tf.while_loop(cond, body, [b, out])
    out = out.concat()
 
    out_size = {0: tf.stack([tf.tensordot(enc_lens, dec_lens, axes=1), F1+F2], axis=0)}

    self.output.placeholder = out
    self.output.size_placeholder = out_size

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    assert len(sources) == 2
    return Data(
      name="%s_output" % name,
      shape=(None, sources[0].output.dim + sources[1].output.dim),
      dtype="float32",
      batch_dim_axis=None,
      time_dim_axis=0,
      feature_dim_axis=1)

register_layer_class(CompressedConcatLayer)
"""

compressed_add_code = """
import tensorflow as tf
from returnn.tf.layers.basic import _ConcatInputLayer, register_layer_class
from returnn.tf.util.data import Data

class CompressedAddLayer(_ConcatInputLayer):
  layer_class = "compressed_add"

  def __init__(self, sources, **kwargs):
    super().__init__(**kwargs)

    enc = sources[0].output
    dec = sources[1].output
    enc_placeholder = enc.get_placeholder_as_batch_major()
    dec_placeholder = dec.get_placeholder_as_batch_major()
    enc_lens = enc.get_sequence_lengths()
    dec_lens = dec.get_sequence_lengths()

    B = tf.shape(enc_placeholder)[0]
    F = enc_placeholder.shape[-1]
    
    b = tf.constant(0)
    out = tf.TensorArray(dtype=tf.float32, size=B, dynamic_size=False, element_shape=[None, F], infer_shape=False, clear_after_read=True)
    cond = lambda b, _: tf.less(b, B)
    body = lambda b, out: [b+1, out.write(b, tf.reshape(tf.expand_dims(enc_placeholder[b, :enc_lens[b]], 1) + tf.expand_dims(dec_placeholder[b, :dec_lens[b]], 0), [enc_lens[b] * dec_lens[b], F]))]
    b, out = tf.while_loop(cond, body, [b, out])
    out = out.concat()
    
    out_size = {0: tf.stack([tf.tensordot(enc_lens, dec_lens, axes=1), F], axis=0)}

    self.output.placeholder = out
    self.output.size_placeholder = out_size

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    return Data(
      name="%s_output" % name,
      shape=(None, sources[0].output.placeholder.shape[-1]),
      dtype="float32",
      batch_dim_axis=None,
      time_dim_axis=0,
      feature_dim_axis=1)

register_layer_class(CompressedAddLayer)
"""

compressed_multiply_code = """
import tensorflow as tf
from returnn.tf.layers.basic import _ConcatInputLayer, register_layer_class
from returnn.tf.util.data import Data

class CompressedMultiplyLayer(_ConcatInputLayer):
  layer_class = "compressed_multiply"

  def __init__(self, sources, **kwargs):
    super().__init__(**kwargs)

    enc = sources[0].output
    dec = sources[1].output
    enc_placeholder = enc.get_placeholder_as_batch_major()
    dec_placeholder = dec.get_placeholder_as_batch_major()
    enc_lens = enc.get_sequence_lengths()
    dec_lens = dec.get_sequence_lengths()

    B = tf.shape(enc_placeholder)[0]
    F = enc_placeholder.shape[-1]
    
    b = tf.constant(0)
    out = tf.TensorArray(dtype=tf.float32, size=B, dynamic_size=False, element_shape=[None, F], infer_shape=False, clear_after_read=True)
    cond = lambda b, _: tf.less(b, B)
    body = lambda b, out: [b+1, out.write(b, tf.reshape(tf.expand_dims(enc_placeholder[b, :enc_lens[b]], 1) * tf.expand_dims(dec_placeholder[b, :dec_lens[b]], 0), [enc_lens[b] * dec_lens[b], F]))]
    b, out = tf.while_loop(cond, body, [b, out])
    out = out.concat()
    
    out_size = {0: tf.stack([tf.tensordot(enc_lens, dec_lens, axes=1), F], axis=0)}

    self.output.placeholder = out
    self.output.size_placeholder = out_size

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    return Data(
      name="%s_output" % name,
      shape=(None, sources[0].output.placeholder.shape[-1]),
      dtype="float32",
      batch_dim_axis=None,
      time_dim_axis=0,
      feature_dim_axis=1)

register_layer_class(CompressedMultiplyLayer)
"""
