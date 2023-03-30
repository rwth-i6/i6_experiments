"""
Functions used directly in the RETURNN network,
handled via get_serializable_config.
"""


def get_chunked_align(source, self, chunk_step, eoc_idx, ignore_indices=None, **_kwargs):
    """
    Gets a time-sync RNA alignment of length T, including L labels
    and blank otherwise.
    It generates a chunked alignment with eoc_idx as chunk separator.
    The output length is K + L, where K is the number of chunks, K = T // chunk_step.
    """
    import tensorflow as tf

    data = source(0, as_data=True)
    blank_idx = data.dim - 1

    @tf.function
    def _f(in_: tf.Tensor, in_sizes: tf.Tensor):
        batch_size = tf.shape(in_)[0]
        batched_ta = tf.TensorArray(dtype=tf.int32, size=batch_size, element_shape=[None])
        batched_ta_seq_lens = tf.TensorArray(dtype=tf.int32, size=batch_size, element_shape=[])

        for b in tf.range(batch_size):
            x = in_[b][: in_sizes[b]]
            ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=[])
            i = 0
            for t in tf.range(in_sizes[b]):
                if tf.logical_and(tf.equal(t % chunk_step, 0), tf.greater(t, 0)):
                    ta = ta.write(i, eoc_idx)
                    i += 1
                if tf.equal(x[t], blank_idx):
                    continue
                if ignore_indices:
                    if tf.reduce_any([tf.equal(x[t], idx) for idx in ignore_indices]):
                        continue
                ta = ta.write(i, x[t])
                i += 1

            ta = ta.write(i, eoc_idx)
            batched_ta = batched_ta.write(b, ta.stack())  # [i]
            batched_ta_seq_lens = batched_ta_seq_lens.write(b, i + 1)

        seq_lens_ = batched_ta_seq_lens.stack()
        # stack batched_ta using padding
        max_len = tf.reduce_max(seq_lens_)
        batched_ta_ = tf.TensorArray(dtype=tf.int32, size=batch_size, element_shape=[None])
        for b in tf.range(batch_size):
            x = batched_ta.read(b)
            batched_ta_ = batched_ta_.write(b, tf.pad(x, [[0, max_len - tf.shape(x)[0]]]))
        return batched_ta_.stack(), seq_lens_

    y, seq_lens = _f(data.placeholder, data.get_sequence_lengths())
    out = self.output
    out.set_dynamic_size(1, seq_lens)
    return y


def get_chunked_align_out_type(sources, **_kwargs):
    """
    For :func:`get_chunked_align`
    """
    from returnn.tf.util.data import Data, batch_dim, SpatialDim, FeatureDim

    dim = sources[0].output.sparse_dim
    dim = FeatureDim("out", dim.dimension - 1)
    out_time_dim = SpatialDim("out-time")
    out_time_dim.dyn_size_ext = Data("out-time:size", dim_tags=[batch_dim], dtype="int32")
    return Data("out", dim_tags=[batch_dim, out_time_dim], sparse_dim=dim)
