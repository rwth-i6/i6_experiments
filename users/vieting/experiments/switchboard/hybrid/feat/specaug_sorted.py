def sort_filters_by_center_freq(x):
    """
    This function returns the indices that would sort the filters.
    :param tf.Tensor x: The filter layer to sort.
    :return: Sorted indices.
    """
    from returnn.tf.compat import v1 as tf
    import numpy as np

    x = tf.convert_to_tensor(x)  # (N, 1, C) where N is the filter size and C the number of channels
    # implementation similar to scipy.signal.freqz, which uses numpy.polynomial.polynomial.polyval
    filters = tf.transpose(tf.squeeze(x))  # (C, N)
    num_freqs = 512  # F
    w = tf.linspace(0.0, np.pi - np.pi / num_freqs, num_freqs)  # (F,)
    zm1 = tf.expand_dims(tf.exp(-1j * tf.cast(w, "complex64")), 1)  # (F, 1)
    exponents = tf.expand_dims(tf.range(tf.shape(filters)[1]), 0)  # (1, N)
    zm1_pow = tf.pow(zm1, tf.cast(exponents, dtype="complex64"))  # (F, N)
    f_resp = tf.tensordot(zm1_pow, tf.cast(tf.transpose(filters), dtype="complex64"), axes=1)  # (F, C)
    f_resp = tf.abs(f_resp)

    # sorted by increasing center frequency
    center_freqs = tf.argmax(f_resp, axis=0)
    sorted_idcs = tf.argsort(center_freqs)

    return sorted_idcs


def _mask(x, batch_axis, axis, pos, max_amount, sorted_indices=None):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
    :param tf.Tensor|None sorted_indices: sorted indices (optional)
    """
    from returnn.tf.compat import v1 as tf

    ndim = x.get_shape().ndims
    n_batch = tf.shape(x)[batch_axis]
    dim = tf.shape(x)[axis]
    amount = tf.random_uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
    pos2 = tf.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
    if batch_axis > axis:
        cond = tf.transpose(cond)  # (dim,batch)
    cond = tf.reshape(cond, [tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)])
    if sorted_indices is not None:
        inverse_permutation = tf.argsort(sorted_indices)
        cond = tf.gather(cond, inverse_permutation, axis=axis)   
    from TFUtil import where_bc
    x = where_bc(cond, 0.0, x)
    return x


def _random_mask(x, batch_axis, axis, min_num, max_num, max_dims, sorted_indices=None):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
    :param tf.Tensor|None sorted_indices: sorted indices (optional)
    """
    from returnn.tf.compat import v1 as tf

    n_batch = tf.shape(x)[batch_axis]
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
    else:
        num = tf.random_uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.log(-tf.log(tf.random_uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
    _, indices = tf.nn.top_k(z, num if isinstance(num, int) else tf.reduce_max(num))
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    if isinstance(num, int):
        for i in range(num):
            x = _mask(
                x,
                batch_axis=batch_axis,
                axis=axis,
                pos=indices[:, i],
                max_amount=max_dims,
                sorted_indices=sorted_indices,
            )
    else:
        _, x = tf.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
            body=lambda i, x: (
                i + 1,
                tf.where(
                    tf.less(i, num),
                    _mask(
                        x,
                        batch_axis=batch_axis,
                        axis=axis,
                        pos=indices[:, i],
                        max_amount=max_dims,
                        sorted_indices=sorted_indices,
                    ),
                    x,
                ),
            ),
            loop_vars=(0, x),
        )
    return x


def specaugment_eval_func(data, network, mask_divisor=5, time_factor=1):
    x = data.placeholder
    from returnn.tf.compat import v1 as tf

    filter_layer = network.layers["conv_h_filter"].output.placeholder
    sorted_idce = sort_filters_by_center_freq(filter_layer)
    # summary("features", x)
    step = network.global_train_step
    step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
    step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)

    def get_masked():

        x_masked = _random_mask(
            x,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=step1 + step2,
            max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // 100, 2) * (1 + step1 + step2 * 2),
            max_dims=20 // time_factor,
        )

        # Apply freq masking
        x_masked = _random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=step1 + step2,
            max_num=2 + step1 + step2 * 2,
            max_dims=data.dim // mask_divisor,
            sorted_indices=sorted_idce,
        )
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


def specaug_layer_sorted(in_layer, mask_divisor=None):
    """
    specaug layer with default hybrid settings

    :param in_layer:
    :param mask_divisor:
    """
    if mask_divisor is not None:
        return {
            "class": "eval",
            "from": in_layer,
            "eval": "self.network.get_config().typed_value('specaugment_eval_func')("
            "source(0, as_data=True, auto_convert=False),"
            "network=self.network,"
            "mask_divisor="+str(mask_divisor)+")",
        }
    else:
        return {
            "class": "eval",
            "from": in_layer,
            "eval": "self.network.get_config().typed_value('specaugment_eval_func')("
            "source(0, as_data=True, auto_convert=False),"
            "network=self.network)",
        }


def get_funcs_sorted():
    funcs = []
    for k, v in list(globals().items()):
        if k in ["sort_filters_by_center_freq", "_mask", "_random_mask", "specaugment_eval_func"]:
            funcs.append(v)
    return funcs
