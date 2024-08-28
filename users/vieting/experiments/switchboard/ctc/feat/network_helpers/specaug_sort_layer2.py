from typing import Dict, Union, Optional, List


def add_specaug_layer(
    network: Dict,
    name: str = "specaug",
    from_list: Optional[Union[str, List[str]]] = None,
    max_time_num: int = 3,
    max_time: int = 10,
    max_feature_num: int = 4,
    max_feature: int = 5,
) -> List[str]:
    if from_list is None:
        from_list = ["data"]
    network[name] = {
        "class": "eval",
        "from": from_list,
        "eval": f'self.network.get_config().typed_value("transform")(source(0, as_data=True), max_time_num={max_time_num}, max_time={max_time}, max_feature_num={max_feature_num}, max_feature={max_feature}, network=self.network)',
    }

    return [name], get_specaug_funcs()


def get_frequency_response(x):
    """
    This function returns the frequency response of the filters.
    :param tf.Tensor x: The filter layer to sort.
    :return: Frequency response of the filters.
    """
    import numpy as np
    import tensorflow as tf

    x = tf.convert_to_tensor(x)  # (256, 1, 150) = (N, 1, C)

    # implementation similar to scipy.signal.freqz, which uses numpy.polynomial.polynomial.polyval
    filters = tf.transpose(tf.squeeze(x))  # (C, N)
    num_freqs = 128  # F
    w = tf.linspace(0.0, np.pi - np.pi / num_freqs, num_freqs)  # (F,)
    zm1 = tf.expand_dims(tf.exp(-1j * tf.cast(w, "complex64")), 1)  # (F, 1)
    exponents = tf.expand_dims(tf.range(tf.shape(filters)[1]), 0)  # (1, N)
    zm1_pow = tf.pow(zm1, tf.cast(exponents, dtype="complex64"))  # (F, N)
    f_resp = tf.tensordot(zm1_pow, tf.cast(tf.transpose(filters), dtype="complex64"), axes=1)  # (F, C)
    f_resp = tf.abs(f_resp)
    # move to log domain, not needed for center frequencies
    # f_resp = 20 * tf.math.log(f_resp) / tf.math.log(tf.constant(10.0))

    return f_resp


def sort_filters_by_center_freq(x):
    """
    This function either sorts the filters by their center frequency and returns them,
    or it returns the indices that would sort the filters.
    :param tf.Tensor x: The filter layer to sort.
    :return: Sorted filters or sorted indices.
    """
    import tensorflow as tf

    f_resp = get_frequency_response(x)
    # move to log domain, not needed for center frequencies
    # f_resp = 20 * tf.math.log(f_resp) / tf.math.log(tf.constant(10.0))

    # sorted by increasing center frequency
    center_freqs = tf.argmax(f_resp, axis=0)
    sorted_idcs = tf.argsort(center_freqs)

    return sorted_idcs


def _mask(x, batch_axis, axis, pos, max_amount, sorted_indices):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
    :param tf.Tensor|None sorted_indices: sorted indices (optional)
    """
    import tensorflow as tf

    ndim = x.get_shape().ndims
    n_batch = tf.shape(x)[batch_axis]
    dim = tf.shape(x)[axis]
    amount = tf.random.uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
    pos2 = tf.math.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.math.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
    if batch_axis > axis:
        cond = tf.transpose(cond)  # (dim,batch)
    cond = tf.reshape(cond, [tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)])
    inverse_permutation = tf.argsort(sorted_indices)
    cond = tf.gather(cond, inverse_permutation, axis=axis)
    from TFUtil import where_bc

    x = where_bc(cond, 0.0, x)
    return x


def random_mask(x, batch_axis, axis, min_num, max_num, max_dims, sorted_indices):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
    :param tf.Tensor|None sorted_indices: sorted indices (optional)
    """
    import tensorflow as tf

    n_batch = tf.shape(x)[batch_axis]
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
    else:
        num = tf.random.uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.math.log(-tf.math.log(tf.random.uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
    _, indices = tf.math.top_k(z, num if isinstance(num, int) else tf.reduce_max(num))
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
                    tf.expand_dims(tf.expand_dims(tf.less(i, num), axis=-1), axis=-1),
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


def transform(data, max_time_num, max_time, max_feature_num, max_feature, network):
    # halved before this step
    conservative_step = 2000

    x = data.placeholder
    import tensorflow as tf

    step = network.global_train_step
    increase_flag = tf.where(tf.greater_equal(step, conservative_step), 0, 1)

    filter_layer = network.layers["features"].subnetwork.layers["conv_h_filter"].output.placeholder
    sorted_filter_indices = sort_filters_by_center_freq(filter_layer)
    num_filters = network.layers["features"].subnetwork.layers["conv_l"].output.shape[-1]
    sorted_indices = tf.stack([sorted_filter_indices * num_filters + filter_idx for filter_idx in range(num_filters)])
    sorted_indices = tf.reshape(tf.transpose(sorted_indices), (-1,))

    def get_masked():
        x_masked = x
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=0,
            max_num=tf.maximum(
                tf.shape(x)[data.time_dim_axis] // int(1.0 / 0.7 * max_time),
                max_time_num,
            )
            // (1 + increase_flag),
            max_dims=max_time,
            sorted_indices=tf.range(tf.shape(x)[data.time_dim_axis]),
        )
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=0,
            max_num=max_feature_num // (1 + increase_flag),
            max_dims=max_feature,
            sorted_indices=sorted_indices,
        )
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


def get_specaug_funcs() -> list:
    return [sort_filters_by_center_freq, _mask, random_mask, transform]
