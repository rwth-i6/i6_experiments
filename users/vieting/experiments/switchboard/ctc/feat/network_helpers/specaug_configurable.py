from typing import Dict, Union, Optional, List

def add_specaug_layer(
    network: Dict,
    name: str = "specaug",
    from_list: Optional[Union[str, List[str]]] = None,
    num_epochs: int = 450,
    config: Dict = None,
) -> tuple:
    """
    Add a customizable SpecAugment layer to the network.

    This function adds a highly configurable SpecAugment layer to the given network,
    allowing for fine-grained control over various aspects of the augmentation process.

    Args:
        network (Dict): The network to which the SpecAugment layer will be added.
        name (str, optional): The name of the SpecAugment layer. Defaults to "specaug".
        from_list (Optional[Union[str, List[str]]], optional): The input layer(s) for the SpecAugment layer. 
            Default is ["data"].
        config (Dict, optional): A dictionary containing configuration options for the SpecAugment layer.
        num_epochs (int, optional): The total number of epochs for which the training will run. default 450

    Returns:
        tuple: A tuple containing the name of the SpecAugment layer and the functions required for it.
    """
    from .specaug_param_helper import generate_specaug_params

    if from_list is None:
        from_list = ["data"]
    
    default_config = {
        "max_time_num": 3,
        "max_time": 10,
        "max_feature_num": 4,
        "max_feature": 5,
        "enable_sorting": True,
        "sorting_start_epoch": 0,
        "steps_per_epoch": 2080,
        "mask_growth_strategy": "linear",
        "time_mask_num_schedule": {0: 1, 10: 2, 20: 3},
        "time_mask_size_schedule": {0: 1, 15: 1.5, 25: 2},
        "freq_mask_num_schedule": {0: 1, 5: 1.5, 15: 2},
        "freq_mask_size_schedule": {0: 1, 10: 1.2, 20: 1.5},
        "time_mask_max_proportion": 0.7,
    }
    
    if config is None:
        config = {}
    
    # Merge provided config with defaults
    full_config = {**default_config, **config}

    # Generate the SpecAugment parameters in advance
    base_values = {
        "time_mask_max_num": full_config["max_time_num"],
        "time_mask_max_size": full_config["max_time"],
        "freq_mask_max_num": full_config["max_feature_num"],
        "freq_mask_max_size": full_config["max_feature"],
    }
    schedules = {
        "time_mask_max_num": full_config["time_mask_num_schedule"],
        "time_mask_max_size": full_config["time_mask_size_schedule"],
        "freq_mask_max_num": full_config["freq_mask_num_schedule"],
        "freq_mask_max_size": full_config["freq_mask_size_schedule"],
    }

    specaug_params = generate_specaug_params(
        num_epochs=num_epochs,
        base_values=base_values,
        schedules=schedules,
        growth_strategy=full_config["mask_growth_strategy"]
    )

    # Update config with pre-generated parameters
    full_config["specaug_params"] = specaug_params
    
    network[name] = {
        "class": "eval",
        "from": from_list,
        "eval": f"self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network, **{full_config})",
    }

    return [name], get_specaug_funcs()


def sort_filters_by_center_freq(x):
    """
    This function either sorts the filters by their center frequency and returns them,
    or it returns the indices that would sort the filters.
    :param tf.Tensor x: The filter layer to sort.
    :param bool return_sorted_filters: Whether to return the sorted filters or the sorted indices.
    :return: Sorted filters or sorted indices.
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
    amount = tf.random.uniform(
        shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32
    )
    pos2 = tf.math.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.math.logical_and(
        tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc)
    )  # (batch,dim)
    if batch_axis > axis:
        cond = tf.transpose(cond)  # (dim,batch)
    cond = tf.reshape(
        cond, [tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)]
    )
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
        num = tf.random.uniform(
            shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32
        )
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.math.log(
        -tf.math.log(tf.random.uniform((n_batch, tf.shape(x)[axis]), 0, 1))
    )
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

def transform(data, network, **config):
    x = data.placeholder
    step = network.global_train_step
    current_epoch = tf.cast(step / config['steps_per_epoch'], tf.int32)

    specaug_params = config['specaug_params']

    # Determine if we should use sorting
    use_sorting = tf.logical_and(
        config['enable_sorting'],
        tf.greater_equal(current_epoch, config['sorting_start_epoch'])
    )
    
    # Get filter indices (sorted or unsorted)
    filter_layer = network.layers["features"].subnetwork.layers["conv_h_filter"].output.placeholder
    num_filters = network.layers["features"].subnetwork.layers["conv_l"].output.shape[-1]
    
    def get_sorted_indices():
        sorted_filter_indices = sort_filters_by_center_freq(filter_layer)
        sorted_indices = tf.stack(
            [sorted_filter_indices * num_filters + filter_idx for filter_idx in range(num_filters)])
        return tf.reshape(tf.transpose(sorted_indices), (-1,))
    
    def get_unsorted_indices():
        return tf.range(tf.shape(x)[data.feature_dim_axis])
    
    sorted_indices = tf.cond(use_sorting, get_sorted_indices, get_unsorted_indices)
    
    def get_masked():
        x_masked = x
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=0,
            max_num=tf.maximum(
                tf.shape(x)[data.time_dim_axis] // int(1.0 / config['time_mask_max_proportion'] * specaug_params['time_mask_max_size'][current_epoch]),
                specaug_params['time_mask_max_num'][current_epoch],
            ),
            max_dims=specaug_params['time_mask_max_size'][current_epoch],
            sorted_indices=tf.range(tf.shape(x)[data.time_dim_axis]),
        )
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=0,
            max_num=specaug_params['freq_mask_max_num'][current_epoch],
            max_dims=specaug_params['freq_mask_max_size'][current_epoch],
            sorted_indices=sorted_indices,
        )
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x

def get_specaug_funcs() -> list:
    return [sort_filters_by_center_freq, _mask, random_mask, transform]