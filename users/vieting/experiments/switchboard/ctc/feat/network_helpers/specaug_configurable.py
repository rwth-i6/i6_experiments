from typing import Dict, Union, Optional, List
from .specaug_sort_layer2 import sort_filters_by_center_freq, _mask, random_mask, get_frequency_response


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
            Config Dict Parameters:
                max_time_num (int): The beginning maximum number of time masks to be applied. Default is 3.
                max_time (int): The beginning maximum size (in time frames) of each time mask. Default is 10.
                max_feature_num (int): The beginning maximum number of frequency masks to be applied. Default is 5.
                max_feature (int): The beginning maximum size (in frequency bins) of each frequency mask. Default is 4.
                enable_sorting (bool): Whether to sort filters by their center frequency before applying masks.
                    Default is False.
                steps_per_epoch (int): The number of steps per epoch.
                    Make sure this parameter is accurate since the all the scheduling depends on it.
                sorting_start_epoch (int): The subepoch number to start sorting filters. Default is 1.
                mask_growth_strategy (str):
                    The strategy for increasing the mask sizes over epochs (linear or step). Default is "linear".
                time_mask_num_schedule (Dict[int, float]):
                    A dictionary mapping subepoch numbers to the multiplicator for the time mask number.
                    Default is None (unchanged over the entire training).
                time_mask_size_schedule (Dict[int, float]):
                    A dictionary mapping subepoch numbers to the multiplicator for the time mask size.
                    Default is None (unchanged over the entire training).
                freq_mask_num_schedule (Dict[int, float]):
                    A dictionary mapping subepoch numbers to the multiplicator for the frequency mask number.
                    Default is None (unchanged over the entire training).
                freq_mask_size_schedule (Dict[int, float]):
                    A dictionary mapping subepoch numbers to the multiplicator for the frequency mask size.
                    Default is None (unchanged over the entire training).
                time_mask_max_proportion (float): The maximum proportion of the time axis that can be masked. Default is 1.
                freq_mask_max_proportion (float): The maximum proportion of the frequency axis that can be masked. Default is 1.
                max_time_num_seq_len_divisor (int):
                    The divisor for the sequence length to determine the maximum number of time masks. Default is 0.7.
                filter_based_masking_strategy (string): Which filter strategy to use for masking. Default is None.
                filter_factor (float): The factor that determines the probability of masking a feature based on a filter. Default is 0.5.
                max_number_masks_for_filter_based_specaug (int): The total maximum number of masks to be applied if any filter-based masking is used. Default is 50.
                filter_mask_schedule (Dict[int, float]):
                    A dictionary mapping subepoch numbers to the multiplicator for the max_number_masks_for_filter_based_specaug.
                    Default is None (unchanged over the entire training).
                enable_logging (bool): Whether to enable logging for debugging purposes. Default is False.
        num_epochs (int, optional): The total number of epochs for which the training will run. default 450

    Returns:
        tuple: A tuple containing the name of the SpecAugment layer and the functions required for it.
    """
    from .specaug_param_helper import generate_specaug_params

    if from_list is None:
        from_list = ["data"]

    default_config = {
        "max_time_num": 1,
        "max_time": 15,
        "max_feature_num": 5,
        "max_feature": 4,
        "enable_sorting": False,
        "enable_logging": False,
        "sorting_start_epoch": 1,
        "steps_per_epoch": None,
        "mask_growth_strategy": "linear",
        "time_mask_max_proportion": 1.0,
        "freq_mask_max_proportion": 1.0,
        "max_time_num_seq_len_divisor": 0.7,
        "filter_based_masking_strategy": None,
        "filter_factor": 0.5,
        "max_number_masks_for_filter_based_specaug": 50,
    }

    if config is None:
        config = {}

    # Merge provided config with defaults
    full_config = {**default_config, **config}

    assert full_config["steps_per_epoch"] is not None, "steps_per_epoch must be provided in the config"

    # Generate the SpecAugment parameters in advance
    base_values = {
        "time_mask_max_num": full_config["max_time_num"],
        "time_mask_max_size": full_config["max_time"],
        "freq_mask_max_num": full_config["max_feature_num"],
        "freq_mask_max_size": full_config["max_feature"],
        "max_number_masks_for_filter_based_specaug": full_config["max_number_masks_for_filter_based_specaug"],
    }

    schedules = {
        "time_mask_max_num": full_config.get("time_mask_num_schedule", {}),
        "time_mask_max_size": full_config.get("time_mask_size_schedule", {}),
        "freq_mask_max_num": full_config.get("freq_mask_num_schedule", {}),
        "freq_mask_max_size": full_config.get("freq_mask_size_schedule", {}),
        "max_number_masks_for_filter_based_specaug": full_config.get("filter_mask_schedule", {}),
    }

    specaug_params = generate_specaug_params(
        num_epochs=num_epochs,
        base_values=base_values,
        schedules=schedules,
        growth_strategy=full_config["mask_growth_strategy"],
    )

    # Update config with pre-generated parameters
    full_config["specaug_params"] = specaug_params

    if full_config["filter_based_masking_strategy"] is not None:

        network[name] = {
            "class": "eval",
            "from": from_list,
            "eval": f"self.network.get_config().typed_value('transform_with_filter_masking')(source(0, as_data=True), network=self.network, **{full_config})",
        }
        return [name], [
            sort_filters_by_center_freq,
            get_frequency_response,
            transform_with_filter_masking,
            filter_based_masking,
            _mask,
            random_mask,
        ]
    elif full_config["filter_based_masking_strategy"] == "variance":
        network[name] = {
            "class": "eval",
            "from": from_list,
            "eval": f"self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network, **{full_config})",
        }
        return [name], [
            sort_filters_by_center_freq,
            _mask,
            random_mask,
            transform,
        ]


def filter_based_masking(x, batch_axis, axis, probability_distribution, max_number_masks_for_probability_specaug):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor probability_distribution:
    :param int max_number_masks_for_probability_specaug:
    """
    import tensorflow as tf

    n_batch = tf.shape(x)[batch_axis]  # Batch size
    ndim = x.get_shape().ndims  # Number of dimensions in x
    n_features = tf.shape(x)[axis]  # Feature dimension size

    # Random number of masks for each batch  shape: (n_batch,)
    num_masks = tf.random.uniform(
        shape=[n_batch], minval=0, maxval=max_number_masks_for_probability_specaug + 1, dtype=tf.int32
    )
    # Total number of masks to generate across all batches
    total_masks = tf.reduce_sum(num_masks)

    # Sample feature indices to be masked based on the probability distribution (categorical needs 2d) shape: (1, total_masks)
    logits = tf.math.log(probability_distribution)
    masked_features = tf.random.categorical(tf.expand_dims(logits, 0), total_masks)
    # Generate batch offsets to map the masks to the corresponding batches. example: [0,0,1,2,2,2]
    batch_indices = tf.repeat(tf.range(n_batch), num_masks)

    # Gather the selected feature indices (have to match the shape)
    feature_indices = tf.gather(tf.reshape(masked_features, [-1]), tf.range(total_masks))  # shape: (total_masks,)
    feature_indices = tf.reshape(tf.cast(feature_indices, tf.int32), [-1])  # shape: (total_masks,)

    # Combine batch offsets and feature indices
    indices = tf.stack([batch_indices, feature_indices], axis=1)  # shape: (total_masks, 2)

    # Create a boolean mask initialized to False. shape: (n_batch, n_features)
    mask = tf.tensor_scatter_nd_update(
        tf.zeros((n_batch, n_features), dtype=tf.bool), indices, tf.ones_like(indices[:, 0], dtype=tf.bool)
    )
    # Reshape the mask to match the input tensor shape
    mask_shape = [
        tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)
    ]  # shape: (batch, time, feature)
    mask = tf.reshape(mask, mask_shape)  # shape: (batch, time, feature)

    x = tf.where(mask, tf.zeros_like(x), x)

    return x


def transform(data, network, **config):
    import tensorflow as tf

    x = data.placeholder
    step = network.global_train_step
    current_epoch = tf.cast(step / config["steps_per_epoch"], tf.int32)
    max_time_num_seq_len_divisor = tf.constant(config["max_time_num_seq_len_divisor"], dtype=tf.float32)
    variance_factor = tf.cast(config["variance_factor"], tf.float32)
    specaug_params = config["specaug_params"]

    # Determine if we should use sorting
    use_sorting = tf.logical_and(
        config["enable_sorting"], tf.greater_equal(current_epoch, config["sorting_start_epoch"])
    )

    # Get filter indices (sorted or unsorted)
    filter_layer = network.layers["features"].subnetwork.layers["conv_h_filter"].output.placeholder
    num_filters = network.layers["features"].subnetwork.layers["conv_l"].output.shape[-1]

    def get_sorted_indices():
        sorted_filter_indices = sort_filters_by_center_freq(filter_layer)
        sorted_indices = tf.stack(
            [sorted_filter_indices * num_filters + filter_idx for filter_idx in range(num_filters)]
        )
        return tf.reshape(tf.transpose(sorted_indices), (-1,))

    def get_unsorted_indices():
        return tf.range(tf.shape(x)[data.feature_dim_axis])

    sorted_indices = tf.cond(use_sorting, get_sorted_indices, get_unsorted_indices)

    def get_masked():
        x_masked = x
        time_mask_max_size = tf.gather(specaug_params["time_mask_max_size"], current_epoch)
        time_mask_max_num = tf.gather(specaug_params["time_mask_max_num"], current_epoch)
        freq_mask_max_num = tf.gather(specaug_params["freq_mask_max_num"], current_epoch)
        freq_mask_max_size = tf.gather(specaug_params["freq_mask_max_size"], current_epoch)
        total_time_masks_max_frames = tf.cast(
            tf.math.floor(config["time_mask_max_proportion"] * tf.cast(tf.shape(x)[data.time_dim_axis], tf.float32)),
            tf.int32,
        )
        total_freq_masks_max_size = tf.cast(
            tf.math.floor(config["freq_mask_max_proportion"] * tf.cast(tf.shape(x)[data.feature_dim_axis], tf.float32)),
            tf.int32,
        )
        max_time_num_seq_len = tf.cast(
            tf.math.floordiv(
                tf.cast(tf.shape(x)[data.time_dim_axis], tf.float32),
                tf.cast(1.0, tf.float32) / max_time_num_seq_len_divisor * tf.cast(time_mask_max_size, tf.float32),
            ),
            tf.int32,
        )
        # check for the limits
        actual_time_mask_max_num = tf.minimum(
            tf.maximum(
                time_mask_max_num,
                max_time_num_seq_len,
            ),
            total_time_masks_max_frames // time_mask_max_size,
        )
        actual_freq_mask_max_num = tf.minimum(freq_mask_max_num, total_freq_masks_max_size // freq_mask_max_size)

        enable_logging = tf.convert_to_tensor(config["enable_logging"], dtype=tf.bool)

        def logging_ops():
            with tf.control_dependencies(
                [
                    tf.print(
                        "Specaug Log: ",
                        current_epoch,
                        actual_time_mask_max_num,
                        actual_freq_mask_max_num,
                        max_time_num_seq_len,
                        tf.shape(x)[data.time_dim_axis],
                        sep=", ",
                    )
                ]
            ):
                return tf.identity(x_masked)

        x_masked = tf.cond(enable_logging, logging_ops, lambda: tf.identity(x_masked))

        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=0,
            max_num=actual_time_mask_max_num,
            max_dims=time_mask_max_size,
            sorted_indices=tf.range(tf.shape(x)[data.time_dim_axis]),
        )
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=0,
            max_num=actual_freq_mask_max_num,
            max_dims=freq_mask_max_size,
            sorted_indices=sorted_indices,
        )
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


def transform_with_filter_masking(data, network, **config):
    import tensorflow as tf

    x = data.placeholder
    step = network.global_train_step
    current_epoch = tf.cast(step / config["steps_per_epoch"], tf.int32)
    max_time_num_seq_len_divisor = tf.constant(config["max_time_num_seq_len_divisor"], dtype=tf.float32)
    filter_factor = tf.cast(config["filter_factor"], tf.float32)
    specaug_params = config["specaug_params"]

    # Determine if we should use sorting
    use_sorting = tf.logical_and(
        config["enable_sorting"], tf.greater_equal(current_epoch, config["sorting_start_epoch"])
    )

    # Get filter indices (sorted or unsorted)
    filter_layer = network.layers["features"].subnetwork.layers["conv_h_filter"].output.placeholder
    num_filters = network.layers["features"].subnetwork.layers["conv_l"].output.shape[-1]

    def get_sorted_indices():
        sorted_filter_indices = sort_filters_by_center_freq(filter_layer)
        sorted_indices = tf.stack(
            [sorted_filter_indices * num_filters + filter_idx for filter_idx in range(num_filters)]
        )
        return tf.reshape(tf.transpose(sorted_indices), (-1,))

    def get_unsorted_indices():
        return tf.range(tf.shape(x)[data.feature_dim_axis])

    sorted_indices = tf.cond(use_sorting, get_sorted_indices, get_unsorted_indices)

    def get_masked():
        x_masked = x
        time_mask_max_size = tf.gather(specaug_params["time_mask_max_size"], current_epoch)
        time_mask_max_num = tf.gather(specaug_params["time_mask_max_num"], current_epoch)
        freq_mask_max_num = tf.gather(specaug_params["freq_mask_max_num"], current_epoch)
        freq_mask_max_size = tf.gather(specaug_params["freq_mask_max_size"], current_epoch)
        total_time_masks_max_frames = tf.cast(
            tf.math.floor(config["time_mask_max_proportion"] * tf.cast(tf.shape(x)[data.time_dim_axis], tf.float32)),
            tf.int32,
        )
        total_freq_masks_max_size = tf.cast(
            tf.math.floor(config["freq_mask_max_proportion"] * tf.cast(tf.shape(x)[data.feature_dim_axis], tf.float32)),
            tf.int32,
        )
        max_time_num_seq_len = tf.cast(
            tf.math.floordiv(
                tf.cast(tf.shape(x)[data.time_dim_axis], tf.float32),
                tf.cast(1.0, tf.float32) / max_time_num_seq_len_divisor * tf.cast(time_mask_max_size, tf.float32),
            ),
            tf.int32,
        )
        # check for the limits
        actual_time_mask_max_num = tf.minimum(
            tf.maximum(
                time_mask_max_num,
                max_time_num_seq_len,
            ),
            total_time_masks_max_frames // time_mask_max_size,
        )
        actual_freq_mask_max_num = tf.minimum(freq_mask_max_num, total_freq_masks_max_size // freq_mask_max_size)

        if config["filter_based_masking_strategy"] == "variance":
            f_resp = get_frequency_response(filter_layer)
            variance = tf.math.reduce_variance(f_resp, axis=0)
            n_features = tf.shape(x)[data.feature_dim_axis]
            probs = variance / tf.reduce_sum(variance)
            uniform_probs = tf.ones_like(probs) / tf.cast(n_features, tf.float32)
            final_probs = filter_factor * probs + (1 - filter_factor) * uniform_probs
        elif config["filter_based_masking_strategy"] == "peakToAverage":
            # Get peak to average ratio for each filter
            f_resp = get_frequency_response(filter_layer)
            peak = tf.reduce_max(f_resp, axis=0)
            average = tf.reduce_mean(f_resp, axis=0)
            ratio = peak / average
            n_features = tf.shape(x)[data.feature_dim_axis]
            # Normalize the ratio to get probabilities
            probs = ratio / tf.reduce_sum(ratio)
            uniform_probs = tf.ones_like(probs) / tf.cast(n_features, tf.float32)
            final_probs = filter_factor * probs + (1 - filter_factor) * uniform_probs

        enable_logging = tf.convert_to_tensor(config["enable_logging"], dtype=tf.bool)

        def logging_ops():
            with tf.control_dependencies(
                [
                    tf.print(
                        "Specaug Log: ",
                        current_epoch,
                        tf.shape(x)[data.time_dim_axis],
                        sep=", ",
                    )
                ]
            ):
                return tf.identity(x_masked)

        if config["filter_based_masking_strategy"] is not None:
            x_masked = tf.cond(enable_logging, logging_ops, lambda: tf.identity(x_masked))

            x_masked = filter_based_masking(
                x_masked,
                batch_axis=data.batch_dim_axis,
                axis=data.feature_dim_axis,
                probability_distribution=final_probs,
                max_number_masks_for_probability_specaug=tf.gather(
                    config["specaug_params"]["max_number_masks_for_filter_based_specaug"], current_epoch
                ),
            )


        enable_logging = tf.convert_to_tensor(config["enable_logging"], dtype=tf.bool)

        def logging_ops():
            with tf.control_dependencies(
                [
                    tf.print(
                        "Specaug Log: ",
                        current_epoch,
                        tf.shape(x)[data.time_dim_axis],
                        sep=", ",
                    )
                ]
            ):
                return tf.identity(x_masked)
        if config["filter_based_masking_strategy"] is not None:
            x_masked = tf.cond(enable_logging, logging_ops, lambda: tf.identity(x_masked))

        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=0,
            max_num=actual_time_mask_max_num,
            max_dims=time_mask_max_size,
            sorted_indices=tf.range(tf.shape(x)[data.time_dim_axis]),
        )
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=0,
            max_num=actual_freq_mask_max_num,
            max_dims=freq_mask_max_size,
            sorted_indices=sorted_indices,
        )
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x