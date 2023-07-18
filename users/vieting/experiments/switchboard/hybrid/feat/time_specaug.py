from .specaug_jingjing import (
    _mask,
    _random_mask,
)


def specaugment_time_only_eval_func(data, network, time_factor=1):
    x = data.placeholder
    from returnn.tf.compat import v1 as tf

    # summary("features", x)
    step = network.global_train_step
    step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
    step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)

    def get_masked():
        x_masked = x
        x_masked = _random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=step1 + step2,
            max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // 100, 2) * (1 + step1 + step2 * 2),
            max_dims=20 // time_factor,
        )
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


def specaug_layer_only_time(in_layer):
    """
    specaug layer with default hybrid settings

    :param in_layer:
    """
    return {
        "class": "eval",
        "from": in_layer,
        "eval": "self.network.get_config().typed_value('specaugment_time_only_eval_func')("
        "source(0, as_data=True),"
        "network=self.network)",
    }


def get_funcs_only_time():
    funcs = []
    for k, v in list(globals().items()):
        if k in ["_mask", "_random_mask", "specaugment_time_only_eval_func"]:
            funcs.append(v)
    return funcs
