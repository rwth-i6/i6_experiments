from typing import Dict, List, Optional, Union


def add_specaug_layer_v2(
    network: Dict,
    name: str = "specaug",
    from_list: Optional[Union[str, List[str]]] = None,
    min_reps_time: int = 0,
    max_reps_time: Optional[int] = None,
    max_len_time: int = 20,
    min_reps_feature: int = 0,
    max_reps_feature: int = 1,
    max_len_feature: Optional[int] = None,
) -> List[str]:
    if from_list is None:
        from_list = ["data"]
    network[name] = {
        "class": "eval",
        "from": from_list,
        "eval": f'self.network.get_config().typed_value("transform")(source(0, as_data=True), network=self.network, min_reps_time={min_reps_time}, max_reps_time={max_reps_time}, max_len_time={max_len_time}, min_reps_feature={min_reps_feature}, max_reps_feature={max_reps_feature}, max_len_feature={max_len_feature})',
    }

    return [name]


def mask(x, axis, pos, max_amount):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int max_amount: inclusive
    """
    import tensorflow as tf

    ndim = x.get_shape().ndims
    n_batch = tf.shape(x)[0]
    dim = tf.shape(x)[axis]
    amount = tf.random.uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
    pos2 = tf.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
    cond = tf.reshape(cond, [tf.shape(x)[i] if i in (0, axis) else 1 for i in range(ndim)])
    from TFUtil import where_bc

    x = where_bc(cond, 0.0, x)
    return x


def random_mask(x, axis, min_num, max_num, max_dims):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int max_dims: inclusive
    """
    import tensorflow as tf

    n_batch = tf.shape(x)[0]
    num = tf.random.uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.math.log(-tf.math.log(tf.random.uniform((n_batch, tf.shape(x)[axis]), 0, 1)))

    ## if the time axis dim. is smaller than maxval - 1
    num = tf.cond(
        tf.less(tf.shape(x)[axis], max_num),
        lambda: tf.random.uniform(
            shape=(n_batch,),
            minval=min_num,
            maxval=tf.shape(x)[axis] + 1,
            dtype=tf.int32,
        ),
        lambda: num,
    )

    _, indices = tf.nn.top_k(z, tf.reduce_max(num))
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
    _, x = tf.while_loop(
        cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
        body=lambda i, x: (
            i + 1,
            tf.compat.v1.where(
                tf.less(i, num),
                mask(x, axis=axis, pos=indices[:, i], max_amount=max_dims),
                x,
            ),
        ),
        loop_vars=(0, x),
    )
    return x


def transform(
    data,
    network,
    min_reps_time=0,
    max_reps_time=None,
    max_len_time=20,
    min_reps_feature=0,
    max_reps_feature=1,
    max_len_feature=None,
):
    x = data.placeholder

    import tensorflow as tf

    # number of repetitions for time masking
    if max_reps_time is None:
        max_reps_time = tf.maximum(tf.shape(x)[data.time_dim_axis] // max_len_time, 1)

    # length of feature masking
    if max_len_feature is None:
        max_len_feature = tf.shape(x)[data.feature_dim_axis] // 5

    def get_masked():
        x_masked = x
        x_masked = random_mask(
            x_masked,
            axis=data.time_dim_axis,
            min_num=min_reps_time,
            max_num=max_reps_time,
            max_dims=max_len_time,
        )
        x_masked = random_mask(
            x_masked,
            axis=data.feature_dim_axis,
            min_num=min_reps_feature,
            max_num=max_reps_feature,
            max_dims=max_len_feature,
        )
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)

    return x


def get_specaug_funcs_v2() -> list:
    return [mask, random_mask, transform]
