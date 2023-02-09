from typing import Dict, Union, Optional, List
from i6_core.returnn.config import CodeWrapper
import returnn_common.asr.specaugment as rc_specaug


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

    return [name]


def add_specaug_layer_v2(
    network: Dict,
    name: str = "specaug",
    from_list: Optional[Union[str, List[str]]] = "data",
) -> List[str]:
    network[name] = {
        "class": "eval",
        "from": from_list,
        "eval": CodeWrapper(rc_specaug.specaugment_v1_eval_func.__name__),
    }

    return [name], get_specaug_func_v2()


def _mask(x, batch_axis, axis, pos, max_amount):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
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
    from TFUtil import where_bc

    x = where_bc(cond, 0.0, x)
    return x


def random_mask(x, batch_axis, axis, min_num, max_num, max_dims):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
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
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
    if isinstance(num, int):
        for i in range(num):
            x = _mask(
                x,
                batch_axis=batch_axis,
                axis=axis,
                pos=indices[:, i],
                max_amount=max_dims,
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
        )
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=0,
            max_num=max_feature_num // (1 + increase_flag),
            max_dims=max_feature,
        )
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


def get_specaug_funcs() -> list:
    return [_mask, random_mask, transform]


def get_specaug_func_v2() -> list:
    return [
        rc_specaug._mask_v1,
        rc_specaug.random_mask_v1,
        rc_specaug.specaugment_v1_eval_func,
    ]
