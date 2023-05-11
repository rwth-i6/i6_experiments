"""
SpecAugment helpers from returnn_common
"""


# Use this for an EvalLayer
def specaugment_v1_eval_func(*, source, global_train_step_dependent: bool = True, only_on_train: bool = True, **kwargs):
    """
    :rtype: tf.Tensor
    """
    from returnn.tf.compat import v1 as tf

    data = source(0, as_data=True)
    time_factor = 1  # for switchout == 6
    x = data.placeholder
    network = kwargs["self"].network
    if global_train_step_dependent:
        step = network.global_train_step
        step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
        step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)
    else:
        step1 = step2 = 1

    def get_masked():
        """
        :return: masked tensor
        """
        time_len = tf.shape(x)[data.time_dim_axis]
        x_masked = x
        x_masked = random_mask_v1(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=tf.minimum(step1 + step2, time_len),
            max_num=tf.minimum(tf.maximum(time_len // 100, 2) * (1 + step1 + step2 * 2), time_len),
            max_dims=20 // time_factor,
        )
        x_masked = random_mask_v1(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=step1 + step2,
            max_num=2 + step1 + step2 * 2,
            max_dims=data.dim // 5,
        )
        return x_masked

    if only_on_train:
        x = network.cond_on_train(get_masked, lambda: x)
    else:
        x = get_masked()
    return x


def random_mask_v1(x, *, batch_axis, axis, min_num, max_num, max_dims, mask_value=0.0):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
    :param float|int mask_value:
    :rtype: tf.Tensor
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
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
    if isinstance(num, int):
        for i in range(num):
            x = _mask_v1(
                x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims, mask_value=mask_value
            )
    else:
        _, x = tf.while_loop(
            cond=lambda i_, _: tf.less(i_, tf.reduce_max(num)),
            body=lambda i_, x_: (
                i_ + 1,
                tf.where(
                    tf.less(i_, num),
                    _mask_v1(
                        x_,
                        batch_axis=batch_axis,
                        axis=axis,
                        pos=indices[:, i_],
                        max_amount=max_dims,
                        mask_value=mask_value,
                    ),
                    x_,
                ),
            ),
            loop_vars=(0, x),
        )
    return x


def _mask_v1(x, *, batch_axis, axis, pos, max_amount, mask_value=0.0):
    """
    :param tf.Tensor x: (batch,time,[feature])
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
    :param float|int mask_value:
    """
    from returnn.tf.compat import v1 as tf
    from returnn.tf.util.basic import where_bc

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
    x = where_bc(cond, mask_value, x)
    return x