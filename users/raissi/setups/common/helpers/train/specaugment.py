def summary(name, x):
    """
    :param str name:
    :param tf.Tensor x: (batch,time,feature)
    """
    import tensorflow as tf

    # tf.summary.image wants [batch_size, height,  width, channels],
    # we have (batch, time, feature).
    img = tf.expand_dims(x, axis=3)  # (batch,time,feature,1)
    img = tf.transpose(img, [0, 2, 1, 3])  # (batch,feature,time,1)
    tf.summary.image(name, img, max_outputs=10)
    tf.summary.scalar("%s_max_abs" % name, tf.reduce_max(tf.abs(x)))
    mean = tf.reduce_mean(x)
    tf.summary.scalar("%s_mean" % name, mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
    tf.summary.scalar("%s_stddev" % name, stddev)
    tf.summary.histogram("%s_hist" % name, tf.reduce_max(tf.abs(x), axis=2))


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
    # amount = tf.random_uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
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
    # num = tf.random_uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    num = tf.random.uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    # z = -tf.log(-tf.log(tf.random_uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
    z = -tf.math.log(-tf.math.log(tf.random.uniform((n_batch, tf.shape(x)[axis]), 0, 1)))

    ## if the time axis dim. is smaller than maxval - 1
    # num = tf.cond(tf.less(tf.shape(x)[axis], max_num), lambda: tf.random_uniform(shape=(n_batch,), minval=min_num, maxval=tf.shape(x)[axis] + 1, dtype=tf.int32), lambda: num)
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


def transform_no_mask(x, network):
    import tensorflow as tf

    # summary("features", x)
    # x = tf.clip_by_value(x, -3.0, 3.0)
    # summary("features_clip", x)

    max_reps_time = network.get_config().typed_value("max_reps_time")
    min_reps_time = network.get_config().typed_value("min_reps_time")
    max_reps_feature = network.get_config().typed_value("max_reps_feature")
    min_reps_feature = network.get_config().typed_value("min_reps_feature")

    max_len_time = network.get_config().typed_value("max_len_time")
    max_len_feature = network.get_config().typed_value("max_len_feature")

    # number of repetitions for time masking
    if max_reps_time is None:
        max_reps_time = tf.maximum(tf.shape(x)[1] // (max_len_time or 20), 1)  # // 100, 1)
    if min_reps_time is None:
        min_reps_time = 1

    # number of repetitions for feature masking
    if max_reps_feature is None:
        max_reps_feature = 2
    if min_reps_feature is None:
        min_reps_feature = 1

    # length of time masking
    if max_len_time is None:
        max_len_time = 20
    # length of feature masking
    if max_len_feature is None:
        max_len_feature = tf.shape(x)[-1] // 5

    def get_masked():
        x_masked = x
        x_masked = random_mask(
            x_masked,
            axis=1,
            min_num=min_reps_time,
            max_num=max_reps_time,
            max_dims=max_len_time,
        )
        x_masked = random_mask(
            x_masked,
            axis=2,
            min_num=min_reps_feature,
            max_num=max_reps_feature,
            max_dims=max_len_feature,
        )
        # summary("features_mask", x_masked)
        return x_masked

    # x = network.cond_on_train(x, lambda: x)

    return x


def transform(x, network):
    import tensorflow as tf

    # summary("features", x)
    # x = tf.clip_by_value(x, -3.0, 3.0)
    # summary("features_clip", x)

    max_reps_time = network.get_config().typed_value("max_reps_time")
    min_reps_time = network.get_config().typed_value("min_reps_time")
    max_reps_feature = network.get_config().typed_value("max_reps_feature")
    min_reps_feature = network.get_config().typed_value("min_reps_feature")

    max_len_time = network.get_config().typed_value("max_len_time")
    max_len_feature = network.get_config().typed_value("max_len_feature")

    # number of repetitions for time masking
    if max_reps_time is None:
        max_reps_time = tf.maximum(tf.shape(x)[1] // (max_len_time or 20), 1)  # // 100, 1)
    if min_reps_time is None:
        min_reps_time = 1

    # number of repetitions for feature masking
    if max_reps_feature is None:
        max_reps_feature = 2
    if min_reps_feature is None:
        min_reps_feature = 1

    # length of time masking
    if max_len_time is None:
        max_len_time = 20
    # length of feature masking
    if max_len_feature is None:
        max_len_feature = tf.shape(x)[-1] // 5

    def get_masked():
        x_masked = x
        x_masked = random_mask(
            x_masked,
            axis=1,
            min_num=min_reps_time,
            max_num=max_reps_time,
            max_dims=max_len_time,
        )
        x_masked = random_mask(
            x_masked,
            axis=2,
            min_num=min_reps_feature,
            max_num=max_reps_feature,
            max_dims=max_len_feature,
        )
        # summary("features_mask", x_masked)
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)

    return x


# ---------------------------- these are from Jingjing legacy hybrid -------------------------------------------------


def _mask(x, batch_axis, axis, pos, max_amount):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
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
    from TFUtil import where_bc

    x = where_bc(cond, 0.0, x)
    return x


def _random_mask(x, batch_axis, axis, min_num, max_num, max_dims):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
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
            x = _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims)
    else:
        _, x = tf.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
            body=lambda i, x: (
                i + 1,
                tf.where(
                    tf.less(i, num),
                    _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims),
                    x,
                ),
            ),
            loop_vars=(0, x),
        )
    return x


def specaugment_eval_func(data, network, time_factor=1):
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
        x_masked = _random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=step1 + step2,
            max_num=2 + step1 + step2 * 2,
            max_dims=data.dim // 5,
        )
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


def specaug_layer_jingjing(in_layer):
    """
    specaug layer with default hybrid settings

    :param in_layer:
    """
    return {
        "class": "eval",
        "from": in_layer,
        "eval": "self.network.get_config().typed_value('specaugment_eval_func')("
        "source(0, as_data=True), "
        "network=self.network)",
    }


def get_funcs_jingjing():
    funcs = []
    for k, v in list(globals().items()):
        if k in ["_mask", "_random_mask", "specaugment_eval_func"]:
            funcs.append(v)
    return funcs


# ---------------------------- tina Legacy BLSTM -------------------------------------------------
def get_legacy_specaugment_epilog_blstm(t_num=3, t=10, f_num=5, f=4):
    code = f"""
def _mask(x, batch_axis, axis, pos, max_amount):
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
  from TFUtil import where_bc
  x = where_bc(cond, 0.0, x)
  return x


def random_mask(x, batch_axis, axis, min_num, max_num, max_dims):
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
      x = _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims)
  else:
    _, x = tf.while_loop( cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
                          body=lambda i, x: ( i + 1, tf.where( tf.less(i, num),
                          _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims), x )),
                          loop_vars=(0, x)
                        )
  return x

def transform(data, network):
  # to be adjusted (20-50%)
  max_time_num = {t_num}
  max_time = {t}

  max_feature_num = {f_num}
  max_feature = {f}

  # halved before this step
  conservatvie_step = 2000

  x = data.placeholder
  from returnn.tf.compat import v1 as tf
  # summary("features", x)
  step = network.global_train_step
  increase_flag = tf.where(tf.greater_equal(step, conservatvie_step), 0, 1)

  def get_masked():
    x_masked = x
    x_masked = random_mask( x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
                            min_num=0, max_num=tf.maximum(tf.shape(x)[data.time_dim_axis]//int(1/0.70*max_time), max_time_num) // (1+increase_flag),
                            max_dims=max_time
                          )
    x_masked = random_mask( x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
                            min_num=0, max_num=max_feature_num // (1+increase_flag),
                            max_dims=max_feature
                          )
    #summary("features_mask", x_masked)
    return x_masked
  x = network.cond_on_train(get_masked, lambda: x)
  return x """
    return code
