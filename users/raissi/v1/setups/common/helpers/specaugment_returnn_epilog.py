# based on https://arxiv.org/abs/2004.00960
def get_specaugment_epilog(t_num=3, t=10, f_num=5, f=4):
    code = f"""
def _mask(x, batch_axis, axis, pos, max_amount):
  import tensorflow as tf
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
  import tensorflow as tf
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


def transform(data, network):
    # to be adjusted (20-50%)
    max_time_num = 3
    max_time = 10

    max_feature_num = 4
    max_feature = 5

    # halved before this step
    conservatvie_step = 2000

    x = data.placeholder
    from returnn.tf.compat import v1 as tf

    # summary("features", x)
    step = network.global_train_step
    increase_flag = tf.where(tf.greater_equal(step, conservatvie_step), 0, 1)

    def get_masked():
        x_masked = x
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=0,
            max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // int(1 / 0.70 * max_time), max_time_num)
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
        # summary("features_mask", x_masked)
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


mask_code_10_simon = """

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
  max_time_num = 3
  max_time = 10

  max_feature_num = 5
  max_feature = 4

  # halved before this step
  conservative_step = 2000

  x = data.placeholder
  from returnn.tf.compat import v1 as tf
  # summary("features", x)
  step = network.global_train_step
  increase_flag = tf.where(tf.greater_equal(step, conservative_step), 0, 1)

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
  return x
"""
