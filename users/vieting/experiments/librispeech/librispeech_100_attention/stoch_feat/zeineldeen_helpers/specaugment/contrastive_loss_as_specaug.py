def summary(name, x):
  """
  :param str name:
  :param tf.Tensor x: (batch,time,feature)
  """
  from returnn.tf.compat import v1 as tf
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


def _get_mask(x, axis, pos, max_amount):
  """
  :param tf.Tensor x: (batch,time,feature)
  :param int axis:
  :param tf.Tensor pos: (batch,)
  :param int max_amount: inclusive
  """
  from returnn.tf.compat import v1 as tf
  n_batch = tf.shape(x)[0]
  dim = tf.shape(x)[axis]
  amount = tf.random_uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
  pos2 = tf.minimum(pos + amount, dim)
  idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
  pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
  pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
  cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
  return cond


def get_contrastive_loss_mask(source, **_kwargs):
  def _random_mask(x, axis, min_num, max_num, max_dims):
    from returnn.tf.compat import v1 as tf
    n_batch = tf.shape(x)[0]
    num = tf.random_uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.log(-tf.log(tf.random_uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
    _, indices = tf.nn.top_k(z, tf.reduce_max(num))
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])

    res_mask = tf.zeros(shape=[n_batch, tf.shape(x)[axis]], dtype=tf.bool)  # all False
    _, res_mask = tf.while_loop(
      cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
      body=lambda i, res_mask: (
        i + 1,
        tf.where(
          tf.less(i, num),
          tf.math.logical_or(res_mask, _get_mask(x, axis=axis, pos=indices[:, i], max_amount=max_dims)),
          res_mask)),
      loop_vars=(0, res_mask))
    return res_mask  # (batch,dim)

  from returnn.tf.compat import v1 as tf
  data = source(0, as_data=True, auto_convert=False)
  assert (data.batch_dim_axis, data.time_dim_axis) == (0, 1)
  x = data.placeholder
  mask = _random_mask(x, axis=1, min_num=1, max_num=tf.maximum(tf.shape(x)[1] // 100, 1), max_dims=20)
  return mask


def _mask(x, axis, pos, max_amount):
  from returnn.tf.compat import v1 as tf
  ndim = x.get_shape().ndims
  cond = _get_mask(x, axis, pos, max_amount)
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
  from returnn.tf.compat import v1 as tf
  n_batch = tf.shape(x)[0]
  num = tf.random_uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
  # https://github.com/tensorflow/tensorflow/issues/9260
  # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
  z = -tf.log(-tf.log(tf.random_uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
  _, indices = tf.nn.top_k(z, tf.reduce_max(num))
  # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
  # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
  _, x = tf.while_loop(
    cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
    body=lambda i, x: (
      i + 1,
      tf.where(
        tf.less(i, num),
        _mask(x, axis=axis, pos=indices[:, i], max_amount=max_dims),
        x)),
    loop_vars=(0, x))
  return x


def transform(data, network):
  x = data.placeholder
  from returnn.tf.compat import v1 as tf
  x = tf.clip_by_value(x, -3.0, 3.0)

  def get_masked():
    x_masked = x
    #x_masked = random_mask(x_masked, axis=1, min_num=1, max_num=tf.maximum(tf.shape(x)[1] // 100, 1), max_dims=20)
    x_masked = random_mask(x_masked, axis=2, min_num=1, max_num=2, max_dims=40 // 5)
    return x_masked

  x = network.cond_on_train(get_masked, lambda: x)
  return x


def get_funcs():
  funcs = []
  for k, v in list(globals().items()):
    if callable(v):
      if k == 'get_funcs':
        continue
      funcs.append(v)
  return funcs
