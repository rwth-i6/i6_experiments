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


def random_mask(x, batch_axis, axis, min_num, max_num, max_dims):
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
                  x)),
          loop_vars=(0, x))
  return x


def transform(data, network, time_factor=1):
  x = data.placeholder
  from returnn.tf.compat import v1 as tf
  # summary("features", x)
  step = network.global_train_step
  step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
  step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)
  def get_masked():
      x_masked = x
      x_masked = random_mask(
        x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
        min_num=step1 + step2, max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // 100, 2) * (1 + step1 + step2 * 2),
        max_dims=20 // time_factor)
      x_masked = random_mask(
        x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
        min_num=step1 + step2, max_num=2 + step1 + step2 * 2,
        max_dims=data.dim // 5)
      #summary("features_mask", x_masked)
      return x_masked
  cond1 = network.train_flag
  cond2 = tf.greater_equal(tf.shape(x)[data.time_dim_axis], 20)  # ignore specaug for utterances less than 20 frames
  x = tf.cond(tf.logical_and(cond1, cond2), get_masked, lambda: x)
  return x


def get_funcs():
  funcs = []
  for k, v in list(globals().items()):
    if callable(v):
      if k == 'get_funcs':
        continue
      funcs.append(v)
  return funcs
