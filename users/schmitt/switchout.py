

def switchout_target(source, network, **kwargs):
  import tensorflow as tf
  from returnn.tf.util.basic import where_bc
  time_factor = 6
  data = source(0, as_data=True)
  assert data.is_batch_major  # just not implemented otherwise
  x = data.placeholder

  def get_switched():
    x_ = x
    shape = tf.shape(x)
    n_batch = tf.shape(x)[data.batch_dim_axis]
    n_time = tf.shape(x)[data.time_dim_axis]
    take_rnd_mask = tf.less(tf.random.uniform(shape=shape, minval=0., maxval=1.), 0.05)
    take_blank_mask = tf.less(tf.random.uniform(shape=shape, minval=0., maxval=1.), 0.5)
    rnd_label = tf.random.uniform(shape=shape, minval=0, maxval=eval("target_num_labels"), dtype=tf.int32)
    rnd_label = where_bc(take_blank_mask, eval("targetb_blank_idx"), rnd_label)
    x_ = where_bc(take_rnd_mask, rnd_label, x_)
    x_ = eval("random_mask")(x_, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis, min_num=0,
      max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // (50 // time_factor), 1), max_dims=20 // time_factor,
      mask_value=eval("targetb_blank_idx"))
    # x_ = tf.Print(x_, ["switch", x[0], "to", x_[0]], summarize=100)
    return x_

  x = network.cond_on_train(get_switched, lambda: x)
  return x