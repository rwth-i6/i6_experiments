# --------------------- helpers ----------------------------------

## x should be the energy tensor of shape (batch, att_heads, key, query)
## y should be the attention weight tensor of shape (batch, att_heads, query, key)

def att_weight_suppression(x, y, upsilon=0.5):
  ## shape=(8, None, None), time_dim_axis=2, batch_shape_meta=[B,F|8,T|'time:var:extern_data:data','time:var:extern_data:data'])
  ## input is the energy matrix whose row indices are key positions (a column vector for each query)

  ## (batch, att_heads, key, query)

  import tensorflow as tf

  ## match the shapes of the energy and the attention weight tensors
  att_weights_reshaped = tf.transpose(y, [0, 1, 3, 2])

  mean = tf.reduce_mean(att_weights_reshaped, axis=-2, keepdims=True)
  #std = tf.math.reduce_std(att_weights_reshaped, axis=-2, keepdims=True)

  std = tf.sqrt(tf.reduce_mean(tf.square(att_weights_reshaped - mean), axis=-2, keepdims=True))
  # img_before = tf.transpose(x[:,0:1,:,:], [0, 2, 3, 1])  # (batch,feature, time,1)
  # tf.summary.image('energies_before', img_before, max_outputs=10)

  cond = tf.greater_equal(att_weights_reshaped, mean - upsilon*std)

  from TFUtil import where_bc

  x = where_bc(cond, x, float("-inf"))

  # network.get_layer(layer_name).allow_inf_in_output = True
  # img_after = tf.transpose(x[:,0:1,:,:], [0, 2, 3, 1])  # (batch,feature, time,1)
  # tf.summary.image('energies_after', img_after, max_outputs=10)

  return x

# starting from 12 layers, and grow 4 layers at each step until 24 layers
def custom_construction_algo(idx, net_dict):

  start_num_layers = 12
  growing_factor = 4

  assert growing_factor % 2 == 0

  orig_num_sa_layers = 0
  while "enc_{:03d}".format(orig_num_sa_layers+1) in net_dict:
      orig_num_sa_layers += 1
  ## at least six layers
  assert orig_num_sa_layers >= 6

  idx = max(idx - 2, 0)  # repeat for idx 0, 1, 2
  num_sa_layers = idx * growing_factor + start_num_layers  # idx starts at 0

  ## the original architecture reached
  if num_sa_layers >= orig_num_sa_layers:
    return None

  # net_dict["#config"] = {}
  # if num_sa_layers <= 16:
  #   net_dict["#config"]["batch_size"] = 8192

  # grow from the end of the encoder
  #net_dict["encoder"]["from"] = ["enc_{:03d}".format(num_sa_layers)]

  # grow frow from the middle of the encoder
  middle_pos = int(start_num_layers/2)

  # start connecting point: "enc_{:03d}".format(middle_pos)
  # end connecting point: "enc_{:03d}".format(orig_num_sa_layers - middle_pos + 1)

  # idx = 1 -> start = 8, end = 17
  conn_start = middle_pos + int(growing_factor/2) * idx
  conn_end = orig_num_sa_layers - middle_pos - int(growing_factor/2) * idx + 1

  conn_end_old = (orig_num_sa_layers - middle_pos - int(growing_factor/2) * (idx-1) + 1) if idx > 0 else (orig_num_sa_layers - middle_pos + 1)

  net_dict["enc_{:03d}_ff1_laynorm".format(conn_end_old)]["from"] = ["enc_{:03d}".format(conn_end_old-1)]
  net_dict["enc_{:03d}_ff1_out".format(conn_end_old)]["from"][0] = "enc_{:03d}".format(conn_end_old-1)

  net_dict["enc_{:03d}_ff1_laynorm".format(conn_end)]["from"] = ["enc_{:03d}".format(conn_start)]
  net_dict["enc_{:03d}_ff1_out".format(conn_end)]["from"][0] = "enc_{:03d}".format(conn_start)

  for i in range((conn_start+1), conn_end):
    name = "enc_{:03d}".format(i)
    # del net_dict["{}_self_att_laynorm".format(name)]
    # del net_dict["{}_self_att_att".format(name)]
    # del net_dict["{}_self_att_lin".format(name)]
    # del net_dict["{}_self_att_drop".format(name)]
    # del net_dict["{}_self_att_out".format(name)]
    # del net_dict["{}_ff_laynorm".format(name)]
    # del net_dict["{}_ff_conv1".format(name)]
    # del net_dict["{}_ff_conv2".format(name)]
    # del net_dict["{}_ff_drop".format(name)]
    # del net_dict["{}_ff_out".format(name)]
    del net_dict["{}_ff1_laynorm".format(name)]
    del net_dict["{}_ff1_conv1".format(name)]
    del net_dict["{}_ff1_conv2".format(name)]
    del net_dict["{}_ff1_drop".format(name)]
    del net_dict["{}_ff1_drop_half".format(name)]
    del net_dict["{}_ff1_out".format(name)]

    del net_dict["{}_self_att_laynorm".format(name)]
    del net_dict["{}_self_att_att".format(name)]
    del net_dict["{}_self_att_lin".format(name)]
    del net_dict["{}_self_att_drop".format(name)]
    del net_dict["{}_self_att_out".format(name)]

    del net_dict["{}_conv_laynorm".format(name)]
    del net_dict["{}_conv_pointwise1".format(name)]
    del net_dict["{}_conv_GLU".format(name)]
    del net_dict["{}_conv_depthwise".format(name)]
    del net_dict["{}_conv_batchnorm".format(name)]
    del net_dict["{}_conv_act".format(name)]
    del net_dict["{}_conv_pointwise2".format(name)]
    del net_dict["{}_conv_dropout".format(name)]
    del net_dict["{}_conv_output".format(name)]

    del net_dict["{}_ff2_laynorm".format(name)]
    del net_dict["{}_ff2_conv1".format(name)]
    del net_dict["{}_ff2_conv2".format(name)]
    del net_dict["{}_ff2_drop".format(name)]
    del net_dict["{}_ff2_drop_half".format(name)]
    del net_dict["{}_ff2_out".format(name)]
    del net_dict[name]

  return net_dict


#starting from 12 layers, and grow 4 layers at each step until 24 layers
def custom_construction_algo2(idx, net_dict):

  start_num_layers = 12
  growing_factor = 4

  assert growing_factor % 2 == 0

  orig_num_sa_layers = 0
  while "enc_{:03d}".format(orig_num_sa_layers+1) in net_dict:
      orig_num_sa_layers += 1
  ## at least six layers
  assert orig_num_sa_layers >= 6

  idx = max(idx - 2, 0)  # repeat for idx 0, 1, 2
  num_sa_layers = idx * growing_factor + start_num_layers  # idx starts at 0

  ## the original architecture reached
  if num_sa_layers >= orig_num_sa_layers:
    return None

  net_dict["encoder"]["from"] = ["enc_{:03d}".format(num_sa_layers)]

  # delete the trailing layers
  for i in range((num_sa_layers+1), orig_num_sa_layers + 1):
    name = "enc_{:03d}".format(i)
    # del net_dict["{}_self_att_laynorm".format(name)]
    # del net_dict["{}_self_att_att".format(name)]
    # del net_dict["{}_self_att_lin".format(name)]
    # del net_dict["{}_self_att_drop".format(name)]
    # del net_dict["{}_self_att_out".format(name)]
    # del net_dict["{}_ff_laynorm".format(name)]
    # del net_dict["{}_ff_conv1".format(name)]
    # del net_dict["{}_ff_conv2".format(name)]
    # del net_dict["{}_ff_drop".format(name)]
    # del net_dict["{}_ff_out".format(name)]
    del net_dict["{}_ff1_laynorm".format(name)]
    del net_dict["{}_ff1_conv1".format(name)]
    del net_dict["{}_ff1_conv2".format(name)]
    del net_dict["{}_ff1_drop".format(name)]
    del net_dict["{}_ff1_drop_half".format(name)]
    del net_dict["{}_ff1_out".format(name)]

    del net_dict["{}_self_att_laynorm".format(name)]
    del net_dict["{}_self_att_att".format(name)]
    del net_dict["{}_self_att_lin".format(name)]
    del net_dict["{}_self_att_drop".format(name)]
    del net_dict["{}_self_att_out".format(name)]

    del net_dict["{}_conv_laynorm".format(name)]
    del net_dict["{}_conv_pointwise1".format(name)]
    del net_dict["{}_conv_GLU".format(name)]
    del net_dict["{}_conv_depthwise".format(name)]
    del net_dict["{}_conv_batchnorm".format(name)]
    del net_dict["{}_conv_act".format(name)]
    del net_dict["{}_conv_pointwise2".format(name)]
    del net_dict["{}_conv_dropout".format(name)]
    del net_dict["{}_conv_output".format(name)]

    del net_dict["{}_ff2_laynorm".format(name)]
    del net_dict["{}_ff2_conv1".format(name)]
    del net_dict["{}_ff2_conv2".format(name)]
    del net_dict["{}_ff2_drop".format(name)]
    del net_dict["{}_ff2_drop_half".format(name)]
    del net_dict["{}_ff2_out".format(name)]
    del net_dict[name]

  return net_dict

def custom_dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
  def noam(n, warmup_n, model_d):
    """
    Noam style learning rate scheduling

    (k is identical to the global learning rate)

    :param int|float|tf.Tensor n:
    :param int|float|tf.Tensor warmup_n:
    :param int|float|tf.Tensor model_d:
    :return:
    """
    from returnn.tf.compat import v1 as tf
    model_d = tf.cast(model_d, tf.float32)
    n = tf.cast(n, tf.float32)
    warmup_n = tf.cast(warmup_n, tf.float32)
    return tf.pow(model_d, -0.5) * tf.minimum(tf.pow(n, -0.5), n * tf.pow(warmup_n, -1.5))

  """

  :param TFNetwork network:
  :param tf.Tensor global_train_step:
  :param tf.Tensor learning_rate: current global learning rate
  :param kwargs:
  :return:
  """
  WARMUP_N = 25000
  MODEL_D = 512
  return learning_rate * noam(n=global_train_step, warmup_n=WARMUP_N, model_d=MODEL_D)

def custom_dynamic_learning_rate_256(*, network, global_train_step, learning_rate, **kwargs):
  def noam(n, warmup_n, model_d):
    """
    Noam style learning rate scheduling

    (k is identical to the global learning rate)

    :param int|float|tf.Tensor n:
    :param int|float|tf.Tensor warmup_n:
    :param int|float|tf.Tensor model_d:
    :return:
    """
    from returnn.tf.compat import v1 as tf
    model_d = tf.cast(model_d, tf.float32)
    n = tf.cast(n, tf.float32)
    warmup_n = tf.cast(warmup_n, tf.float32)
    return tf.pow(model_d, -0.5) * tf.minimum(tf.pow(n, -0.5), n * tf.pow(warmup_n, -1.5))

  """

  :param TFNetwork network:
  :param tf.Tensor global_train_step:
  :param tf.Tensor learning_rate: current global learning rate
  :param kwargs:
  :return:
  """
  WARMUP_N = 25000
  MODEL_D = 256
  return learning_rate * noam(n=global_train_step, warmup_n=WARMUP_N, model_d=MODEL_D)

def dynamic_learning_rate_cyclic_with_warmup(*, network, global_train_step, learning_rate, **kwargs):
  def cyclic(n, warmup_n, decay, interval):
    """
    Noam style learning rate scheduling

    (k is identical to the global learning rate)

    :param int|float|tf.Tensor n:
    :param int|float|tf.Tensor warmup_n:
    :param int|float|tf.Tensor model_d:
    :return:
    """
    from returnn.tf.compat import v1 as tf

    n = tf.cast(n, tf.float32)
    warmup_n = tf.cast(warmup_n, tf.float32)

    if n <= warmup_n:
      return learning_rate/2 + learning_rate/2 * n/warmup_n
    else:
      return

  """

  :param TFNetwork network:
  :param tf.Tensor global_train_step:
  :param tf.Tensor learning_rate: current global learning rate
  :param kwargs:
  :return:
  """
  WARMUP_N = 18000
  decay = 0.999
  interval = 11000
  return learning_rate * \
         cyclic(n=global_train_step, warmup_n=WARMUP_N, decay=decay, interval=interval)

# (B, H, key-T, 1)
def rel_shift(r, tile=False):
  # expected (B, H, key-T, query-T) --> (query-T, key-T, B, H)
  import sys
  import tensorflow as tf

  # (?, 1, 4, ?)
  print('r shape:', r.shape, file=sys.stdout)

  if tile:
    r = tf.repeat(r, [tf.shape(r)[-1]], axis=-3)
    print('r shape after tiling:', r.shape, file=sys.stdout)
    r = tf.transpose(r, perm=[1, 3, 0, 2])
  else:
    r = tf.transpose(r, perm=[3, 2, 0, 1])

  print('r shape after transposing:', r.shape, file=sys.stdout)

  x = tf.reverse(tf.slice(r, [0, 1, 0, 0], [-1, -1, -1, -1]), [1])
  x = tf.concat([x, r], 1)

  x_size = tf.shape(x)

  x = tf.pad(x, [[0, 0], [0, 1], [0, 0], [0, 0]])

  x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])

  x = tf.slice(x, [0, 0, 0, 0], [x_size[1], -1, -1, -1])

  x = tf.reshape(x, x_size)

  result = tf.slice(x, [0, x_size[1] // 2, 0, 0], [-1] * 4)

  result = tf.transpose(result, perm=[2, 3, 1, 0])
  print('r shape final:', result.shape, file=sys.stdout)
  return result

# da: (B, T, F)
# ma: (B, T=1, H, F)
def tile_weight_matrix(ma, da, num_heads, key_per_head):
  import tensorflow as tf

  #ma_after = tf.squeeze(ma, axis = [1, 2, 3, 4]) # eliminate the influence of returnn
  #ma_after = tf.expand_dims(ma_after, 1)

  ma_after = tf.repeat(ma, [tf.shape(da)[-1]], axis = 1)

  ma_after.set_shape([None, None, num_heads, key_per_head])

  return ma_after

# def tile_weight_matrix(ma, num_heads):
#   import tensorflow as tf
#   # (B, 4, T, T')
#   # (?, 1, 4, ?)
#   ma_after = tf.transpose(ma, [0, 2, 3, 1])
#   ma_after = tf.tile(ma_after, [1, 1, 1, tf.shape(ma)[-2]])
#   ma_after.set_shape([None, num_heads, None, None])
#   return ma_after
