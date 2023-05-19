from dataclasses import dataclass


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
                    x)),
            loop_vars=(0, x))
    return x


def specaugment_eval_func(data, network,
                          min_frame_masks=2, mask_each_n_frames=25, max_frames_per_mask=20,
                          min_feature_masks=2, max_feature_masks=5, max_features_per_mask=8):
    x = data.placeholder
    from returnn.tf.compat import v1 as tf
    def get_masked():
        x_masked = x
        x_masked = _random_mask(
            x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
            min_num=min_frame_masks,
            max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // mask_each_n_frames, min_frame_masks),
            max_dims=max_frames_per_mask)
        x_masked = _random_mask(
            x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
            min_num=min_feature_masks, max_num=max_feature_masks,
            max_dims=max_features_per_mask)
        return x_masked
    x = network.cond_on_train(get_masked, lambda: x)
    return x


@dataclass(eq=False, frozen=True)
class SpecAugmentSettings:
    """
    default hybrid settings from chris
    """
    min_frame_masks: int = 1
    mask_each_n_frames: int = 100
    max_frames_per_mask: int = 20
    min_feature_masks: int = 1
    max_feature_masks: int = 2
    max_features_per_mask: int = 10

    def get_options(self):
        return self.__dict__


def specaug_layer(in_layer, min_frame_masks=1, mask_each_n_frames=100, max_frames_per_mask=20,
                  min_feature_masks=1, max_feature_masks=2, max_features_per_mask=10):
    """
    specaug layer with default hybrid settings

    :param in_layer:
    :param min_frame_masks:
    :param mask_each_n_frames:
    :param max_frames_per_mask:
    :param min_feature_masks:
    :param max_feature_masks:
    :param max_features_per_mask:
    :return:
    """
    return {
        "class": "eval",
        "from": in_layer,
        "eval":"self.network.get_config().typed_value('specaugment_eval_func')("
               "source(0, as_data=True), "
               "network=self.network, "
               "min_frame_masks=%i, "
               "mask_each_n_frames=%i, "
               "max_frames_per_mask=%i, "
               "min_feature_masks=%i, "
               "max_feature_masks=%i, "
               "max_features_per_mask=%i)" % (
                  min_frame_masks, mask_each_n_frames, max_frames_per_mask,
                  min_feature_masks, max_feature_masks, max_features_per_mask
               ),
    }


def get_funcs():
    funcs = []
    for k, v in list(globals().items()):
        if k in ["_mask", "_random_mask", "specaugment_eval_func"]:
            funcs.append(v)
    return funcs