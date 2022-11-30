"""
Modified SpecAugment helpers.

They are used because SpecAugment is modified in wav2vec 2.0 to statically sample the mask length.
"""

from dataclasses import dataclass

from returnn.import_ import import_
common = import_("github.com/rwth-i6/returnn_common", "models/base", "20210929-2243f105ba0befb2eba63f53a2350d4e26639532")
from returnn_import.github_com.rwth_i6.returnn_common.v20210929142536_2243f105ba0b.models.base import Module
import returnn_import.github_com.rwth_i6.returnn_common.v20210929142536_2243f105ba0b.models._generated_layers as layers


def _mask(x, batch_axis, axis, pos, max_amount, sampler):
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
    if sampler == 'uniform':
        amount = tf.random_uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
    elif sampler == 'static':
        amount = max_amount
    else:
        assert False, f"sampling method {sampler} not implemented"
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


def _random_mask(x, batch_axis, axis, min_num, max_num, max_dims, sampler):
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
            x = _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims, sampler=sampler)
    else:
        _, x = tf.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
            body=lambda i, x: (
                i + 1,
                tf.where(
                    tf.less(i, num),
                    _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims, sampler=sampler),
                    x)),
            loop_vars=(0, x))
    return x


def specaugment_eval_func(data, network,
                          min_mask_each_n_frames=2, max_mask_each_n_frames=25,
                          max_frames_per_mask=20, frames_per_mask_sampler='uniform',
                          min_feature_masks=2, max_feature_masks=5,
                          max_features_per_mask=8, features_per_mask_sampler='uniform'):
    x = data.placeholder
    from returnn.tf.compat import v1 as tf
    def get_masked():
        x_masked = x
        x_masked = _random_mask(
            x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
            min_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // min_mask_each_n_frames, 0),
            max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // max_mask_each_n_frames, 0),
            max_dims=max_frames_per_mask, sampler=frames_per_mask_sampler)
        x_masked = _random_mask(
            x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
            min_num=min_feature_masks, max_num=max_feature_masks,
            max_dims=max_features_per_mask, sampler=features_per_mask_sampler)
        return x_masked
    x = network.cond_on_train(get_masked, lambda: x)
    return x


@dataclass(eq=False, frozen=True)
class SpecAugmentSettings:
    min_mask_each_n_frames: int = 2
    max_mask_each_n_frames: int = 25
    max_frames_per_mask: int = 20
    min_feature_masks: int = 2
    max_feature_masks: int = 5
    max_features_per_mask: int = 8

    def get_options(self):
        return self.__dict__


class SpecAugment(Module):

    def __init__(self, min_mask_each_n_frames=2000, max_mask_each_n_frames=25,
                    max_frames_per_mask=20, frames_per_mask_sampler='uniform',
                    min_feature_masks=2, max_feature_masks=5,
                    max_features_per_mask=8, features_per_mask_sampler='uniform'):
        """
        :param int min_frame_mask_ratio: lower limit for random number of masks on the time axis
                                           in relation to the number of frames in the sequence
        :param int max_frame_mask_ratio: upper limit for random number of masks on the time axis
                                           in relation to the number of frames in the sequence
        :param int max_frames_per_mask: maximum number of feature dimensions covered per mask on the feature axis
        :param string frames_per_mask_sampler (uniform, static): how to sample length of frame mask
        :param int min_feature_masks: lower limit for random number of masks on the time axis
        :param int max_feature_masks: upper limit for random number of masks on the feature axis
        :param int max_features_per_mask: maximum number of feature dimensions covered per mask on the feature axis
        :param string features_per_mask_sampler (uniform, static): how to sample length of feature mask
        """
        super().__init__()

        self.eval_layer = layers.Eval(
            eval=("self.network.get_config().typed_value('specaugment_eval_func')("
                  "source(0, as_data=True), "
                  "network=self.network, "
                  "min_mask_each_n_frames=%i, "
                  "max_mask_each_n_frames=%i, "
                  "max_frames_per_mask=%i, "
                  "frames_per_mask_sampler='%s', " 
                  "min_feature_masks=%i, "
                  "max_feature_masks=%i, "
                  "max_features_per_mask=%i, "
                  "features_per_mask_sampler='%s')" % (
                      min_mask_each_n_frames, max_mask_each_n_frames,
                      max_frames_per_mask, frames_per_mask_sampler,
                      min_feature_masks, max_feature_masks,
                      max_features_per_mask, features_per_mask_sampler
                  ))
            )

    def forward(self, inp):
        out = self.eval_layer(inp)
        return out

def get_funcs():
    funcs = []
    for k, v in list(globals().items()):
        if k in ["_mask", "_random_mask", "specaugment_eval_func"]:
            funcs.append(v)
    return funcs
