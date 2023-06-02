"""
Mixup as layer for TF net dict

https://arxiv.org/abs/1710.09412
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, Dict, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim


@dataclass
class MixupOpts:
    """
    Arguments:
        buffer_size: number of frames.
        apply_prob: probability to apply mixup at all
        max_num_mix: maximum number of mixups (random int in [1, max_num_mix])
        lambda_min: minimum lambda value
        lambda_max: maximum lambda value
    """

    buffer_size: int = 1_000_000
    apply_prob: float = 1.0
    max_num_mix: int = 4
    lambda_min: float = 0.1
    lambda_max: float = 0.4


def make_mixup_layer_dict(
    src: str,
    *,
    dim: Union[int, Dim],
    opts: MixupOpts,
) -> Dict[str, Any]:
    """
    :param src: source layer name
    :param dim: same as src
    :param opts:
    """
    return {
        "class": "subnetwork",
        "subnetwork": {
            "buffer": {
                "class": "variable",
                "shape": (opts.buffer_size, dim),
                "trainable": False,
                "saveable": False,
            },
            "buffer_pos": {
                "class": "variable",
                "dtype": "int32",
                "shape": (),
                "trainable": False,
                "saveable": False,
                "init": 0,
            },
            "buffer_filled": {
                "class": "variable",
                "dtype": "bool",
                "shape": (),
                "trainable": False,
                "saveable": False,
                "init": False,
            },
            "output": {
                "class": "eval",
                "from": [f"base:{src}", "buffer", "buffer_pos", "buffer_filled"],
                "eval": _mixup_eval_layer_func,
                "eval_locals": {"dim": dim, "opts": opts},
                "out_type": _mixup_eval_layer_out_type_func,
            },
        },
    }


def _mixup_eval_layer_func(*, source, dim: Union[int, Dim], opts: MixupOpts, self, **_kwargs):
    import tensorflow as tf
    from returnn.tensor import Tensor, Dim, batch_dim
    from returnn.tf.network import LayerBase

    assert isinstance(self, LayerBase)

    src = source(0, as_data=True, auto_convert=False)
    buffer = source(1, as_data=True, auto_convert=False)
    buffer_pos = source(2, as_data=True, auto_convert=False)
    buffer_filled = source(3, as_data=True, auto_convert=False)

    assert (
        isinstance(src, Tensor)
        and isinstance(buffer, Tensor)
        and isinstance(buffer_pos, Tensor)
        and isinstance(buffer_filled, Tensor)
    )
    assert src.dim == (dim.dimension if isinstance(dim, Dim) else dim)

    time_dim = src.get_time_dim_tag()
    feat_dim = src.feature_dim
    src = src.copy_transpose([batch_dim, time_dim, feat_dim])

    func = tf.function(_get_raw_func(dim=src.dim, opts=opts), autograph=True)

    return func(
        src.raw_tensor,
        src.get_sequence_lengths(),
        buffer.raw_tensor,
        buffer_pos.raw_tensor,
        buffer_filled.raw_tensor,
        tf.convert_to_tensor(self.network.train_flag),
    )


def _mixup_eval_layer_out_type_func(sources, name, **_kwargs):
    from returnn.tensor import Tensor, batch_dim

    src = sources[0].output
    assert isinstance(src, Tensor)
    src = src.copy_template(name="%s_output" % name)
    time_dim = src.get_time_dim_tag()
    feat_dim = src.feature_dim
    src = src.copy_transpose([batch_dim, time_dim, feat_dim])
    return src


def _get_raw_func(*, dim: int, opts: MixupOpts):
    def _raw_func(src_raw, src_seq_lens, buffer_raw, buffer_pos_raw, buffer_filled_raw, train_flag):
        import tensorflow as tf

        assert (
            isinstance(src_raw, tf.Tensor)
            and isinstance(src_seq_lens, tf.Tensor)
            and isinstance(buffer_raw, tf.Variable)
            and isinstance(buffer_pos_raw, tf.Variable)
            and isinstance(buffer_filled_raw, tf.Variable)
            and isinstance(train_flag, (bool, tf.Tensor))
        )

        if not train_flag:
            return src_raw

        dtype = src_raw.dtype
        n_batch = tf.shape(src_raw)[0]
        n_time = tf.shape(src_raw)[1]
        n_feat = dim

        # Fill buffer with new data:
        pos = buffer_pos_raw
        for b in tf.range(n_batch):
            new_pos = tf.minimum(pos + src_seq_lens[b], opts.buffer_size)
            part_fill_len = new_pos - pos
            tf.raw_ops.ResourceStridedSliceAssign(
                ref=buffer_raw.handle, begin=[pos], end=[new_pos], strides=[1], value=src_raw[b, :part_fill_len]
            )
            if part_fill_len <= src_seq_lens[b]:
                buffer_filled_raw.assign(True)
            if part_fill_len < src_seq_lens[b]:
                part_fill_len_ = tf.minimum(src_seq_lens[b] - part_fill_len, opts.buffer_size)
                tf.raw_ops.ResourceStridedSliceAssign(
                    ref=buffer_raw.handle,
                    begin=[0],
                    end=[part_fill_len_],
                    strides=[1],
                    value=src_raw[b, part_fill_len : part_fill_len + part_fill_len_],
                )
                new_pos = part_fill_len_
            pos = new_pos
        buffer_pos_raw.assign(pos)

        if tf.random.uniform(()) >= opts.apply_prob:
            return src_raw

        buffer_filled_size = opts.buffer_size if buffer_filled_raw else buffer_pos_raw
        if buffer_filled_size == 0:
            return src_raw

        # Apply Mixup. Collect all data we are going to add for each sequence.
        # Use TensorArray to iterate over the batch dim.
        ta = tf.TensorArray(dtype, size=n_batch, element_shape=(None, n_feat))  # [N] * [T, F]
        for b in tf.range(n_batch):
            num_mixup = tf.random.uniform((), minval=1, maxval=opts.max_num_mix + 1, dtype=tf.int32)
            mixup_values = tf.TensorArray(dtype, size=num_mixup, element_shape=(None, n_feat))  # [N] * [T, F]
            for n in tf.range(num_mixup):
                src_left = 0
                src_right = src_seq_lens[b]
                if buffer_filled_size <= src_seq_lens[b]:
                    src_left = tf.random.uniform(
                        (),
                        maxval=src_seq_lens[b] - buffer_filled_size + 1,
                        dtype=tf.int32,
                    )
                    src_right = src_left + buffer_filled_size
                src_size = src_right - src_left
                buffer_start = tf.random.uniform((), maxval=buffer_filled_size - src_size, dtype=tf.int32)
                buffer_end = buffer_start + src_size
                buffer_part = tf.raw_ops.ResourceGather(
                    resource=buffer_raw.handle, indices=tf.range(buffer_start, buffer_end), dtype=dtype
                )

                mixup_values = mixup_values.write(
                    n,
                    tf.concat(
                        [
                            tf.zeros((src_left, n_feat), dtype=dtype),
                            buffer_part,
                            tf.zeros((n_time - src_right, n_feat), dtype=dtype),
                        ],
                        axis=0,
                    ),
                )
            mixup_values = mixup_values.stack()  # [N, T, F]

            # Scale the mixup values.
            lambda_ = tf.random.uniform((), minval=opts.lambda_min, maxval=opts.lambda_max, dtype=dtype)
            mixup_scales = tf.random.uniform((num_mixup,), minval=0.001, maxval=1.0, dtype=dtype)
            mixup_scales *= lambda_ / tf.reduce_sum(mixup_scales)  # [N]
            mixup_values *= mixup_scales[:, None, None]

            mixup_value = tf.reduce_sum(mixup_values, axis=0)  # [T, F]
            ta = ta.write(b, mixup_value)

        mixup_value = ta.stack()  # [B,T,F]
        mixup_value.set_shape(src_raw.shape)
        src_raw = src_raw + mixup_value
        return src_raw

    return _raw_func
