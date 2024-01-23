"""
Mixup as layer for TF net dict

https://arxiv.org/abs/1710.09412
"""

from __future__ import annotations
from typing import Dict, Any

from i6_core.returnn.config import CodeWrapper


def make_mixup_layer_dict(
    src: str,
    *,
    dim: int,
    opts: dict,
    is_recog: bool = False,
) -> Dict[str, Any]:
    """
    :param src: source layer name
    :param dim: same as src
    :param opts: mixup opts
    :param is_recog: whether this is a recognition net
    """
    d = {}
    d["mixup"] = {
        "class": "subnetwork",
        "subnetwork": {
            "buffer": {
                "class": "variable",
                "shape": (opts["buffer_size"], dim),
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
                "eval": CodeWrapper("get_global_config().typed_value('_mixup_eval_layer_func')")
                if not is_recog
                else CodeWrapper("_mixup_eval_layer_func"),
                "eval_locals": {"dim": dim, "opts": opts},
                "out_type": CodeWrapper("get_global_config().typed_value('_mixup_eval_layer_out_type_func')")
                if not is_recog
                else CodeWrapper("_mixup_eval_layer_out_type_func"),
            },
        },
    }
    return d


def _mixup_eval_layer_func(*, source, dim: int, opts: dict, self, **_kwargs):
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
    assert src.dim == dim

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


def _get_raw_func(*, dim: int, opts: dict):
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
        t_mask = tf.sequence_mask(src_seq_lens, maxlen=n_time)  # [B,T]
        src_flat = tf.boolean_mask(src_raw, t_mask)  # [B_T',F]
        src_flat_len = tf.shape(src_flat)[0]  # B_T'
        pos = buffer_pos_raw
        new_pos = tf.minimum(pos + src_flat_len, opts["buffer_size"])
        part_fill_len = new_pos - pos
        tf.raw_ops.ResourceStridedSliceAssign(
            ref=buffer_raw.handle, begin=[pos], end=[new_pos], strides=[1], value=src_flat[:part_fill_len]
        )
        if pos + src_flat_len >= opts["buffer_size"]:
            buffer_filled_raw.assign(True)
            part_fill_len_ = tf.minimum(src_flat_len - part_fill_len, opts["buffer_size"])
            tf.raw_ops.ResourceStridedSliceAssign(
                ref=buffer_raw.handle,
                begin=[0],
                end=[part_fill_len_],
                strides=[1],
                value=src_flat[part_fill_len : part_fill_len + part_fill_len_],
            )
            new_pos = part_fill_len_
        buffer_pos_raw.assign(new_pos)

        if tf.random.uniform(()) >= opts["apply_prob"]:
            return src_raw

        buffer_filled_size = opts["buffer_size"] if buffer_filled_raw else buffer_pos_raw
        if buffer_filled_size < n_time:
            return src_raw

        # Apply Mixup. Collect all data we are going to add for each sequence.
        num_mixup = tf.random.uniform([n_batch], minval=1, maxval=opts["max_num_mix"] + 1, dtype=tf.int32)  # [B]
        max_num_mix = tf.reduce_max(num_mixup)  # N

        buffer_start = tf.random.uniform(
            [n_batch, max_num_mix], maxval=buffer_filled_size - n_time + 1, dtype=tf.int32
        )  # [B, N]
        n_mask = tf.sequence_mask(num_mixup, maxlen=max_num_mix)  # [B, N]
        buffer_start_flat = tf.boolean_mask(buffer_start, n_mask)  # [B_N']

        idx = tf.range(n_time)[None, :]  # [1, T]
        idx = idx + buffer_start_flat[:, None]  # [B_N', T]

        mixup_values = tf.raw_ops.ResourceGather(resource=buffer_raw.handle, indices=idx, dtype=dtype)  # [B_N', T, F]

        # Scale the mixup values.
        lambda_ = tf.random.uniform(
            [n_batch, max_num_mix], minval=opts["lambda_min"], maxval=opts["lambda_max"], dtype=dtype
        )
        mixup_scales = tf.random.uniform([n_batch, max_num_mix], minval=0.001, maxval=1.0, dtype=dtype)
        mixup_scales *= lambda_ / tf.reduce_sum(mixup_scales, axis=1, keepdims=True)  # [B,N]
        mixup_scales_flat = tf.boolean_mask(mixup_scales, n_mask)  # [B_N']
        mixup_values *= mixup_scales_flat[:, None, None]  # [B_N', T, F]

        idx_b = tf.range(n_batch)[:, None]  # [B,1]
        idx_b = tf.tile(idx_b, [1, max_num_mix])  # [B,N]
        idx_b = tf.boolean_mask(idx_b, n_mask)  # [B_N']

        mixup_value = tf.scatter_nd(
            indices=idx_b[:, None], updates=mixup_values, shape=[n_batch, n_time, n_feat]
        )  # [B,T,F]

        src_raw = src_raw + tf.stop_gradient(mixup_value)
        return src_raw

    return _raw_func
