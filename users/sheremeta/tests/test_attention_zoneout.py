import torch

import returnn.frontend as rf
from returnn.config import Config, global_config_ctx
from returnn.frontend import _packed_backend as packed
from returnn.frontend.encoder.conformer import ConformerEncoder
from returnn.tensor import Dim, Tensor
from returnn.util import BehaviorVersion

from i6_experiments.users.sheremeta.nn_rf.encoder.attention_zoneout import (
    AttentionZoneoutRelPosSelfAttention,
    AttentionZoneoutSequential,
)

rf.select_backend_torch()
if BehaviorVersion.get_if_set() is None:
    BehaviorVersion.set_min_behavior_version(31)

_PROB = 0.4


def _source(seed: int, model_dim: Dim, dims=None):
    """
    :param dims: (batch, time) to reuse, new ones if not given
    :return: random x [batch, time, model], batch dim, time dim
    """
    if dims is None:
        batch_dim = Dim(2, name="batch")
        lens = Tensor("lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([7, 5], dtype=torch.int32))
        dims = (batch_dim, Dim(lens, name="time"))
    batch_dim, time_dim = dims
    x = Tensor("x", dims=[batch_dim, time_dim, model_dim], dtype="float32")
    x.raw_tensor = torch.randn(2, 7, model_dim.dimension, generator=torch.Generator().manual_seed(seed))
    return x, batch_dim, time_dim


def _valid(out: Tensor, batch_dim: Dim, time_dim: Dim, model_dim: Dim) -> torch.Tensor:
    if packed.is_packed(out):
        out = packed.unpack(out)
    values = out.copy_transpose([batch_dim, time_dim, model_dim]).raw_tensor
    mask = rf.sequence_mask([batch_dim, time_dim]).copy_transpose([batch_dim, time_dim]).raw_tensor
    return values[mask]


def _fixed_keep(pattern):
    """
    :param pattern: function from the dim sizes to a bool torch tensor
    :return: a draw_keep which returns the pattern
    """

    def draw_keep(dims, *, device=None):
        return Tensor("keep", dims=dims, dtype="bool", raw_tensor=pattern([d.get_dim_value() for d in dims]))

    return draw_keep


def test_attention_keeps_the_layer_below_or_the_frame_before():
    """
    Every mode in training with all, none or a fixed pattern of keeps, and the expected output in evaluation.
    """
    model_dim = Dim(16, name="model")
    x, batch_dim, time_dim = _source(0, model_dim)
    x_below, _, _ = _source(1, model_dim, (batch_dim, time_dim))
    for mode in ("head", "row", "time"):
        rf.set_random_seed(0)
        att = AttentionZoneoutRelPosSelfAttention(
            model_dim,
            model_dim,
            key_dim_total=model_dim,
            value_dim_total=model_dim,
            num_heads=2,
            att_dropout=0.0,
            zoneout_mode=mode,
            zoneout_prob=_PROB,
        )
        q_below, k_below, _ = att.forward_qkv(x_below)
        _, _, v = att.forward_qkv(x)
        rf.init_train_step_run_ctx(train_flag=False, step=0, epoch=1)
        new = _valid(rf.RelPosSelfAttention.__call__(att, x, axis=time_dim), batch_dim, time_dim, model_dim)
        below = _valid(att._attend(x, q_below, k_below, v, axis=time_dim), batch_dim, time_dim, model_dim)

        def run(train: bool, pattern):
            rf.init_train_step_run_ctx(train_flag=train, step=0, epoch=1)
            att.draw_keep = _fixed_keep(pattern)
            att.qk_below = (q_below, k_below)
            return _valid(att(x, axis=time_dim), batch_dim, time_dim, model_dim)

        torch.testing.assert_close(run(True, lambda shape: torch.zeros(shape, dtype=torch.bool)), new)
        if mode == "time":
            # frames 1, 2, 4 and 5 of the first sequence and frame 3 of the second keep the frame before
            keep = torch.tensor([[0, 1, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 0, 0]], dtype=torch.bool)
            src = [0, 0, 0, 3, 3, 3, 6, 7 + 0, 7 + 1, 7 + 2, 7 + 2, 7 + 4]
            torch.testing.assert_close(run(True, lambda shape: keep), new[src])
            torch.testing.assert_close(run(False, lambda shape: keep), new)
        else:
            torch.testing.assert_close(run(True, lambda shape: torch.ones(shape, dtype=torch.bool)), below)
            expected = _PROB * below + (1.0 - _PROB) * new
            torch.testing.assert_close(run(False, lambda shape: torch.ones(shape, dtype=torch.bool)), expected)


def test_packed_encoder_matches_padded():
    """
    The encoder with attention zoneout gives the same frames packed and padded.
    The zoneout ops have no fallback, the attention itself unpacks on cpu, which has no packed rel-pos path in training.
    """
    model_dim = Dim(16, name="model")
    x, batch_dim, time_dim = _source(2, model_dim)

    class PatternKeepAttention(AttentionZoneoutRelPosSelfAttention):
        draw_keep = staticmethod(
            _fixed_keep(lambda shape: (torch.arange(torch.Size(shape).numel()) % 3 == 1).reshape(shape))
        )

    class NoKeepAttention(AttentionZoneoutRelPosSelfAttention):
        draw_keep = staticmethod(_fixed_keep(lambda shape: torch.zeros(shape, dtype=torch.bool)))

    def encoder(mode: str, att_class: type) -> ConformerEncoder:
        rf.set_random_seed(0)
        return ConformerEncoder(
            model_dim,
            model_dim,
            num_layers=3,
            input_layer=None,
            input_dropout=0.0,
            ff_dim=Dim(32, name="ff"),
            dropout=0.0,
            conv_kernel_size=3,
            conv_norm=rf.LayerNorm,
            num_heads=2,
            att_dropout=0.0,
            encoder_layer_opts=dict(self_att=att_class, self_att_opts=dict(zoneout_mode=mode, zoneout_prob=_PROB)),
            sequential=AttentionZoneoutSequential,
        )

    rf.init_train_step_run_ctx(train_flag=True, step=0, epoch=1)
    with global_config_ctx(Config(dict(packed_fallback_allowed=["rel_pos_self_attention"]))):
        for mode in ("head", "row", "time"):
            got, _ = encoder(mode, PatternKeepAttention)(
                packed.pack(x, dims=[batch_dim, time_dim]), in_spatial_dim=time_dim
            )
            assert packed.is_packed(got)
            got = _valid(got, batch_dim, time_dim, model_dim)
            want, _ = encoder(mode, PatternKeepAttention)(x, in_spatial_dim=time_dim)
            want = _valid(want, batch_dim, time_dim, model_dim)
            torch.testing.assert_close(got, want)
            plain, _ = encoder(mode, NoKeepAttention)(x, in_spatial_dim=time_dim)
            assert (want - _valid(plain, batch_dim, time_dim, model_dim)).abs().max() > 1e-3, mode


def test_chunked_attention_keeps_the_layer_below_or_the_frame_before_in_the_chunk():
    """
    The chunked attention in every mode with all, none or a fixed pattern of keeps, and the expected output in
    evaluation. Time zoneout takes the frame before within a chunk, the first frame of a chunk never keeps.
    """
    from i6_experiments.users.sheremeta.nn_rf.encoder.chunked_conformer import (
        kernel_attention_class,
        kernel_attention_zoneout_class,
    )

    batch_dim = Dim(2, name="batch")
    lens = Tensor("chunk_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([3, 2], dtype=torch.int32))
    chunked_time_dim = Dim(lens, name="chunked_time")
    chunk_dim, end_chunk_dim, model_dim = Dim(6, name="chunk"), Dim(4, name="end_chunk"), Dim(16, name="model")
    dims = [batch_dim, chunked_time_dim, chunk_dim, model_dim]
    x, x_below = (Tensor(name, dims=dims, dtype="float32") for name in ("x", "x_below"))
    x.raw_tensor, x_below.raw_tensor = torch.randn(2, 2, 3, 6, 16, generator=torch.Generator().manual_seed(4))

    def valid(out: Tensor) -> torch.Tensor:
        values = out.copy_transpose(dims).raw_tensor
        return values[rf.sequence_mask([batch_dim, chunked_time_dim]).copy_transpose(dims[:2]).raw_tensor]

    for mode in ("head", "row", "time"):
        rf.set_random_seed(0)
        att = kernel_attention_zoneout_class()(
            chunk_history=2,
            end_chunk_size_dim=end_chunk_dim,
            in_dim=model_dim,
            proj_dim=model_dim,
            key_dim_total=model_dim,
            value_dim_total=model_dim,
            num_heads=2,
            att_dropout=0.0,
            zoneout_mode=mode,
            zoneout_prob=_PROB,
        )
        call = dict(axis=chunk_dim, chunked_time_dim=chunked_time_dim)
        q_below, k_below, _ = att.forward_qkv(x_below)
        _, _, v = att.forward_qkv(x)
        rf.init_train_step_run_ctx(train_flag=False, step=0, epoch=1)
        new = valid(kernel_attention_class().__call__(att, x, **call))
        below = valid(att._attend(x, q_below, k_below, v, **call))

        def run(train: bool, pattern):
            rf.init_train_step_run_ctx(train_flag=train, step=0, epoch=1)
            att.draw_keep = _fixed_keep(pattern)
            att.qk_below = (q_below, k_below)
            return valid(att(x, **call))

        torch.testing.assert_close(run(True, lambda shape: torch.zeros(shape, dtype=torch.bool)), new)
        if mode == "time":
            # frames 1, 2 and 4 of every chunk keep the frame before
            keep = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.bool)
            torch.testing.assert_close(run(True, lambda shape: keep.expand(shape)), new[:, [0, 0, 0, 3, 3, 5]])
            torch.testing.assert_close(run(False, lambda shape: keep.expand(shape)), new)
        else:
            torch.testing.assert_close(run(True, lambda shape: torch.ones(shape, dtype=torch.bool)), below)
            expected = _PROB * below + (1.0 - _PROB) * new
            torch.testing.assert_close(run(False, lambda shape: torch.ones(shape, dtype=torch.bool)), expected)


def test_packed_chunked_encoder_matches_padded():
    """
    The chunked encoder with attention zoneout gives the same frames packed and padded, and other frames than
    without any keep.
    """
    from i6_experiments.users.sheremeta.nn_rf.encoder.chunked_conformer import kernel_attention_zoneout_class
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v1 import ChunkedConformerEncoder

    model_dim = Dim(16, name="model")
    batch_dim = Dim(2, name="batch")
    lens = Tensor("lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([13, 9], dtype=torch.int32))
    time_dim = Dim(lens, name="time")
    x = Tensor("x", dims=[batch_dim, time_dim, model_dim], dtype="float32")
    x.raw_tensor = torch.randn(2, 13, 16, generator=torch.Generator().manual_seed(6))

    def encoder(mode: str, pattern) -> ChunkedConformerEncoder:
        class FixedKeepAttention(kernel_attention_zoneout_class()):
            draw_keep = staticmethod(_fixed_keep(pattern))

        rf.set_random_seed(0)
        end_chunk_dim = Dim(4, name="end_chunk")
        att_opts = dict(chunk_history=2, end_chunk_size_dim=end_chunk_dim, zoneout_mode=mode, zoneout_prob=_PROB)
        enc = ChunkedConformerEncoder(
            model_dim,
            model_dim,
            num_layers=3,
            input_layer=None,
            input_dropout=0.0,
            ff_dim=Dim(32, name="ff"),
            dropout=0.0,
            conv_kernel_size=3,
            conv_norm=rf.LayerNorm,
            num_heads=2,
            att_dropout=0.0,
            encoder_layer_opts=dict(self_att=FixedKeepAttention, self_att_opts=att_opts),
            input_chunk_size_dim=6,
            chunk_stride=4,
            chunk_history=2,
            end_chunk_size_dim=end_chunk_dim,
        )
        enc.layers.__class__ = AttentionZoneoutSequential
        return enc

    def frames(out: Tensor, out_dim: Dim) -> torch.Tensor:
        if packed.is_packed(out):
            out = packed.unpack(out)
        return _valid(out, batch_dim, out_dim, model_dim)

    def some(shape):
        return (torch.arange(torch.Size(shape).numel()) % 3 == 1).reshape(shape)

    rf.init_train_step_run_ctx(train_flag=True, step=0, epoch=1)
    with global_config_ctx(Config(dict(packed_fallback_allowed=["rel_pos_self_attention"]))):
        for mode in ("head", "row", "time"):
            got, got_dim = encoder(mode, some)(packed.pack(x, dims=[batch_dim, time_dim]), in_spatial_dim=time_dim)
            assert packed.is_packed(got)
            want, want_dim = encoder(mode, some)(x, in_spatial_dim=time_dim)
            torch.testing.assert_close(frames(got, got_dim), frames(want, want_dim))
            plain, plain_dim = encoder(mode, lambda shape: torch.zeros(shape, dtype=torch.bool))(
                x, in_spatial_dim=time_dim
            )
            assert (frames(want, want_dim) - frames(plain, plain_dim)).abs().max() > 1e-3, mode
