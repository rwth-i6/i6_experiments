"""
run:
export PYTHONPATH=recipe:ext/returnn
python -m i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2_test

Real-world examples:
i6_experiments/users/zeyer/experiments/exp2025_10_21_chunked_ctc.py

Similar tests, see: RETURNN tests.test_rf_encoder_conformer.test_e_branchformer,
or other RF tests in RETURNN.
"""

from typing import Dict, Any, Tuple

import torch

from returnn.util import BehaviorVersion, better_exchook
from returnn.util.debug import PyTracer, check_py_traces_rf_to_pt_equal
import returnn.frontend as rf
from returnn.tensor import Dim, Tensor, batch_dim

from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v1 import (
    ChunkedConformerEncoder,
    ChunkedConformerEncoderLayer,
)
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
    ChunkedConformerEncoderV2,
    ChunkedConformerEncoderLayerV2,
    ChunkedRelPosSelfAttentionV2,
    ChunkedRotaryPosSelfAttentionV2,
    _average_overlapping_chunks,
)

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep

_log_mel_feature_dim = 80
feat_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)


def _setup_test():
    BehaviorVersion.set_min_behavior_version(25)
    rf.select_backend_torch()
    if batch_dim.dyn_size_ext is None:
        batch_dim.dyn_size_ext = rf.convert_to_tensor(3, dims=[])


def tests():
    better_exchook.install()
    _setup_test()
    test_conformer_v2()
    test_conformer_v2_with_chunk_num_overlaps()
    test_average_overlapping_chunks()


def bench_rope_vs_relpos():
    """
    Benchmark :class:`ChunkedRelPosSelfAttentionV2` vs :class:`ChunkedRotaryPosSelfAttentionV2`.

    With ``--bench-profile`` the function also runs a per-operation timing breakdown to show
    exactly where time is spent inside each attention variant.
    """
    import torch.utils.benchmark as benchmark

    better_exchook.install()
    _setup_test()
    batch_dim.dyn_size_ext = rf.convert_to_tensor(32, dims=[])

    chunk_size = 10
    chunk_history_size = 80
    chunk_lookahead_size = 4

    def _build(self_att_cls):
        return _build_model(
            rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(
                    ChunkedConformerEncoderLayerV2,
                    self_att=rf.build_dict(self_att_cls),
                ),
                chunk_size=chunk_size,
                chunk_history_size=chunk_history_size,
                chunk_lookahead_size=chunk_lookahead_size,
                version=3,
                adapt_chunk_history_for_short_seqs=False,
            )
        )

    model_relpos = _build(ChunkedRelPosSelfAttentionV2)
    model_rope = _build(ChunkedRotaryPosSelfAttentionV2)

    input_data, time_dim = _make_input_data(seq_len=1001)

    def _bench(model, label):
        t = benchmark.Timer(
            stmt="with torch.no_grad(): model(input_data, in_spatial_dim=time_dim)",
            globals={"torch": torch, "model": model, "input_data": input_data, "time_dim": time_dim},
            label=label,
            num_threads=1,
        )
        m = t.blocked_autorange()
        print(f"  {label:42s}  median={m.median * 1000:.2f}ms  mean={m.mean * 1000:.2f}ms")
        return m.median

    print("\n=== bench_rope_vs_relpos ===")
    median_relpos = _bench(model_relpos, "ChunkedRelPosSelfAttentionV2")
    median_rope = _bench(model_rope, "ChunkedRotaryPosSelfAttentionV2")
    print(
        f"  rope / relpos  = {median_rope / median_relpos:.2f}x  "
        f"(rope {'slower' if median_rope > median_relpos else 'faster'} than relpos)"
    )
    print("=== done ===\n")

    if "--bench-profile" in sys.argv:
        _bench_rope_vs_relpos_profile(input_data, time_dim, chunk_size, chunk_history_size)


def _bench_rope_vs_relpos_profile(input_data: Tensor, time_dim: Dim, chunk_size: int, chunk_history_size: int):
    """
    Per-operation timing breakdown showing where time is spent inside each attention variant.

    ``_apply_rope`` (torch backend, torch.compile-fused) is faster than ``_apply_rope_real``
    (plain RF ops) because torch.compile fuses the element-wise operations into a single kernel.
    """
    import torch.utils.benchmark as benchmark
    from returnn.frontend.attention import _apply_rope, sinusoidal_positional_encoding

    chunk_lookahead_size = 4
    ds = 6
    chunk_stride = chunk_size
    chunk_history = chunk_history_size // chunk_stride
    input_chunk_size = (chunk_size + chunk_lookahead_size) * ds
    input_chunk_stride = chunk_stride * ds

    input_chunk_size_dim = Dim(input_chunk_size, name="input_chunk_size_profile")
    end_chunk_size_dim = Dim(chunk_stride, name="chunk_stride_profile")

    # Build a single model to get the attention module and realistic tensors.
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
        _mem_chunks,
        _BatchChunkingSettings,
    )
    from returnn.frontend.encoder.conformer import ConformerConvSubsample

    out_dim = Dim(64, name="model_profile")
    num_heads = 8
    head_dim = Dim(out_dim.dimension // num_heads, name="head_dim_profile")

    att_relpos = ChunkedRelPosSelfAttentionV2(
        in_dim=out_dim,
        proj_dim=out_dim,
        key_dim_total=out_dim,
        value_dim_total=out_dim,
        num_heads=num_heads,
        att_dropout=0.0,
        version=3,
    )
    att_rope = ChunkedRotaryPosSelfAttentionV2(
        in_dim=out_dim,
        proj_dim=out_dim,
        key_dim_total=out_dim,
        value_dim_total=out_dim,
        num_heads=num_heads,
        att_dropout=0.0,
        version=3,
    )

    # Build a minimal ConformerConvSubsample to produce a realistic source tensor.
    subsample = ConformerConvSubsample(
        feat_dim,
        out_dims=[32, 64, 64],
        filter_sizes=[(3, 3), (3, 3), (3, 3)],
        pool_sizes=[(1, 2)],
        strides=[(1, 1), (3, 1), (2, 1)],
    )

    # Produce realistic windowed + subsampled input.
    source_windowed, chunked_time_dim = rf.window(
        input_data,
        spatial_dim=time_dim,
        window_dim=input_chunk_size_dim,
        window_left=0,
        stride=input_chunk_stride,
        pad_value=0.0,
    )
    source_sub, enc_spatial_dim = subsample(source_windowed, in_spatial_dim=input_chunk_size_dim)
    # Project to out_dim.
    proj = rf.Linear(subsample.out_dim, out_dim, with_bias=False)
    x = proj(source_sub)

    chunking = _BatchChunkingSettings(
        chunk_history=chunk_history,
        end_chunk_size_dim=end_chunk_size_dim,
        chunked_time_dim=chunked_time_dim,
    )

    axis = enc_spatial_dim
    rope_base = 10_000 ** (1 - 2 / head_dim.dimension)
    query_offset = chunk_history * end_chunk_size_dim.dimension

    q_relpos, k_relpos, v_relpos = att_relpos.forward_qkv(x)
    q_rope, k_rope, v_rope = att_rope.forward_qkv(x)

    hist_dim = Dim(None, name="kv_profile")
    k_relpos_r, _ = rf.replace_dim(k_relpos, in_dim=axis, out_dim=hist_dim)
    k_ext, hist_dim_ = _mem_chunks(
        k_relpos_r,
        spatial_dim=hist_dim,
        chunked_time_dim=chunked_time_dim,
        mem_size=chunk_history,
        end_chunk_size_dim=end_chunk_size_dim,
    )

    hist_dim2 = Dim(None, name="kv_rope_profile")
    k_rope_r, _ = rf.replace_dim(k_rope, in_dim=axis, out_dim=hist_dim2)
    k_ext_rope, hist_dim_rope_ = _mem_chunks(
        k_rope_r,
        spatial_dim=hist_dim2,
        chunked_time_dim=chunked_time_dim,
        mem_size=chunk_history,
        end_chunk_size_dim=end_chunk_size_dim,
    )

    # Use the actual key_dim_per_head from the modules.
    kd = att_rope.key_dim_per_head  # e.g. Dim(8) for 64/8

    def _bench(fn, label, *, with_grad: bool = False, **extra_globals):
        """Time fn() using torch.utils.benchmark (handles warmup and statistics)."""
        stmt = "fn()" if with_grad else "with torch.no_grad(): fn()"
        t = benchmark.Timer(
            stmt=stmt,
            globals={"torch": torch, "fn": fn, **extra_globals},
            label=label,
            num_threads=1,
        )
        m = t.blocked_autorange()
        return m.median * 1e6  # µs

    print("\n=== per-op profile (median µs, torch.utils.benchmark) ===")

    # -- _apply_rope on q (axis frames) and k_ext (kv frames) --
    static_axis_dim = Dim(axis.dimension, name="axis_static_profile")
    static_kv_dim = Dim(hist_dim_rope_.dimension, name="kv_static_profile")
    pos_enc_q = rf.sinusoidal_positional_encoding(
        spatial_dim=static_axis_dim, feat_dim=kd, base=rope_base, device="cpu"
    )
    pos_enc_q, _ = rf.replace_dim(pos_enc_q, in_dim=static_axis_dim, out_dim=axis)
    q_for_rope, _, _ = att_rope.forward_qkv(x)
    # noinspection PyProtectedMember
    from returnn.frontend.attention import _apply_rope, _apply_rope_real

    t = _bench(lambda: _apply_rope(q_for_rope, pos_enc_q, kd), "_apply_rope compiled q")
    print(f"  _apply_rope       (q,     T={axis.dimension:2d}) [torch.compile]                         {t:7.1f} µs")
    t = _bench(lambda: _apply_rope_real(q_for_rope, pos_enc_q, kd), "_apply_rope_real q")
    print(f"  _apply_rope_real  (q,     T={axis.dimension:2d}) [RF ops]                                {t:7.1f} µs")

    pos_enc_k = rf.sinusoidal_positional_encoding(
        spatial_dim=static_kv_dim, feat_dim=kd, offset=-query_offset, base=rope_base, device="cpu"
    )
    pos_enc_k, _ = rf.replace_dim(pos_enc_k, in_dim=static_kv_dim, out_dim=hist_dim_rope_)
    t = _bench(lambda: _apply_rope(k_ext_rope, pos_enc_k, kd), "_apply_rope compiled k_ext")
    print(
        f"  _apply_rope       (k_ext, T={hist_dim_rope_.dimension:2d}) [torch.compile]                         {t:7.1f} µs"
    )
    t = _bench(lambda: _apply_rope_real(k_ext_rope, pos_enc_k, kd), "_apply_rope_real k_ext")
    print(
        f"  _apply_rope_real  (k_ext, T={hist_dim_rope_.dimension:2d}) [RF ops]                                {t:7.1f} µs"
    )

    # -- RelPos components --
    t = _bench(
        lambda: rf.relative_positional_encoding(
            query_spatial_dim=axis,
            key_value_spatial_dim=hist_dim_,
            feat_dim=att_relpos.pos_emb_feat_dim,
            query_offset=query_offset,
            device="cpu",
        ),
        "relative_positional_encoding",
    )
    print(
        f"  relative_positional_encoding(q={axis.dimension}, kv={hist_dim_.dimension})                       {t:7.1f} µs"
    )

    pos_emb, pos_emb_dim = rf.relative_positional_encoding(
        query_spatial_dim=axis,
        key_value_spatial_dim=hist_dim_,
        feat_dim=att_relpos.pos_emb_feat_dim,
        query_offset=query_offset,
        device="cpu",
    )
    if att_relpos.linear_pos is not None:
        pos_emb_proj = att_relpos.linear_pos(pos_emb)
    else:
        pos_emb_proj = pos_emb
    if att_relpos.separate_pos_emb_per_head:
        pos_emb_proj = rf.split_dims(
            pos_emb_proj, axis=att_relpos.key_dim_total, dims=(att_relpos.num_heads, att_relpos.key_dim_per_head)
        )
    q_bias_v = q_relpos + att_relpos.pos_bias_v if att_relpos.pos_bias_v is not None else q_relpos
    t = _bench(lambda: rf.matmul(q_bias_v, pos_emb_proj, reduce=att_relpos.key_dim_per_head), "matmul matrix_bd")
    print(f"  matmul matrix_bd  (q={axis.dimension}×pos={pos_emb_dim.dimension})                           {t:7.1f} µs")

    q_bias_u = q_relpos + att_relpos.pos_bias_u if att_relpos.pos_bias_u is not None else q_relpos
    t = _bench(lambda: rf.matmul(q_bias_u, k_ext, reduce=att_relpos.key_dim_per_head), "matmul matrix_ac")
    print(f"  matmul matrix_ac  (q={axis.dimension}×kv={hist_dim_.dimension})                              {t:7.1f} µs")

    # -- full attention call --
    t = _bench(lambda: att_relpos(x, axis=axis, chunking=chunking), "ChunkedRelPosSelfAttentionV2")
    print(f"  ChunkedRelPosSelfAttentionV2  total                                  {t:7.1f} µs")

    t = _bench(lambda: att_rope(x, axis=axis, chunking=chunking), "ChunkedRotaryPosSelfAttentionV2")
    print(f"  ChunkedRotaryPosSelfAttentionV2 total                                {t:7.1f} µs")

    # -- _apply_rope scaling with T --
    # Use a large fixed batch so even small T is computation-dominated (not overhead-dominated).
    batch_s = 256
    print(f"\n  --- _apply_rope scaling with T (batch={batch_s}, heads={num_heads}, head_dim={head_dim.dimension}) ---")
    print(f"  {'T':>6}  {'compiled':>10}  {'RF ops':>10}  {'speedup':>8}")
    from returnn.frontend.attention import _apply_rope, _apply_rope_real

    from returnn.torch.util.rope import apply_rope as _rope_compiled_raw

    batch_dim_s = Dim(batch_s, name="batch_s")
    for T in [14, 94, 200, 500, 1000, 2000, 5000]:
        t_dim = Dim(T, name=f"T{T}")
        heads_dim_s = Dim(num_heads, name="heads_s")
        x_s = rf.random_uniform([batch_dim_s, t_dim, heads_dim_s, head_dim])
        pe_s = rf.sinusoidal_positional_encoding(spatial_dim=t_dim, feat_dim=head_dim)
        t_c = _bench(lambda: _apply_rope(x_s, pe_s, head_dim), f"compiled T={T}")
        t_r = _bench(lambda: _apply_rope_real(x_s, pe_s, head_dim), f"RF T={T}")
        print(f"  {T:6d}  {t_c:9.1f}µ  {t_r:9.1f}µ  {t_r / t_c:7.1f}x")

    # -- fwd+bwd scaling with T --
    # head_dim is already last in x_s, so no movedim needed.
    # We pre-allocate leaf tensors with requires_grad once per T; grads accumulate across
    # benchmark iterations but that does not affect timing.
    print(f"\n  --- _apply_rope fwd+bwd scaling with T (batch={batch_s}, heads={num_heads}, head_dim={head_dim.dimension}) ---")
    print(f"  {'T':>6}  {'compiled':>10}  {'RF ops':>10}  {'speedup':>8}")
    for T in [14, 94, 200, 500, 1000, 2000, 5000]:
        t_dim = Dim(T, name=f"T_bwd{T}")
        heads_dim_s = Dim(num_heads, name="heads_s_bwd")
        x_s = rf.random_uniform([batch_dim_s, t_dim, heads_dim_s, head_dim])
        pe_s = rf.sinusoidal_positional_encoding(spatial_dim=t_dim, feat_dim=head_dim)
        # head_dim is already last → no movedim; pe broadcast-aligned to x_s dims
        pe_raw = pe_s.copy_compatible_to_dims_raw(x_s.dims)
        # leaf tensor for compiled (raw): reused across iterations, grad just accumulates
        x_leaf_c = x_s.raw_tensor.clone().requires_grad_(True)
        # leaf tensor for RF: wrap raw leaf in RF Tensor so _apply_rope_real sees it
        x_leaf_rf = x_s.copy_template()
        x_leaf_rf.raw_tensor = x_s.raw_tensor.clone().requires_grad_(True)
        # Warmup: torch.compile compiles the backward lazily on first call;
        # without this the first timed iteration includes compilation time.
        _rope_compiled_raw(x_leaf_c, pe_raw).sum().backward()
        _apply_rope_real(x_leaf_rf, pe_s, head_dim).raw_tensor.sum().backward()
        t_c = _bench(
            lambda: _rope_compiled_raw(x_leaf_c, pe_raw).sum().backward(),
            f"compiled fwd+bwd T={T}",
            with_grad=True,
        )
        t_r = _bench(
            lambda: _apply_rope_real(x_leaf_rf, pe_s, head_dim).raw_tensor.sum().backward(),
            f"RF fwd+bwd T={T}",
            with_grad=True,
        )
        print(f"  {T:6d}  {t_c:9.1f}µ  {t_r:9.1f}µ  {t_r / t_c:7.1f}x")

    print("=== end profile ===\n")


def test_conformer_v2():
    downsampling = 6
    left_n, center_size, right_size = (16, 5, 4)

    build_dict = rf.build_dict(
        ChunkedConformerEncoder,
        encoder_layer=rf.build_dict(ChunkedConformerEncoderLayer),
        chunk_stride=center_size * downsampling,
        chunk_history=left_n,
        input_chunk_size_dim=(center_size + right_size) * downsampling,
        end_chunk_size_dim=center_size,
    )
    model = _build_model(build_dict)

    # f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2"
    # model.enc_build_dict
    build_dict_v2 = rf.build_dict(
        ChunkedConformerEncoderV2,
        encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
        chunk_size=center_size,
        chunk_history_size=left_n * center_size,
        chunk_lookahead_size=right_size,
        version=3,
        adapt_chunk_history_for_short_seqs=False,
    )
    model_v2 = _build_model(build_dict_v2)

    num_params = sum(p.num_elements() for p in model.parameters())
    num_params_v2 = sum(p.num_elements() for p in model_v2.parameters())
    print(f"num_params: {num_params} vs {num_params_v2}")
    assert num_params_v2 == num_params
    params_by_name = {name: p for name, p in model.named_parameters()}
    params_by_name_v2 = {name: p for name, p in model_v2.named_parameters()}
    assert set(params_by_name_v2.keys()) == set(params_by_name.keys())
    for name, p in params_by_name_v2.items():
        p0 = params_by_name[name]
        assert p.shape == p0.shape, f"{name} {p.shape} vs {p0.shape}"
        assert p.dtype == p0.dtype, f"{name} {p.dtype} vs {p0.dtype}"
        with torch.no_grad():
            p.raw_tensor.copy_(p0.raw_tensor)

    input_data, time_dim = _make_input_data()

    with (
        PyTracer(
            [
                ConformerEncoder.__call__,
                ChunkedConformerEncoderLayer.__call__,
                ChunkedConformerEncoderLayerV2.__call__,
            ],
            Tensor,
        ) as trace_v1,
        torch.no_grad(),
    ):
        res, out_spatial_dim = model(input_data, in_spatial_dim=time_dim)

    with (
        PyTracer(
            [
                ConformerEncoder.__call__,
                ChunkedConformerEncoderLayer.__call__,
                ChunkedConformerEncoderLayerV2.__call__,
            ],
            Tensor,
        ) as trace_v2,
        torch.no_grad(),
    ):
        res_v2, out_spatial_dim_v2 = model_v2(input_data, in_spatial_dim=time_dim)

    print(f"out: {res} vs {res_v2}")
    # Final check.
    # Actually the check_py_traces_rf_to_pt_equal should already have covered also this final output,
    # but anyway do it again now to be sure.
    assert res.dims == (batch_dim, out_spatial_dim, model.out_dim)
    assert res_v2.dims == (batch_dim, out_spatial_dim_v2, model_v2.out_dim)
    assert res.raw_tensor.shape == res_v2.raw_tensor.shape  # [B,T,D]
    assert res.raw_tensor.shape[:1] == out_spatial_dim.dyn_size_ext.raw_tensor.shape  # [B]
    for b in range(res.raw_tensor.shape[0]):
        seq_len = out_spatial_dim.dyn_size_ext.raw_tensor[b]
        torch.testing.assert_allclose(
            res.raw_tensor[b, :seq_len],
            res_v2.raw_tensor[b, :seq_len],
            rtol=1e-5,
            atol=1e-5,
        )
    # Check that there is sth non-zero.
    assert out_spatial_dim.dyn_size_ext.raw_tensor.max() > 0
    assert torch.mean(res.raw_tensor**2) > 0.1
    print("All matching!")


def test_conformer_v2_with_chunk_num_overlaps():
    """Smoke test: chunk_num_overlaps=2 (2x overlap) runs without errors."""
    chunk_size = 10
    chunk_num_overlaps = 2  # stride = chunk_size // 2 = 5
    chunk_history_size = 80
    chunk_lookahead_size = 4

    build_dict_v2 = rf.build_dict(
        ChunkedConformerEncoderV2,
        encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
        chunk_size=chunk_size,
        chunk_history_size=chunk_history_size,
        chunk_lookahead_size=chunk_lookahead_size,
        chunk_num_overlaps=chunk_num_overlaps,
        version=3,
        adapt_chunk_history_for_short_seqs=False,
    )
    model = _build_model(build_dict_v2)

    input_data, time_dim = _make_input_data()
    with torch.no_grad():
        res, out_spatial_dim = model(input_data, in_spatial_dim=time_dim)

    # With 2x overlap the output has ~2x more steps than without.
    assert out_spatial_dim.dyn_size_ext is not None
    assert out_spatial_dim.dyn_size_ext.raw_tensor.max() > 0
    assert torch.mean(res.raw_tensor**2) > 0.01
    print(f"test_conformer_v2_with_chunk_num_overlaps: out={res}, out_spatial_dim={out_spatial_dim}  OK")


def test_average_overlapping_chunks():
    """Unit test for _average_overlapping_chunks with known small tensors."""
    import torch

    # chunk_size=4, chunk_num_overlaps=2 → chunk_stride=2, n_chunks=3, feat=1
    chunk_size = 4
    chunk_stride = 2  # = chunk_size // chunk_num_overlaps
    n_chunks = 3
    feat = 1

    batch_dim_ = Dim(1, name="batch_test")
    chunked_time_dim = Dim(n_chunks, name="chunks")
    chunk_size_dim = Dim(chunk_size, name="chunk_size_enc")
    chunk_stride_enc_dim = Dim(chunk_stride, name="chunk_stride_enc")
    feat_dim_ = Dim(feat, name="feat_test")

    # x[b, chunk, frame, feat]:
    #   chunk 0: [a0, a1, a2, a3] = [10, 11, 12, 13]
    #   chunk 1: [b0, b1, b2, b3] = [20, 21, 22, 23]
    #   chunk 2: [c0, c1, c2, c3] = [30, 31, 32, 33]
    raw = torch.tensor(
        [[[[10.0], [11.0], [12.0], [13.0]], [[20.0], [21.0], [22.0], [23.0]], [[30.0], [31.0], [32.0], [33.0]]]]
    )  # [1, 3, 4, 1]
    x = rf.convert_to_tensor(raw, dims=[batch_dim_, chunked_time_dim, chunk_size_dim, feat_dim_])

    result = _average_overlapping_chunks(
        x,
        chunked_time_dim=chunked_time_dim,
        chunk_size_dim=chunk_size_dim,
        chunk_stride_enc_dim=chunk_stride_enc_dim,
    )
    # result shape: [batch, n_chunks, chunk_stride, feat]
    assert chunk_stride_enc_dim in result.dims, f"unexpected dims: {result.dims}"

    out = result.raw_tensor  # [1, 3, 2, 1]
    # With constant 1/n_shifts=0.5 scale (n_shifts=2):
    #   chunk 0: (group0=[10,11] + shift_right(group1=[12,13],by=1)=pad[0,0]) * 0.5 = [5, 5.5]
    #   chunk 1: (group0=[20,21] + shift_right(group1=[22,23],by=1)=[12,13])  * 0.5 = [16, 17]
    #   chunk 2: (group0=[30,31] + shift_right(group1=[32,33],by=1)=[22,23])  * 0.5 = [26, 27]
    expected = torch.tensor([[[[5.0], [5.5]], [[16.0], [17.0]], [[26.0], [27.0]]]])  # [1, 3, 2, 1]
    torch.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    print("test_average_overlapping_chunks: OK")


def _build_model(build_dict: Dict[str, Any]):
    base_build_dict = rf.build_dict(
        ConformerEncoder,
        input_layer=rf.build_dict(
            ConformerConvSubsample,
            out_dims=[32, 64, 64],
            filter_sizes=[(3, 3), (3, 3), (3, 3)],
            pool_sizes=[(1, 2)],
            strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
        ),
        # original:
        # num_layers=16,
        # out_dim=1024,
        num_layers=2,
        out_dim=64,
        encoder_layer=rf.build_dict(
            ConformerEncoderLayer,
            ff=rf.build_dict(
                ConformerPositionwiseFeedForward,
                activation=rf.build_dict(rf.relu_square),
                with_bias=False,
            ),
            num_heads=8,
        ),
    )
    build_dict = dict_update_deep(base_build_dict, build_dict)

    # rf.audio.log_mel_filterbank_from_raw(..., sampling_rate=16_000, out_dim=feat_dim) but we skip that here

    encoder = rf.build_from_dict(build_dict, feat_dim)
    encoder: ConformerEncoder  # might not be true, but assume similar/same interface
    return encoder


def _make_input_data(*, seq_len: int = 201) -> Tuple[Tensor, Dim]:
    """Create a random input tensor with dynamic sequence lengths around *seq_len*."""
    time_dim = Dim(
        rf.convert_to_tensor(
            [seq_len - i * 11 for i in range(batch_dim.get_dim_value())],
            dims=[batch_dim],
            name="time",
        )
    )
    return rf.random_normal([batch_dim, time_dim, feat_dim]), time_dim


if __name__ == "__main__":
    import sys

    if "--bench" in sys.argv:
        bench_rope_vs_relpos()
    else:
        tests()
