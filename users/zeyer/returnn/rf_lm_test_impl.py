"""LM test. See :func:`test_lm`"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Any
import sys
from returnn.util.basic import BehaviorVersion
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

if TYPE_CHECKING:
    import torch
    import numpy


def test_lm(lm: Union[TransformerDecoder, Any], *, rtol: float = 1e-5, atol: float = 1e-5):
    """
    Test that the LM behaves as expected, i.e. produces consistent outputs for handling inputs in various ways.

    Also see RETURNN test_causal_self_att_after_concat_state for a similar test.

    :param lm:
        The language model decoder to be tested. Can be of type TransformerDecoder,
        or any other type which implements the same interface.
    """

    batch_dim = Dim(5, name="batch")
    beam1_dim = Dim(3, name="beam1")
    beam2_dim = Dim(2, name="beam2")
    backrefs = rf.convert_to_tensor([2, 1], dims=[beam2_dim], dtype="int32", sparse_dim=beam1_dim, device="cpu")

    time1_dim = Dim(
        rf.random_uniform([batch_dim, beam1_dim], dtype="int32", minval=3, maxval=14, device="cpu"), name="time1"
    )
    time2_dim = Dim(
        rf.random_uniform([batch_dim, beam2_dim], dtype="int32", minval=2, maxval=12, device="cpu"), name="time2"
    )
    _print_dim("** time1_dim:", time1_dim)
    _print_dim("** time2_dim:", time2_dim)

    data1 = rf.random_uniform(
        [batch_dim, beam1_dim, time1_dim],
        dtype="int32",
        minval=0,
        maxval=lm.vocab_dim.dimension,
        sparse_dim=lm.vocab_dim,
    )
    data2 = rf.random_uniform(
        [batch_dim, beam2_dim, time2_dim],
        dtype="int32",
        minval=0,
        maxval=lm.vocab_dim.dimension,
        sparse_dim=lm.vocab_dim,
    )
    data1_beam2, time1_dim_beam2 = rf.nested.gather_nested((data1, time1_dim), indices=backrefs)
    data, time_dim = rf.concat((data1_beam2, time1_dim_beam2), (data2, time2_dim))
    _print_dim("** time1_dim_beam2:", time1_dim_beam2)
    _print_dim("** time_dim:", time_dim)

    # First on the whole seq
    state = lm.default_initial_state(batch_dims=[batch_dim, beam2_dim])
    out_whole_seq, _ = lm(data, spatial_dim=time_dim, state=state)
    assert isinstance(out_whole_seq, Tensor)

    # Now step-by-step
    state = lm.default_initial_state(batch_dims=[batch_dim, beam1_dim])
    res1 = []
    for t in range(time1_dim.get_dim_value()):
        x_t = rf.gather(data1, axis=time1_dim, indices=t)
        res_t, state_ = lm(x_t, spatial_dim=single_step_dim, state=state)
        if t > 0:
            state = rf.nested.where_nested(
                state_,
                state,
                condition=t < time1_dim.get_size_tensor(device=x_t.device),
                condition_cpu=t < time1_dim.get_size_tensor(),
            )
        else:
            state = state_
        res1.append(res_t)
    res1_stack, _ = rf.stack(res1, out_dim=time1_dim)
    res1_stack = rf.nested.gather_nested(res1_stack, indices=backrefs, dim_map={time1_dim: time1_dim_beam2})
    state = rf.nested.gather_nested(state, indices=backrefs)
    res2 = []
    for t in range(time2_dim.get_dim_value()):
        x_t = rf.gather(data2, axis=time2_dim, indices=t)
        res_t, state = lm(x_t, spatial_dim=single_step_dim, state=state)
        res2.append(res_t)
    res2_stack, _ = rf.stack(res2, out_dim=time2_dim)
    out_step_by_step, _ = rf.concat((res1_stack, time1_dim_beam2), (res2_stack, time2_dim), out_dim=time_dim)

    # Now concat the two halves
    state = lm.default_initial_state(batch_dims=[batch_dim, beam1_dim])
    res1, state = lm(data1, spatial_dim=time1_dim, state=state)
    res1 = rf.nested.gather_nested(res1, indices=backrefs, dim_map={time1_dim: time1_dim_beam2})
    state = rf.nested.gather_nested(state, indices=backrefs)
    res2, _ = lm(data2, spatial_dim=time2_dim, state=state)
    out_two_halves, _ = rf.concat((res1, time1_dim_beam2), (res2, time2_dim), out_dim=time_dim)

    # Make consistent, and compare
    out_whole_seq = out_whole_seq.copy_transpose([batch_dim, beam2_dim, time_dim, lm.vocab_dim]).copy_masked(0)
    out_step_by_step = out_step_by_step.copy_transpose([batch_dim, beam2_dim, time_dim, lm.vocab_dim]).copy_masked(0)
    out_two_halves = out_two_halves.copy_transpose([batch_dim, beam2_dim, time_dim, lm.vocab_dim]).copy_masked(0)

    print("** check out whole seq vs step by step")
    assert_all_close(out_whole_seq, out_step_by_step, ndindex_shape_slice_end=-1, rtol=rtol, atol=atol)
    print("** check out whole seq vs two halves")
    assert_all_close(out_whole_seq, out_two_halves, ndindex_shape_slice_end=-1, rtol=rtol, atol=atol)
    print("** all checks passed")


def test_rf_transformer_llama():
    _init()

    lm = TransformerDecoder(
        encoder_dim=None,
        vocab_dim=Dim(32, name="vocab"),
        num_layers=4,
        model_dim=32,
        num_heads=2,
        pos_enc=None,
        norm=rf.build_dict(rf.RMSNorm),
        ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
        decoder_layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
        dropout=0.0,
        att_dropout=0.0,
    )

    test_lm(lm)


def test_qwen2_finetuned():
    from sisyphus import Path
    from i6_experiments.users.zeyer.external_models.qwen2_finetuned import get_lm, Qwen2Model
    import tree
    import functools
    from returnn.util.hot_reload import ConfigHotReloader

    _init()

    lm = get_lm()
    get_model = lm.definition.model_def
    assert isinstance(get_model, functools.partial)
    assert get_model.func is Qwen2Model and get_model.keywords
    print("** get_model.keywords:", get_model.keywords)

    flat_opts = tree.flatten(get_model.keywords)
    paths = [obj for obj in flat_opts if isinstance(obj, Path)]
    assert paths, f"Expected to find at least one Path in the model definition options {get_model.keywords}"
    print(f"** Found {len(paths)} file paths")

    if any(not path.available() for path in paths):
        print("** Warning: Some model files are not available. Skipping the test.")
        for path in paths:
            if not path.available():
                print(f"   - {path} is not available.")
        return

    def _convert_path(obj):
        if isinstance(obj, Path):
            return obj.get_path()
        return obj

    opts = tree.map_structure(_convert_path, get_model.keywords)
    model = Qwen2Model(**opts)
    print("Model:", model.model)
    print("Vocab:", model.vocab_dim)
    model.model.to(rf.get_default_device())

    if sys.stdin.isatty():
        hot_reloader = ConfigHotReloader({"Qwen2Model": Qwen2Model})
        while True:
            try:
                test_lm(model, atol=2e-5, rtol=0.02)
                break
            except Exception as exc:
                print("Exception with hot reloading enabled:")
                sys.excepthook(type(exc), exc, exc.__traceback__)
                hot_reloader.wait_for_user()
                hot_reloader.reload_changed_modules()
                model.__class__ = hot_reloader.config["Qwen2Model"]

    else:  # not interactive
        test_lm(model, atol=2e-5, rtol=0.02)


def _print_dim(prefix: str, dim: Dim):
    print(prefix, dim, int(dim.get_dim_value()), dim.get_size_tensor().raw_tensor.numpy())


def assert_all_close(
    x: Union[Tensor, torch.Tensor, numpy.ndarray],
    y: Union[Tensor, torch.Tensor, numpy.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    equal_nan: bool = True,
    ndindex_shape_slice_end: Optional[int] = None,
    remarks_limit: int = 100,
):
    import numpy as np
    import torch

    shape_info = None
    if isinstance(x, Tensor):
        assert isinstance(y, Tensor)
        shape_info = "(%s)" % ", ".join(
            d.short_repr() + (f"={int(d.get_dim_value())}" if d.dimension is None else "") for d in x.dims
        )
        x = x.copy_masked(0)
        y = y.copy_masked(0)
        y = y.copy_transpose(x.dims)
        x = x.raw_tensor
        y = y.raw_tensor

    if isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    if shape_info is None:
        shape_info = x.shape

    assert x.shape == y.shape, "Shapes do not match: %s vs %s" % (shape_info or x.shape, y.shape)

    if np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan):
        return
    print(f"** not all close. shape: {shape_info}. close:")
    # Iterate over all indices, and check if the values are close.
    # If not, add the index to the mismatches list.
    remarks = []
    count_mismatches = 0
    for idx in sorted(np.ndindex(x.shape[:ndindex_shape_slice_end]), key=sum):
        close = np.allclose(x[idx], y[idx], rtol=rtol, atol=atol, equal_nan=equal_nan)
        count_mismatches += 1 if close else 0
        idx_str = "[%s]" % ",".join([str(i) for i in idx])
        if np.isnan(x[idx]).any() or np.isnan(x[idx]).any():
            remarks.append(f"{idx_str}:? (have nan)")
            continue
        remarks.append(f"{idx_str}:" + ("✓" if close else "✗ (%.5f diff)" % np.abs(x[idx] - y[idx]).sum()))
        if len(remarks) >= remarks_limit and count_mismatches > 0:
            remarks.append("...")
            break
    print("\n".join(remarks))
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


_is_initialized = False


def _init():
    global _is_initialized
    if _is_initialized:
        return

    rf.select_backend_torch()
    rf.set_random_seed(42)
    BehaviorVersion.set_min_behavior_version(24)
    _is_initialized = True

    try:
        # noinspection PyUnusedImports
        import better_exchook

        better_exchook.install()

    except ImportError:
        pass

    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ImportError:
        pass


def tests():
    import torch

    _init()

    print("* Test Transformer++")
    test_rf_transformer_llama()

    if torch.cuda.is_available():
        print("* Test Transformer++ on GPU")
        with rf.set_default_device_ctx("cuda"):
            test_rf_transformer_llama()

    with rf.set_default_device_ctx("cuda" if torch.cuda.is_available() else "cpu"):
        print("* Test Qwen2 finetuned")
        test_qwen2_finetuned()


if __name__ == "__main__":
    # Fixup sys.path for local testing
    sys.path = [path for path in sys.path if not path.endswith("i6_experiments/users/zeyer/returnn")]
    tests()
