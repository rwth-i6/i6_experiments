"""LM test. See :func:`test_lm`"""

from typing import Union, Any
import sys
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder


def test_lm(lm: Union[TransformerDecoder, Any]):
    """
    Test that the LM behaves as expected, i.e. produces consistent outputs for handling inputs in various ways.

    Also see RETURNN test_causal_self_att_after_concat_state for a similar test.

    :param lm:
        The language model decoder to be tested. Can be of type TransformerDecoder,
        or any other type which implements the same interface.
    """
    import numpy as np

    batch_dim = Dim(5, name="batch")
    beam1_dim = Dim(3, name="beam1")
    beam2_dim = Dim(2, name="beam2")
    backrefs = rf.convert_to_tensor([2, 1], dims=[beam2_dim], dtype="int32", sparse_dim=beam1_dim)

    time1_dim = Dim(rf.random_uniform([batch_dim, beam1_dim], dtype="int32", minval=3, maxval=14), name="time1")
    time2_dim = Dim(rf.random_uniform([batch_dim, beam2_dim], dtype="int32", minval=2, maxval=12), name="time2")

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

    # First on the whole seq
    state = lm.default_initial_state(batch_dims=[batch_dim, beam2_dim])
    out_whole_seq, _ = lm(data, spatial_dim=time_dim, state=state)
    assert isinstance(out_whole_seq, Tensor)

    # Now step-by-step
    state = lm.default_initial_state(batch_dims=[batch_dim, beam1_dim])
    res1 = []
    for t in range(time1_dim.get_dim_value()):
        x_t = rf.gather(data1, axis=time1_dim, indices=t)
        res_t, state = lm(x_t, spatial_dim=single_step_dim, state=state)
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

    np.testing.assert_allclose(
        out_whole_seq.raw_tensor.cpu().detach().numpy(),
        out_step_by_step.raw_tensor.cpu().detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        out_whole_seq.raw_tensor.cpu().detach().numpy(),
        out_two_halves.raw_tensor.cpu().detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rf_transformer_llama():
    rf.select_backend_torch()

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


if __name__ == "__main__":
    # Fixup sys.path for local testing
    sys.path = [path for path in sys.path if not path.endswith("i6_experiments/users/zeyer/returnn")]
    test_rf_transformer_llama()
