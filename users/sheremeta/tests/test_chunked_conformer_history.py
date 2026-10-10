import torch

import returnn.frontend as rf
from returnn.frontend import _packed_backend as packed
from returnn.tensor import Dim, Tensor

from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v1 import ChunkedConformerConvBlock

from i6_experiments.users.sheremeta.nn_rf.encoder.chunked_conformer import conv_history_for_kernel

rf.select_backend_torch()


def _chunks(seed: int, feat: int = 8):
    batch_dim = Dim(3, name="batch")
    lens = Tensor("chunk_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([6, 4, 5], dtype=torch.int32))
    chunked_time_dim = Dim(lens, name="chunked_time")
    chunk_dim = Dim(6, name="chunk")
    end_chunk_dim = Dim(4, name="end_chunk")
    feat_dim = Dim(feat, name="feat")
    x = Tensor("x", dims=[batch_dim, chunked_time_dim, chunk_dim, feat_dim], dtype="float32")
    x.raw_tensor = torch.randn(3, 6, 6, feat, generator=torch.Generator().manual_seed(seed))
    return x, batch_dim, chunked_time_dim, chunk_dim, end_chunk_dim, feat_dim


def _valid_rows(out, dim, batch_dim, chunked_time_dim, feat_dim):
    out = packed.unpack(out)
    values = out.copy_transpose([batch_dim, chunked_time_dim, dim, feat_dim]).raw_tensor
    mask = rf.sequence_mask([batch_dim, chunked_time_dim]).copy_transpose([batch_dim, chunked_time_dim]).raw_tensor
    return values[mask]


def test_chunked_attention_matches_the_reference():
    """
    Our attention gives the values and the gradients of the reference class, padded and packed.
    On cuda it runs the Triton kernel, on cpu the composition of the rf op, which is the whole point
    of the op and cannot be read off the kernel test
    """
    from i6_experiments.users.sheremeta.nn_rf.encoder.chunked_conformer import kernel_attention_class
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v1 import ChunkedRelPosSelfAttention

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, batch_dim, chunked_time_dim, chunk_dim, end_chunk_dim, feat_dim = _chunks(3, feat=64)
    x.raw_tensor = x.raw_tensor.to(device)
    chunked_time_dim.dyn_size_ext.raw_tensor = chunked_time_dim.dyn_size_ext.raw_tensor.to(device)
    with rf.set_default_device_ctx(device):
        rf.set_random_seed(5)
        att = kernel_attention_class()(
            chunk_history=2,
            end_chunk_size_dim=end_chunk_dim,
            in_dim=feat_dim,
            proj_dim=feat_dim,
            key_dim_total=feat_dim,
            value_dim_total=feat_dim,
            num_heads=2,
            att_dropout=0.0,
        )
        params = [(name, p) for name, p in att.named_parameters()]
        rf.init_train_step_run_ctx(train_flag=False, step=0, epoch=1)
        for source in (x, packed.pack(x, dims=[batch_dim, chunked_time_dim])):
            ref = ChunkedRelPosSelfAttention.__call__(att, source, axis=chunk_dim, chunked_time_dim=chunked_time_dim)
            ref_rows = _valid_rows(ref, chunk_dim, batch_dim, chunked_time_dim, feat_dim)
            ref_rows.sum().backward()
            ref_grads = [p.raw_tensor.grad.clone() for _, p in params]
            for _, p in params:
                p.raw_tensor.grad = None
            out = att(source, axis=chunk_dim, chunked_time_dim=chunked_time_dim)
            out_rows = _valid_rows(out, chunk_dim, batch_dim, chunked_time_dim, feat_dim)
            torch.testing.assert_close(out_rows, ref_rows, rtol=1e-4, atol=1e-5)
            out_rows.sum().backward()
            for (name, p), ref_grad in zip(params, ref_grads):
                torch.testing.assert_close(p.raw_tensor.grad, ref_grad, rtol=1e-4, atol=1e-4, msg=name)
                p.raw_tensor.grad = None


def test_conv_block_history_follows_the_kernel():
    x, batch_dim, chunked_time_dim, chunk_dim, end_chunk_dim, feat_dim = _chunks(2)
    block = ChunkedConformerConvBlock(
        feat_dim, kernel_size=7, norm=rf.LayerNorm(feat_dim), chunk_history=4, end_chunk_size_dim=end_chunk_dim
    )
    assert conv_history_for_kernel(7, end_chunk_dim.dimension, 4) == 1
    outs = []
    for source in (x, packed.pack(x, dims=[batch_dim, chunked_time_dim])):
        for chunk_history in (4, conv_history_for_kernel(7, end_chunk_dim.dimension, 4)):
            block.chunk_history = chunk_history
            packed._warned_fallback_ops.clear()
            out = block(source, spatial_dim=chunk_dim, chunked_time_dim=chunked_time_dim)
            assert not packed._warned_fallback_ops, packed._warned_fallback_ops
            outs.append(_valid_rows(out, chunk_dim, batch_dim, chunked_time_dim, feat_dim))
    torch.testing.assert_close(outs[1], outs[0])
    torch.testing.assert_close(outs[3], outs[0])
