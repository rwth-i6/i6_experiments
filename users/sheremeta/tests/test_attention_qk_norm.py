import torch

import returnn.frontend as rf
from returnn.tensor import Dim, Tensor
from returnn.util import BehaviorVersion

from i6_experiments.users.sheremeta.nn_rf.encoder.attention_qk_norm import QKNormRelPosSelfAttention

rf.select_backend_torch()
if BehaviorVersion.get_if_set() is None:
    BehaviorVersion.set_min_behavior_version(31)


def test_attention_does_not_grow_with_queries_keys_or_positions():
    """
    Scaling the query and key projection leaves the attention output with the norm unchanged and changes it without,
    scaling the position projection leaves it unchanged only with the position norm.
    """
    model_dim = Dim(16, name="model")
    batch_dim = Dim(2, name="batch")
    lens = Tensor("lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([7, 5], dtype=torch.int32))
    time_dim = Dim(lens, name="time")
    x = Tensor("x", dims=[batch_dim, time_dim, model_dim], dtype="float32")
    x.raw_tensor = torch.randn(2, 7, 16, generator=torch.Generator().manual_seed(0))
    mask = rf.sequence_mask([batch_dim, time_dim]).copy_transpose([batch_dim, time_dim]).raw_tensor
    rf.init_train_step_run_ctx(train_flag=False, step=0, epoch=1)

    def output(att) -> torch.Tensor:
        return att(x, axis=time_dim).copy_transpose([batch_dim, time_dim, model_dim]).raw_tensor[mask]

    def scale_queries_and_keys(att):
        # the projection is laid out heads major, each head's queries, keys and values in a row
        att.qkv.weight.raw_tensor.view(16, 2, 3, 8)[:, :, :2] *= 10.0
        if att.qkv.bias is not None:
            att.qkv.bias.raw_tensor.view(2, 3, 8)[:, :2] *= 10.0

    def scale_positions(att):
        linear = att.linear_pos.linear if hasattr(att.linear_pos, "linear") else att.linear_pos
        linear.weight.raw_tensor.mul_(10.0)

    opts = dict(key_dim_total=model_dim, value_dim_total=model_dim, num_heads=2, att_dropout=0.0)
    cases = [("plain", lambda: rf.RelPosSelfAttention(model_dim, model_dim, **opts), (False, False))]
    for norm in ("layer_norm", "rms"):
        cases.append((norm, lambda norm=norm: QKNormRelPosSelfAttention(model_dim, model_dim, **opts, qk_norm=norm),
                      (True, False)))
        cases.append((f"{norm} positions",
                      lambda norm=norm: QKNormRelPosSelfAttention(model_dim, model_dim, **opts, qk_norm=norm,
                                                                  pos_norm=True), (True, True)))
    for name, build, invariant in cases:
        for scale, stays in zip((scale_queries_and_keys, scale_positions), invariant):
            att = build()
            before = output(att)
            with torch.no_grad():
                scale(att)
            after = output(att)
            if stays:
                torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-4, msg=f"{name} {scale.__name__}")
            else:
                assert (after - before).abs().max() > 1e-2, f"{name} {scale.__name__}"
