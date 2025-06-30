"""
A somewhat hacky way to modify an existing module to disable self-attention everywhere in it.
"""

from typing import Any, Dict
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


def apply_disable_self_attention_scheduled_(model: rf.Module, opts: Dict[str, Any]):
    """
    Disable self-attention

    :param model: will be modified in-place
    :param opts: should have "num_epochs", to specify the epoch until which the self-attention will be disabled
    """
    assert isinstance(opts, dict)

    for self_att in model.modules():
        if not isinstance(self_att, rf.SelfAttentionBase):
            continue

        class _HookedSelfAtt(rf.SelfAttention):
            def __call__(self, source: Tensor, **kwargs) -> Tensor:
                if rf.get_run_ctx().epoch <= opts["num_epochs"]:
                    output = _linear_split_v(source, self)
                    if self.proj:
                        output = self.proj(output)
                    return output

                # Default behavior, not disabled.
                return super().__call__(source, **kwargs)

        _HookedSelfAtt.__bases__ = (self_att.__class__,)
        self_att.__class__ = _HookedSelfAtt


def test_disable_self_att():
    rf.select_backend_torch()

    num_heads = 2
    in_dim = Dim(3, name="in")
    key_dim_total = Dim(5 * num_heads, name="key")
    value_dim_total = Dim(7 * num_heads, name="value")
    self_att = rf.SelfAttention(
        in_dim, num_heads=2, key_dim_total=key_dim_total, value_dim_total=value_dim_total, proj_dim=None
    )
    print(self_att)

    _test_split_v_self_att(self_att)

    # Now all the same Dim, because in Conformer/Transformer, we have:
    # self_att_opts_ = dict(
    #     in_dim=out_dim,
    #     proj_dim=out_dim,
    #     key_dim_total=out_dim,
    #     value_dim_total=out_dim,
    #     num_heads=num_heads,
    #     att_dropout=att_dropout,
    # )
    in_dim = Dim(5 * num_heads, name="in")
    self_att = rf.SelfAttention(in_dim, num_heads=2, key_dim_total=in_dim, value_dim_total=in_dim, proj_dim=in_dim)
    _test_split_v_self_att(self_att)


def _test_split_v_self_att(self_att: rf.SelfAttention):
    import torch

    batch_dim = Dim(3, name="batch")
    time_dim = Dim(5, name="time")
    source = rf.random_normal([batch_dim, time_dim, self_att.in_dim])
    print(source)
    output = self_att(source, axis=time_dim)
    print(output)

    q, k, v = self_att.forward_qkv(source)
    output, _ = rf.merge_dims(
        v, dims=(self_att.num_heads, self_att.value_dim_per_head), out_dim=self_att.value_dim_total
    )
    assert output.dims_set == {batch_dim, time_dim, self_att.value_dim_total}

    output2 = _linear_split_v(source, self_att)
    assert output2.dims_set == {batch_dim, time_dim, self_att.value_dim_total}

    torch.testing.assert_close(output2.copy_compatible_to_dims_raw(output.dims), output.raw_tensor)


def _linear_split_v(source: Tensor, self_att: rf.SelfAttention) -> Tensor:
    qkv_mat = self_att.qkv.weight
    qkv_bias = self_att.qkv.bias

    v_mat = _split_v(qkv_mat, self_att)
    v_bias = _split_v(qkv_bias, self_att) if qkv_bias is not None else None

    out = rf.matmul(source, v_mat, reduce=self_att.in_dim)
    out.feature_dim = self_att.value_dim_total
    if v_bias is not None:
        out += v_bias
    return out


def _split_v(qkv: Tensor, self_att: rf.SelfAttention) -> Tensor:
    if self_att.in_dim in qkv.dims and self_att.in_dim == self_att.value_dim_total:
        qkv, _ = rf.replace_dim(
            qkv,
            in_dim=self_att.in_dim,
            out_dim=rf.dim_match_priority_when_needed(self_att.in_dim, self_att.value_dim_total),
        )
    qkv = rf.split_dims(qkv, axis=self_att.qkv_dim_total, dims=(self_att.num_heads, self_att.qkv_dim_per_head))
    q, k, v = rf.split(
        qkv,
        axis=self_att.qkv_dim_per_head,
        out_dims=(self_att.key_dim_per_head, self_att.key_dim_per_head, self_att.value_dim_per_head),
    )
    v, _ = rf.merge_dims(v, dims=(self_att.num_heads, self_att.value_dim_per_head), out_dim=self_att.value_dim_total)
    return v
