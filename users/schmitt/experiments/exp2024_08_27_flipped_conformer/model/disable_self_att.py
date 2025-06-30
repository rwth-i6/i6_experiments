from typing import Any, Dict
import returnn.frontend as rf
from returnn.tensor import Tensor


def apply_disable_self_attention_(model: rf.Module, opts: Dict[str, Any]):
    """disable self-attention"""
    assert isinstance(opts, dict)

    for self_att in model.modules():
        if not isinstance(self_att, rf.SelfAttentionBase):
            continue

        class _HookedSelfAtt(rf.SelfAttention):
            def __call__(self, source: Tensor, **kwargs) -> Tensor:
                if rf.get_run_ctx().epoch <= opts["num_epochs"]:
                    # Very simple way to disable self-attention. Only the value transformation.
                    # We could make this more efficient here by only do the matmul for the value,
                    # but this here is simpler now, and also more safe that we do it correctly...
                    q, k, v = self.forward_qkv(source)
                    output, _ = rf.merge_dims(
                        v, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total
                    )
                    if self.proj:
                        output = self.proj(output)
                    return output

                # Default behavior, not disabled.
                return super().__call__(source, **kwargs)

        _HookedSelfAtt.__bases__ = (self_att.__class__,)
        self_att.__class__ = _HookedSelfAtt
