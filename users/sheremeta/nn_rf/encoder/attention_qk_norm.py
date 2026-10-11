from typing import Tuple

import returnn.frontend as rf
from returnn.tensor import Dim as ReturnnDim
from returnn.tensor import Tensor as ReturnnTensor

QK_NORMS = {"layer_norm": rf.LayerNorm, "rms": rf.RMSNorm}


class AttentionQKNormMixin:
    """
    Normalizes the queries and keys of a self-attention which gets them from ``forward_qkv`` over the features of each
    head, with learned scales the heads share, so the attention logits cannot grow with the size of the queries and
    keys. With ``pos_norm`` it normalizes the projected relative position encodings of a rel-pos self-attention per
    head as well, which the queries meet in the position term of the logits.
    Mixed in before the attention class, which the calls pass their further arguments to.
    """

    def __init__(self, *args, qk_norm: str, pos_norm: bool = False, **kwargs):
        """
        :param qk_norm: layer_norm or rms
        :param pos_norm: whether to normalize the projected relative position encodings per head as well
        """
        super().__init__(*args, **kwargs)
        assert qk_norm in QK_NORMS, f"unknown qk_norm {qk_norm!r}"
        self.qk_norm = qk_norm
        self.q_norm = QK_NORMS[qk_norm](self.key_dim_per_head)
        self.k_norm = QK_NORMS[qk_norm](self.key_dim_per_head)
        if pos_norm:
            assert getattr(self, "linear_pos", None) is not None and self.separate_pos_emb_per_head, (
                f"{self}: the position norm needs the projection of the positions to every head"
            )
            self.linear_pos = PerHeadNormedLinear(
                self.linear_pos, num_heads=self.num_heads, per_head=self.key_dim_per_head, norm=QK_NORMS[qk_norm]
            )

    def forward_qkv(self, source: ReturnnTensor) -> Tuple[ReturnnTensor, ReturnnTensor, ReturnnTensor]:
        """
        :return: the normalized queries and keys and the values
        """
        q, k, v = super().forward_qkv(source)
        return self.q_norm(q), self.k_norm(k), v


class PerHeadNormedLinear(rf.Module):
    """
    A linear projection to the features of all heads, its output normalized over the features of each head.
    """

    def __init__(self, linear: rf.Linear, *, num_heads: ReturnnDim, per_head: ReturnnDim, norm: type):
        """
        :param linear: the projection, its output dim the heads times the features per head
        :param num_heads: the heads dim
        :param per_head: the features of one head
        :param norm: the norm class, built over the features of one head
        """
        super().__init__()
        self.linear = linear
        self.out_dim = linear.out_dim
        self.num_heads = num_heads
        self.per_head = per_head
        self.norm = norm(per_head)

    def __call__(self, source: ReturnnTensor) -> ReturnnTensor:
        """forward"""
        out = rf.split_dims(self.linear(source), axis=self.out_dim, dims=(self.num_heads, self.per_head))
        out, _ = rf.merge_dims(self.norm(out), dims=(self.num_heads, self.per_head), out_dim=self.out_dim)
        return out


class QKNormRelPosSelfAttention(AttentionQKNormMixin, rf.RelPosSelfAttention):
    """
    Relative-position self-attention with normalized queries and keys, see :class:`AttentionQKNormMixin`.
    """
