from typing import Optional, Tuple
from torch import Tensor

import torch
import math


def _scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    # dimensions
    bsz, num_heads, head_dim = query.size(0), query.size(1), query.size(-1)
    src_len, tgt_len = key.size(-2), query.size(-2)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # reshape back
    query = query.view(-1, tgt_len, head_dim)
    key = key.view(-1, src_len, head_dim)
    value = value.view(-1, src_len, head_dim)
    if attn_mask is not None:
        attn_mask = attn_mask.view(bsz*num_heads, -1, src_len)
        attn_mask_bool = (attn_mask == float("-inf")).bool()

    # attn calculation
    q_scaled = query * scale_factor
    attn_weight = torch.bmm(q_scaled, key.transpose(-2, -1))
    if attn_mask is not None:
        # NOTE the error occurs when torch uses baddbmm (i.e. simply adds the mask instead of using masked_fill)
        attn_weight = attn_weight.masked_fill(attn_mask_bool, float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if attn_mask is not None:
        attn_weight = attn_weight.masked_fill(attn_mask_bool, 0.0)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    output = torch.bmm(attn_weight, value)

    # reshape to expected shapes
    output = output.reshape(bsz, num_heads, output.size(-2), output.size(-1))

    return output