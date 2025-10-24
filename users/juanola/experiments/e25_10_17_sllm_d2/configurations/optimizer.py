from typing import Optional

from torch import nn


def conformer_aed_weight_decay_blacklist_v2(*, module: nn.Module, full_param_name: str, **kwargs) -> Optional[bool]:
    from torch import nn
    from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1

    """Turns off weight decay for embeddings."""
    if full_param_name.endswith("bias"):
        return False
    elif isinstance(module, nn.Embedding):
        return False
    elif isinstance(module, ConformerMHSARelPosV1) and full_param_name == "rel_pos_embeddings":
        return False
    elif isinstance(module, nn.LayerNorm):
        return True  # LayerNorm weight is (!) decayed
    return None  # no decision
