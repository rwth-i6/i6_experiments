from returnn_common.models import layers
from returnn_common.models.base import Module, LayerRef


class MultiHeadSelfAttention(Module):
    """
    Multihead self attention block
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()


    def forward(self, inp: LayerRef) -> LayerRef:
        pass
