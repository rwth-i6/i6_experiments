import returnn_common.models.layers as layers
from returnn_common.models.base import Module, LayerRef


class PositionwiseFeedForward(Module):
    """
    Conformer position-wise feedforward neural network layer
        FF -> Activation -> Dropout -> FF
    """

    def __init__(self, d_model, d_ff, dropout, activation, l2=0.0):
        super().__init__()

        self.dropout = dropout
        self.activation = activation

        self.linear1 = layers.Linear(n_out=d_ff, l2=l2)
        self.linear2 = layers.Linear(n_out=d_model, l2=l2)

    def forward(self, inp: LayerRef) -> LayerRef:
        return self.linear2(layers.dropout(layers.activation(
            self.linear1(inp), activation=self.activation), dropout=self.dropout))
