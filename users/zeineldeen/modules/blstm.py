import returnn_common.models.layers as layers
from returnn_common.models.base import Module, LayerRef


class BLSTMPoolBlock(Module):
    """
    Bi-directional LSTM layer with optional max-pooling
    """

    def __init__(self, l2=0.0, lstm_n_out=512, dropout=0.3, pool_size=1, rec_unit='nativelstm2'):
        super().__init__()
        self.lstm_fw = layers.Rec(direction=1, n_out=lstm_n_out, unit=rec_unit, l2=l2, dropout=dropout)
        self.lstm_bw = layers.Rec(direction=-1, n_out=lstm_n_out, unit=rec_unit, l2=l2, dropout=dropout)

        self.pool_size = pool_size

    def forward(self, inp: LayerRef) -> LayerRef:
        x_fw = self.lstm_fw(inp)
        x_bw = self.lstm_bw(inp)
        if self.pool_size > 1:
            x_out = layers.pool([x_fw, x_bw], pool_size=(self.pool_size,), padding='same', mode='max')
        else:
            x_out = layers.copy([x_fw, x_bw])
        return x_out