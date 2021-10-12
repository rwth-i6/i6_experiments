from returnn_common.models import layers


class BLSTMPoolBlock(Module):
    """

    """

    def __init__(self, l2=1e-07, lstm_n_out=256, dropout=0.3, pool_size=1, rec_unit='nativelstm2'):
        super().__init__()
        self.lstm_fw = layers.Rec(direction=1, n_out=lstm_n_out, unit=rec_unit, l2=l2, dropout=dropout)
        self.lstm_bw = layers.Rec(direction=-1, n_out=lstm_n_out, unit=rec_unit, l2=l2, dropout=dropout)

        if pool_size > 1:
            self.pool = layers.Pool(pool_size=(pool_size,), padding="same", mode="max")
        else:
            self.pool = layers.Copy()

    def forward(self, inp: LayerRef) -> LayerRef:
        x_fw = self.lstm_fw(inp)
        x_bw = self.lstm_bw(inp)
        x_out = self.pool([x_fw, x_bw])
        return x_out