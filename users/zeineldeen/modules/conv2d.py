from returnn_common.models import layers


class Conv2dBlock(Module):
    """
    Conv 2D block with optional max pooling
    """

    def __init__(self, l2=1e-07, dropout=0.3, act='relu', filter_sizes=[(3, 3)],
                 pool_sizes=[(1, 2)], channel_sizes=[32], padding='same'):
        super().__init__()
        self.split_feature_layer = layers.SplitDims(axis="F", dims=(-1, 1))
        self.conv_layers = []
        self.pool_layers = []
        self.dropout_layers = []
        for filter_size, pool_size, channel_size in zip(filter_sizes, pool_sizes, channel_sizes):
            self.conv_layers.append(layers.Conv(
                l2=l2, activation=act, filter_size=filter_size, n_out=channel_size, padding=padding))
            self.pool_layers.append(layers.Pool(pool_size=pool_size, padding='same', mode="max"))
        self.dropout = layers.Dropout(dropout=dropout)
        self.merge_features_layer = layers.MergeDims(axes="static")

    def forward(self, inp: LayerRef) -> LayerRef:
        x = self.split_feature_layer(inp)  # [B,T,F,1]
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x = conv_layer(x)
            x = pool_layer(x)
            x = self.dropout(x)
        out = self.merge_features_layer(x)  # [B,T,F]
        return out