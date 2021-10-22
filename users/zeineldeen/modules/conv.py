from returnn_common.models import layers
from returnn_common.models.base import Module, LayerRef

from recipe.i6_experiments.users.zeineldeen.utils import parse_cnn_args


class Conv2dBlock(Module):
    """
    Conv 2D block with optional max-pooling
    """

    def __init__(self, l2=0.0, dropout=0.3, act='relu', filter_sizes='(3,3)', pool_sizes='(1,2)', channel_sizes='32',
                 padding='same'):
        super().__init__()

        self.dropout = dropout

        filter_sizes = parse_cnn_args(filter_sizes)
        channel_sizes = list(map(int, channel_sizes.split('_')))
        self.pool_sizes = parse_cnn_args(pool_sizes)

        self.conv_layers = []
        for filter_size, channel_size in zip(filter_sizes, channel_sizes):
            self.conv_layers.append(layers.Conv(
                l2=l2, activation=act, filter_size=filter_size, n_out=channel_size, padding=padding))

    def forward(self, inp: LayerRef) -> LayerRef:
        x = layers.split_dims(inp, axis='F', dims=(-1, 1))  # [B,T,F,1]
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = layers.pool(x, pool_size=self.pool_sizes[i], padding='same', mode='max')
            if self.dropout:
                x = layers.dropout(x, dropout=self.dropout)
        out = layers.merge_dims(x, axes='static')  # [B,T,F]
        return out


class ConformerConvBlock(Module):
    """
    Conformer convolution block
    FF -> GLU -> depthwise conv -> BN -> Swish -> FF
    """

    def __init__(self, d_model, kernel_size, l2=0.0):
        super().__init__()

        self.positionwise_conv1 = layers.Linear(n_out=d_model * 2, l2=l2)
        self.depthwise_conv = layers.Conv(n_out=d_model, filter_size=kernel_size, groups=d_model, l2=l2, padding='same')
        self.positionwise_conv2 = layers.Linear(n_out=d_model, l2=l2)

    def forward(self, inp: LayerRef) -> LayerRef:
        x_conv1 = self.positionwise_conv1(inp)
        x_act = layers.activation(x_conv1, activation='glu')
        x_depthwise_conv = self.depthwise_conv(x_act)
        x_bn = layers.batch_norm(x_depthwise_conv)
        x_swish = layers.activation(x_bn, activation='swish')
        x_conv2 = self.positionwise_conv2(x_swish)
        return x_conv2
