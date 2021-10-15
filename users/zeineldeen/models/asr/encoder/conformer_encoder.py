import returnn_common.models.layers as layers
from returnn_common.models.base import Module, LayerRef, get_root_extern_data

from i6_experiments.users.zeineldeen.modules.positionwise_feedforward import PositionwiseFeedForward
from i6_experiments.users.zeineldeen.modules.conv import ConformerConvBlock, Conv2dBlock

from i6_experiments.users.zeineldeen.data_augment.specaugment_clean import SpecAugmentBlock


class ConformerBlock(Module):
    """
    Represents a conformer block
    """

    def __init__(self, conv_kernel_size=32, ff_act='swish', ff_dim=512, dropout=0.1, att_dropout=0.1,
                 enc_key_dim=256, att_n_heads=4, l2=0.0):
        super().__init__()

        self.dropout = dropout

        self.ffn1 = PositionwiseFeedForward(
            d_model=enc_key_dim, d_ff=ff_dim, dropout=dropout, activation=ff_act, l2=l2)

        self.ffn2 = PositionwiseFeedForward(
            d_model=enc_key_dim, d_ff=ff_dim, dropout=dropout, activation=ff_act, l2=l2)

        self.conv_module = ConformerConvBlock(d_model=enc_key_dim, kernel_size=conv_kernel_size)

    def forward(self, inp: LayerRef) -> LayerRef:
        # FFN
        x_ffn1_ln = layers.layer_norm(inp)
        x_ffn1 = self.ffn1(x_ffn1_ln)
        x_ffn1_out = 0.5 * layers.dropout(x_ffn1, dropout=self.dropout) + inp

        # TODO: MHSA
        x_mhsa_out = None

        # Conv
        x_conv_ln = layers.layer_norm(x_mhsa_out)
        x_conv = self.conv_module(x_conv_ln)
        x_conv_out = layers.dropout(x_conv, dropout=self.dropout) + x_mhsa_out

        # FFN
        x_ffn2_ln = layers.layer_norm(x_conv_out)
        x_ffn2 = self.ffn2(x_ffn2_ln)
        x_ffn2_out = 0.5 * layers.dropout(x_ffn2, dropout=self.dropout) + x_conv_out

        # last LN layer
        return layers.layer_norm(x_ffn2_out)


class ConformerEncoder(Module):
    """
    Represents Conformer encoder architecture
    """

    def __init__(self, num_blocks=12, audio_feature_key="audio_features", target_label_key="bpe_labels",
                 input_type='conv4', conv_kernel_size=32, ff_act='swish', ff_dim=512, dropout=0.1,
                 att_dropout=0.1, enc_key_dim=256, att_n_heads=4, l2=0.0, enable_specaugment=True):
        super().__init__()

        self.audio_feature_key = audio_feature_key
        self.target_label_key = target_label_key

        self.dropout = dropout

        self.specaug_block = None
        if enable_specaugment:
            self.specaug_block = SpecAugmentBlock()

        self.subsample_layer = None
        if input_type == 'conv4':  # subsample by 4
            self.subsample_layer = Conv2dBlock(
                filter_sizes='(3,3)_(3,3)',
                pool_sizes='(2,2)_(2,2)',
                channel_sizes='{}_{}'.format(enc_key_dim, enc_key_dim),
                l2=l2,
                dropout=dropout)
        else:
            raise ValueError('{} input type is not supported'.format(input_type))

        self.linear = layers.Linear(n_out=enc_key_dim, l2=l2, with_bias=False)

        self.conformer_blocks = [
            ConformerBlock(
                conv_kernel_size=conv_kernel_size, ff_act=ff_act, ff_dim=ff_dim, dropout=dropout,
                att_dropout=att_dropout, enc_key_dim=enc_key_dim, att_n_heads=att_n_heads, l2=l2
            )
            for _ in range(num_blocks)
        ]

    def forward(self) -> LayerRef:
        x = layers.copy(get_root_extern_data(self.audio_feature_key), name="encoder_in")
        if self.specaug_block:
            x = self.specaug_block(x)
        if self.subsample_layer:
            x = self.subsample_layer(x)
        x = self.linear(x)
        x = layers.dropout(x, dropout=self.dropout)
        for i, conformer_block in enumerate(self.conformer_blocks):
            x = conformer_block(x, name='conformer_block_%i' % i)
        encoder_state = layers.copy(x, name="encoder_state")
        return encoder_state
