import returnn_common.models.layers as layers
from returnn_common.models.base import LayerRef, get_root_extern_data, Module

from recipe.i6_experiments.users.zeineldeen.modules.blstm import BLSTMPoolBlock
from recipe.i6_experiments.users.zeineldeen.modules.conv import Conv2dBlock
from recipe.i6_experiments.users.zeineldeen.data_augment.specaugment_clean import SpecAugmentBlock


class ConvBLSTMEncoder(Module):
    """
    Conv + BLSTM encoder architecture
    """

    def __init__(self, l2=0.0, audio_feature_key="audio_features", target_label_key="bpe_labels",
                 conv_dropout=0.3, conv_filter_sizes='(3,3)_(3,3)', conv_pool_sizes='(1,2)_(1,2)',
                 conv_channel_sizes='32_32', num_lstm_layers=6, lstm_dim=1024, lstm_dropout=0.3,
                 lstm_pool_sizes='3_2', enable_specaugment=True):
        super().__init__()

        self.audio_feauture_key = audio_feature_key
        self.target_label_key = target_label_key
        assert num_lstm_layers >= 2, "Needs two lstm layers as the last layer lstm layer is special"

        if enable_specaugment:
            self.specaug_block = SpecAugmentBlock()
        else:
            self.specaug_block = None

        self.conv_block = Conv2dBlock(l2=l2, dropout=conv_dropout, filter_sizes=conv_filter_sizes,
                                      pool_sizes=conv_pool_sizes, channel_sizes=conv_channel_sizes)

        self.lstm_layers = []
        lstm_pool_sizes = list(map(int, lstm_pool_sizes.split('_')))
        for i in range(num_lstm_layers - 1):
            pool_size = lstm_pool_sizes[i] if i < len(lstm_pool_sizes) else 1
            self.lstm_layers.append(
                BLSTMPoolBlock(l2=l2, lstm_n_out=lstm_dim, dropout=lstm_dropout, pool_size=pool_size)
            )

        self.last_lstm_layer = BLSTMPoolBlock(
            l2=l2, lstm_n_out=lstm_dim, dropout=lstm_dropout)

    def forward(self) -> LayerRef:
        x = layers.copy(get_root_extern_data(self.audio_feauture_key), name="encoder_in")
        if self.specaug_block:
            x = self.specaug_block(x)
        x = self.conv_block(x)
        for i, lstm_layer in enumerate(self.lstm_layers):
            x = lstm_layer(x, name="blstm_block_%i" % i)
        lstm_last = self.last_lstm_layer(x, name="final_lstm")
        encoder_state = layers.copy(lstm_last, name="encoder_state")
        return encoder_state
