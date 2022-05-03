import copy
import numpy
from typing import Union, Optional, Tuple, List, Dict, Any

from returnn_common import nn
from returnn_common.nn import any_feature_dim, SpatialDim, FeatureDim, Dim
from returnn_common.nn import Module, Tensor
from returnn_common.nn import split_dims, Conv2d, pool1d, pool2d, relu, merge_dims, LSTM, concat, dropout
from returnn_common.nn import ReturnnDimTagsProxy
from returnn_common.nn import ModuleList, Sequential


from returnn.util.basic import NotSpecified

from .specaugment import specaugment, SpecAugmentSettings

class Encoder2DConvBlock(Module):

    def __init__(self, in_time_dim, in_dim: Dim, activation=relu, filter_sizes=[(3, 3)],
                 pool_sizes=[(1, 2)], channel_sizes=[32], padding='same'):
        super().__init__()
        self.conv_layers = ModuleList()
        self.pool_sizes = pool_sizes
        self.activation = activation
        self.in_time_dim = in_time_dim

        assert len(filter_sizes) == len(pool_sizes)
        assert len(filter_sizes) == len(channel_sizes)

        self.in_feature_dim = in_dim

        for filter_size, pool_size, channel_size in zip(filter_sizes, pool_sizes, channel_sizes):
            conv_out_dim = FeatureDim("conv_channel", channel_size)
            #self.conv2d = Conv2d(out_dim=conv_out_dim, filter_size=filter_size, in_dim=temp_in_dim, padding=padding)
            conv2d = Conv2d(out_dim=conv_out_dim, filter_size=filter_size, padding=padding)
            self.conv_layers.append(conv2d)

        self.last_conv_out_dim = conv_out_dim

        total_pool_factor = int(numpy.prod([p[1] for p in pool_sizes]))
        self.merge_out_dim = FeatureDim("merge_out", dimension=int(numpy.ceil(in_dim.dimension/total_pool_factor))*channel_sizes[-1])

    @nn.scoped
    def __call__(self, inp: Tensor) -> Tuple[Tensor, Dim, Dim]:
        conv_spatial_dim = SpatialDim("conv_vertical_dim", dimension=self.in_feature_dim.dimension)
        conv_channel_dim = FeatureDim("conv_channel_dim", dimension=1)
        x = split_dims(inp, axis=self.in_feature_dim, dims=[conv_spatial_dim, conv_channel_dim])
        out_spatial_dims = [self.in_time_dim, conv_spatial_dim]
        for conv_layer, pool_size in zip(self.conv_layers, self.pool_sizes):
            x, out_spatial_dims = conv_layer(x, in_spatial_dims=out_spatial_dims)
            x, out_spatial_dims = pool2d(x, mode="max", pool_size=pool_size, padding="same", in_spatial_dims=out_spatial_dims)
        out_time_dim = out_spatial_dims[0]
        print([out_spatial_dims[-1], self.last_conv_out_dim])
        out, out_feature_dim = merge_dims(x, axes=[out_spatial_dims[-1], self.last_conv_out_dim], out_dim=self.merge_out_dim)
        return out, out_time_dim, out_feature_dim


class BLSTMPoolModule(Module):

    def __init__(self, hidden_size, pool=None, dropout=None):
        super().__init__()
        self.lstm_out_dim = FeatureDim("lstm_out_dim", dimension=hidden_size)
        self.out_feature_dim = self.lstm_out_dim + self.lstm_out_dim
        self.fw_rec = LSTM(out_dim=self.lstm_out_dim)
        self.bw_rec = LSTM(out_dim=self.lstm_out_dim)
        self.pool = pool
        self.dropout = dropout

    @nn.scoped
    def __call__(self, inp, time_axis, feature_axis):
        fw_out, _ = self.fw_rec(inp, direction=1, axis=time_axis)
        bw_out, _ = self.bw_rec(inp, direction=-1, axis=time_axis)
        c = concat((fw_out, self.lstm_out_dim), (bw_out, self.lstm_out_dim))
        if self.pool is not None and self.pool > 1:
            #pool, pool_spatial_dim = nn.pool1d(concat, mode="max", pool_size=self.pool, padding="same", in_spatial_dims=nn.any_spatial_dim)
            pool, pool_spatial_dim = pool1d(c, mode="max", pool_size=self.pool, padding="same", in_spatial_dim=time_axis)
            inp = dropout(pool, self.dropout, axis=self.out_feature_dim)
        else:
            inp = dropout(c, self.dropout, axis=self.out_feature_dim)
        return inp

class SoftmaxCtcLossModule(Module):

    def __init__(self,
                 align_target_key,
                 **kwargs):
        """
        :param str|None align_target_key:
        """
        super().__init__()
        self.align_target_key = align_target_key

    @nn.scoped
    def __call__(self, inp, in_feature_axis):
        softmax_out = nn.log_softmax(inp, axis=in_feature_axis)
        ctc_loss = nn.fast_baum_welch(softmax_out,
                           align_target="ctc",
                           align_target_key=self.align_target_key,
                           ctc_opts={"ignore_longer_outputs_than_inputs": True})
        return ctc_loss


class ConvBLSTMEncoder(Module):

    def __init__(self, in_feature_dim: Dim, in_time_dim: Dim,
                 conv_filter_sizes=[(3, 3), (3, 3)], conv_pool_sizes=[(1, 2), (1, 2)],
                 conv_channel_sizes=[32, 32], conv_activation=relu, num_lstm_layers=6, lstm_single_dim=1024,
                 lstm_pool_sizes=[3, 2], specaugment_settings=None):
        """

        :param l2:
        :param audio_feature_key:
        :param target_label_key:
        :param conv_dropout:
        :param conv_filter_sizes:
        :param conv_pool_sizes:
        :param conv_channel_sizes:
        :param num_lstm_layers:
        :param lstm_single_dim:
        :param lstm_dropout:
        :param lstm_pool_sizes:
        :param SpecAugmentSettings specaugment_settings:
        """
        super().__init__()
        assert num_lstm_layers >= 2, "Needs two lstm layers as the last layer lstm layer is special"

        self.in_feature_dim = in_feature_dim
        self.in_time_dim = in_time_dim
        self.spec_settings = specaugment_settings

        self.conv_block = Encoder2DConvBlock(
            in_dim=self.in_feature_dim,
            in_time_dim=self.in_time_dim,
            filter_sizes=conv_filter_sizes,
            pool_sizes=conv_pool_sizes, channel_sizes=conv_channel_sizes,
            activation=conv_activation
        )

        self.lstm_layers = Sequential()
        for i in range(num_lstm_layers - 1):
            pool_size = lstm_pool_sizes[i] if i < len(lstm_pool_sizes) else 1
            self.lstm_layers.append(
                BLSTMPoolModule(
                    hidden_size=lstm_single_dim, dropout=0.3, pool=pool_size))

        self.last_lstm_layer = BLSTMPoolModule(
            hidden_size=lstm_single_dim, dropout=0.3)

        # self.encoder_state_copy_layer = layers.Copy()

        self.ctc_loss_block = SoftmaxCtcLossModule(align_target_key="bpe_labels")

    @nn.scoped
    def __call__(self, audio_tensor: Tensor, target_tensor: Tensor, **kwargs) -> Tensor:
        x = specaugment(audio_tensor, **self.spec_settings.get_options())
        x, out_time_dim, out_feature_dim = self.conv_block(x)
        x = self.lstm_layers(x, time_axis=out_time_dim, feature_axis=out_feature_dim)
        feature_dim = self.lstm_layers[-1].out_feature_dim
        ctc_loss = self.ctc_loss_block(x, in_feature_axis=feature_dim)
        ctc_loss.mark_as_loss(1.0)
        encoder_state = nn.copy(x, name="encoder_state")
        return encoder_state


class EncoderWrapper(Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = ConvBLSTMEncoder(**kwargs)

    @nn.scoped
    def __call__(self, audio_tensor, target_tensor) -> Tensor:
        out = self.encoder(audio_tensor, target_tensor)
        return out


static_decoder = {
    'output': { 'class': 'rec',
                'from': [],
                'max_seq_len': "max_len_from('base:encoder')",
                'target': 'bpe_labels',
                'unit': { 'accum_att_weights': { 'class': 'eval',
                                                 'eval': 'source(0) + source(1) * source(2) * 0.5',
                                                 'from': ['prev:accum_att_weights', 'att_weights', 'inv_fertility'],
                                                 'out_type': {'dim': 1, 'shape': (None, 1)}},
                          'att': {'axes': 'except_batch', 'class': 'merge_dims', 'from': 'att0'},
                          'enc_transformed': {'class': 'linear', 'from': 'base:encoder/encoder_state', 'n_out': 1024},
                          'inv_fertility': {'class': 'linear', 'activation': 'sigmoid', 'from': 'base:encoder/encoder_state', 'n_out': 1, 'with_bias': False},
                          'att0': {'base': 'base:encoder/encoder_state', 'class': 'generic_attention', 'weights': 'att_weights'},
                          'att_weights': {'class': 'dropout', 'dropout': 0.3, 'dropout_noise_shape': {'*': None}, 'from': 'att_weights0'},
                          'att_weights0': {'class': 'softmax_over_spatial', 'from': 'energy'},
                          'end': {'class': 'compare', 'from': 'output', 'kind': 'equal', 'value': 0},
                          'energy': {'activation': None, 'class': 'linear', 'from': 'energy_tanh', 'n_out': 1, 'with_bias': False},
                          'energy_in': {'class': 'combine', 'from': ['enc_transformed', 'weight_feedback', 's_transformed'], 'kind': 'add', 'n_out': 1024},
                          'energy_tanh': {'activation': 'tanh', 'class': 'activation', 'from': 'energy_in'},
                          'exp_energy': {'activation': 'exp', 'class': 'activation', 'from': 'energy'},
                          'output': {'beam_size': 12, 'class': 'choice', 'from': 'output_prob', 'initial_output': 0, 'target': 'bpe_labels'},
                          'output_prob': { 'L2': 0.001,
                                           'class': 'softmax',
                                           'dropout': 0.3,
                                           'from': 'readout',
                                           'loss': 'ce',
                                           'loss_opts': {'label_smoothing': 0.1},
                                           'target': 'bpe_labels'},
                          'readout': {'class': 'reduce_out', 'from': 'readout_in', 'mode': 'max', 'num_pieces': 2},
                          'readout_in': { 'activation': None,
                                          'class': 'linear',
                                          'from': ['s', 'prev:target_embed', 'att'],
                                          'n_out': 1000,
                                          'with_bias': True},
                          's': { 'class': 'rnn_cell',
                                 'from': ['prev:target_embed', 'prev:att'],
                                 'n_out': 1000,
                                 'unit': 'zoneoutlstm',
                                 'unit_opts': {'zoneout_factor_cell': 0.15, 'zoneout_factor_output': 0.05}},
                          's_transformed': {'activation': None, 'class': 'linear', 'from': 's', 'n_out': 1024, 'with_bias': False},
                          'target_embed': {'class': 'dropout', 'dropout': 0.3, 'dropout_noise_shape': {'*': None}, 'from': 'target_embed0'},
                          'target_embed0': { 'activation': None,
                                             'class': 'linear',
                                             'from': 'output',
                                             'initial_output': 0,
                                             'n_out': 621,
                                             'with_bias': False},
                          'weight_feedback': { 'activation': None,
                                               'class': 'linear',
                                               'from': 'prev:accum_att_weights',
                                               'n_out': 1024,
                                               'with_bias': False}}},
    'decision': {'class': 'decide', 'from': 'output', 'loss': 'edit_distance', 'target': 'bpe_labels'},
}


from i6_core.returnn.config import CodeWrapper

def _map(value):
    if isinstance(value, (ReturnnDimTagsProxy, ReturnnDimTagsProxy.DimRefProxy, ReturnnDimTagsProxy.SetProxy)):
        return CodeWrapper(str(value))
    if isinstance(value, dict):
        return {key: _map(value_) for key, value_ in value.items()}
    if isinstance(value, list):
        return [_map(value_) for value_ in value]
    if isinstance(value, tuple):
        return tuple(_map(value_) for value_ in value)
    if isinstance(value, set):
        return set([_map(value_) for value_ in value])
    return value

def get_network(dim_tags_proxy: ReturnnDimTagsProxy, ext_data: nn.Data, feature_dim, time_dim, **kwargs):

    net = EncoderWrapper(
        in_feature_dim=feature_dim,
        in_time_dim=time_dim,
        **kwargs)

    with nn.NameCtx.new_root() as name_ctx_network:
        data = nn.get_extern_data(data=ext_data)
        out = net(
            data,
            None,
            name=name_ctx_network)
        out.mark_as_default_output()

    config = name_ctx_network.get_returnn_config()
    extern_data_dims = list(dim_tags_proxy.dim_refs_by_name.values())
    dim_tags_proxy = dim_tags_proxy.copy()
    config = dim_tags_proxy.collect_dim_tags_and_transform_config(config)

    config = _map(config)
    config['network'].update(copy.deepcopy(static_decoder))

    text_lines = [
        "from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim\n\n",
        "from returnn.config import get_global_config\n",
        "config = get_global_config()\n"]

    for value in extern_data_dims:
        text_lines.append(f"{value.py_id_name()} = config.typed_dict[{value.py_id_name()!r}]\n")
    for key, value in dim_tags_proxy.dim_refs_by_name.items():
        if value not in extern_data_dims:
            text_lines.append(f"{value.py_id_name()} = {value.dim_repr()}\n")

    config["network"]["extern_data"] = config["extern_data"]
    return config["network"], "".join(text_lines)