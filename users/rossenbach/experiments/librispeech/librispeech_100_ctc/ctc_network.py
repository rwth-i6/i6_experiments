from itertools import zip_longest

def make_network_config(fdim, num_classes, dropout=0.1, **kwargs):
    # -- encoder (default: 6 * 512 BLSTM) -- #
    encoder_layers = kwargs.get('encoder_layers', 6)
    encoder_size   = kwargs.get('encoder_size', 512)
    if kwargs.get('subsampling', True):
        maxpool_pos = kwargs.get('maxpool_pos', 'middle')
        if maxpool_pos == 'bottom': max_pool = [2]
        elif maxpool_pos == 'middle': max_pool = [1]*int(encoder_layers/2-1) + [2]
        elif maxpool_pos == 'top': max_pool = [1]*int(encoder_layers-1) + [2]
        elif isinstance(maxpool_pos, list):
            assert len(maxpool_pos) <= encoder_layers, 'invalid maxpool_pos'
            max_pool = maxpool_pos
        else: assert False, 'unknown maxpool_pos %s' %maxpool_pos
    else: max_pool = []
    network, fromList = nn_setup.build_encoder_network(num_layers=encoder_layers, size=encoder_size, max_pool=max_pool, dropout=dropout)


from returnn_common.nn import Module, LayerRef, get_extern_data, get_root_extern_data, NameCtx, make_root_net_dict
from returnn_common import nn

from .specaugment_clean_v2 import specaugment, SpecAugmentSettings

class BLSTMPoolModule(Module):

    def __init__(self, hidden_size, pool, dropout=None, l2=None):
        super().__init__()
        self.fw_rec = nn.LSTM(n_out=hidden_size, direction=1, l2=l2)
        self.bw_rec = nn.LSTM(n_out=hidden_size, direction=-1, l2=l2)
        self.pool = pool
        self.dropout = dropout

    def forward(self, inp):
        fw_out, _ = self.fw_rec(inp)
        bw_out, _ = self.bw_rec(inp)
        concat = nn.concat((fw_out, "F"), (bw_out, "F"))
        if self.pool is not None and self.pool > 1:
            inp = nn.dropout(nn.pool(concat, mode="max", pool_size=(self.pool,), padding="same"), self.dropout)
        else:
            inp = nn.dropout(concat, self.dropout)
        return inp


class BLSTMCTCModel(Module):

    def __init__(self, num_nn, size, max_pool, num_labels, dropout=None, l2=None, specaugment_settings=None):
        """

        :param num_nn:
        :param size:
        :param list[int] max_pool:
        :param dropout:
        :param SpecAugmentSettings specaugment_settings:
        """
        super().__init__()

        self.specaugment_settings = specaugment_settings

        modules = []
        for i, pool in zip_longest(range(num_nn), max_pool):
            modules.append(BLSTMPoolModule(size, pool, dropout=dropout, l2=l2))
        self.blstms = nn.Sequential(modules)
        self.linear = nn.Linear(n_out=num_labels, with_bias=True)

    def forward(self):
        inp = get_root_extern_data("data")
        if self.specaugment_settings:
            inp = specaugment(inp, **self.specaugment_settings.get_options())
        inp = self.blstms(inp)
        out = self.linear(inp)
        out = nn.softmax(out, name="output", axis="F")
        return out


def get_network(*args, **kwargs):
    blstm_ctc = BLSTMCTCModel(*args, **kwargs)
    net_dict = make_root_net_dict(blstm_ctc)

    return net_dict

