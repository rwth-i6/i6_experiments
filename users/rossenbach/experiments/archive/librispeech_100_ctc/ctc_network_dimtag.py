import copy
import time
start = time.time()
from returnn_common import nn

from .specaugment_clean_v2 import specaugment, SpecAugmentSettings
print("new imports took %f" % (time.time() - start))

class BLSTMPoolModule(nn.Module):

    def __init__(self, hidden_size, pool,dropout=None, l2=None):
        super().__init__()
        self.lstm_out_dim = nn.FeatureDim("lstm_out_dim", dimension=hidden_size)
        self.fw_rec = nn.LSTM(out_dim=self.lstm_out_dim)
        self.bw_rec = nn.LSTM(out_dim=self.lstm_out_dim)
        self.pool = pool
        self.dropout = dropout

    def __call__(self, inp, axis):
        fw_out, _ = self.fw_rec(inp, direction=1, axis=axis)
        bw_out, _ = self.bw_rec(inp, direction=-1, axis=axis)
        concat = nn.concat((fw_out, self.lstm_out_dim), (bw_out, self.lstm_out_dim))
        if self.pool is not None and self.pool > 1:
            #pool, pool_spatial_dim = nn.pool1d(concat, mode="max", pool_size=self.pool, padding="same", in_spatial_dims=nn.any_spatial_dim)
            pool, pool_spatial_dim = nn.pool1d(concat, mode="max", pool_size=self.pool, padding="same", in_spatial_dim=axis)
            inp = nn.dropout(pool, self.dropout, axis=nn.any_feature_dim)
        else:
            inp = nn.dropout(concat, self.dropout, axis=nn.any_feature_dim)
        return inp


class BLSTMCTCModel(nn.Module):

    def __init__(self, num_nn, size, max_pool, num_labels, dropout=None, l2=None, feature_dropout=False, specaugment_settings=None, in_dim=None):
        """

        :param num_nn:
        :param size:
        :param list[int] max_pool:
        :param dropout:
        :param SpecAugmentSettings specaugment_settings:
        """
        super().__init__()

        self.specaugment_settings = specaugment_settings
        self.dropout = dropout
        self.feature_dropout = feature_dropout

        self.out_dim = nn.FeatureDim("ctc_out_dim", dimension=num_labels)

        modules = []
        for i in range(num_nn - 1):
            pool = max_pool[i] if i < len(max_pool) else 1
            modules.append(BLSTMPoolModule(size, pool, dropout=dropout, l2=l2))
        last_pool = max_pool[-1] if len(max_pool) == num_nn else 1
        self.last_blstm = BLSTMPoolModule(size, last_pool, dropout=dropout, l2=l2)
        self.blstms = nn.Sequential(modules)
        from returnn_common.nn.conformer import ConformerEncoder, ConformerEncoderLayer
        linear_dim = nn.FeatureDim("linear_dim", 512)
        self.pre_linear = nn.Linear(linear_dim)
        conf_layer = ConformerEncoderLayer(batch_norm=nn.BatchNorm())
        self.conformer = ConformerEncoder(conf_layer, 2)
        self.linear_out_dim = nn.FeatureDim("encoder_linear_out", dimension=num_labels)
        self.linear = nn.Linear(out_dim=self.out_dim, with_bias=True)
        #self.linear = nn.Linear(out_dim=self.out_dim, with_bias=True, in_dim=in_dim)
        nn.glu

    @nn.scoped
    def __call__(self, data, axis):
        inp = data
        if self.specaugment_settings:
            inp = specaugment(inp, **self.specaugment_settings.get_options())
        if self.feature_dropout:
            inp = nn.dropout(inp, self.dropout, axis=nn.any_feature_dim)
        #inp = self.blstms(inp, axis=axis)
        #inp = self.last_blstm(inp, axis)
        #inp = self.pre_linear(inp)
        #inp, out_spatial_dim = self.conformer(inp, in_spatial_dim=axis)
        out = self.linear(inp)
        #out = nn.softmax(out, name="output", axis=out_spatial_dim)
        out = nn.softmax(out, name="output", axis=axis)
        return out

from i6_core.returnn.config import CodeWrapper

def _map(value):
    if isinstance(value, (nn.ReturnnDimTagsProxy, nn.ReturnnDimTagsProxy.DimRefProxy, nn.ReturnnDimTagsProxy.SetProxy)):
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

def get_network(dim_tags_proxy: nn.ReturnnDimTagsProxy, ext_data, time_dim, *args, **kwargs):

    import time
    start = time.time()
    net = BLSTMCTCModel(*args, **kwargs)


    with nn.NameCtx.new_root() as name_ctx_network:
        data = nn.get_extern_data(data=ext_data)
        out = net(data, axis=time_dim, name=name_ctx_network)
        #linear = nn.Linear(out_dim=out_dim, with_bias=True, in_dim=in_dim)
        #out = linear(data)
        #assert isinstance(out, nn.Layer)
        out.mark_as_default_output()

    end = time.time()

    config_code = name_ctx_network.get_returnn_config_serialized()
    config = name_ctx_network.get_returnn_config()
    extern_data_dims = list(dim_tags_proxy.dim_refs_by_name.values())
    dim_tags_proxy = dim_tags_proxy.copy()
    config = dim_tags_proxy.collect_dim_tags_and_transform_config(config)
    #config['network'] = config['network'][0]

    config = _map(config)
    print(config["extern_data"])
    print(config['network']['linear']['out_shape'])

    text_lines = [
        "from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim\n\n",
        "from returnn.config import get_global_config\n",
        "config = get_global_config()\n"]

    for value in extern_data_dims:
        text_lines.append(f"{value.py_id_name()} = config.typed_dict[{value.py_id_name()!r}]\n")
    for key, value in dim_tags_proxy.dim_refs_by_name.items():
        if value not in extern_data_dims:
            text_lines.append(f"{value.py_id_name()} = {value.dim_repr()}\n")

    config["network"]["prolog"] = "".join(text_lines)
    config["network"]["extern_data"] = config["extern_data"]
    return config["network"]

