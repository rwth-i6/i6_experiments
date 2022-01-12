from returnn_common import nn

from .specaugment_clean_v2 import specaugment, SpecAugmentSettings

class BLSTMPoolModule(nn.Module):

    def __init__(self, hidden_size, pool, dropout=None, l2=None):
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
            inp = nn.dropout(nn.pool(concat, mode="max", pool_size=(self.pool,), padding="same", in_spatial_dims=nn.any_spatial_dim), self.dropout, axis=nn.any_feature_dim)
        else:
            inp = nn.dropout(concat, self.dropout, axis=nn.any_feature_dim)
        return inp


class BLSTMCTCModel(nn.Module):

    def __init__(self, num_nn, size, max_pool, num_labels, dropout=None, l2=None, feature_dropout=False, specaugment_settings=None):
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

        modules = []
        for i in range(num_nn - 1):
            pool = max_pool[i] if i < len(max_pool) else 1
            modules.append(BLSTMPoolModule(size, pool, dropout=dropout, l2=l2))
        last_pool = max_pool[-1] if len(max_pool) == num_nn else 1
        self.last_blstm = BLSTMPoolModule(size, last_pool, dropout=dropout, l2=l2)
        self.blstms = nn.Sequential(modules)

        self.linear_out_dim = nn.FeatureDim("encoder_linear_out", dimension=num_labels)
        self.linear = nn.Linear(out_dim=num_labels, with_bias=True)

    def __call__(self, data, axis):
        inp = data
        #if self.specaugment_settings:
        #    inp = specaugment(inp, **self.specaugment_settings.get_options())
        if self.feature_dropout:
            inp = nn.dropout(inp, self.dropout, axis=nn.any_feature_dim)
        inp = self.blstms(inp, axis=axis)
        inp = self.last_blstm(inp, axis)
        out = self.linear(inp)
        out = nn.softmax(out, name="output", axis=nn.any_feature_dim)
        return out


def get_network(*args, **kwargs):
    with nn.NameCtx.new_root() as name_ctx:
        time_dim = nn.SpatialDim("time")
        in_dim = nn.FeatureDim("input", ...)
        data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
        blstm_ctc = BLSTMCTCModel(*args, **kwargs)
        out = blstm_ctc(data, time_dim)
        assert isinstance(out, nn.Layer)
        out.mark_as_default_output()

    config_code = name_ctx.get_returnn_config_serialized()
    config = name_ctx.get_returnn_config()
    dim_tags_proxy = nn.ReturnnDimTagsProxy()
    config = dim_tags_proxy.collect_dim_tags_and_transform_config(config)

    text_lines = ["from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim\n\n"]
    text_lines += dim_tags_proxy.py_code_str()

    return config["network"], text_lines

