from typing import *

import time

from returnn_common.nn.transformer import Transformer

from returnn_common import nn


class BLSTMPoolModule(nn.Module):

    def __init__(self, hidden_size, pool=None, dropout=None):
        super().__init__()
        self.lstm_out_dim = nn.FeatureDim("lstm_out_dim", dimension=hidden_size)
        self.out_feature_dim = self.lstm_out_dim + self.lstm_out_dim
        self.fw_rec = nn.LSTM(out_dim=self.lstm_out_dim)
        self.bw_rec = nn.LSTM(out_dim=self.lstm_out_dim)
        self.pool = pool
        self.dropout = dropout

    @nn.scoped
    def __call__(self, inp, time_axis):
        fw_out, _ = self.fw_rec(inp, direction=1, axis=time_axis)
        bw_out, _ = self.bw_rec(inp, direction=-1, axis=time_axis)
        c = nn.concat((fw_out, self.lstm_out_dim), (bw_out, self.lstm_out_dim))
        if self.pool is not None and self.pool > 1:
            pool, pool_spatial_dim = nn.pool1d(c, mode="max", pool_size=self.pool, padding="same", in_spatial_dim=time_axis)
            inp = nn.dropout(pool, self.dropout, axis=self.out_feature_dim)
            out_time_dim = pool_spatial_dim
        else:
            inp = nn.dropout(c, self.dropout, axis=self.out_feature_dim)
            out_time_dim = time_axis
        return inp, out_time_dim


class BLSTMDownsamplingTransformerASR(nn.Module):
    """
    Standard Transformer Module
    """
    def __init__(self,
                 audio_feature_dim: nn.Dim,
                 target_vocab: nn.Dim,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.downsampling_1 = BLSTMPoolModule(hidden_size=256, pool=2, dropout=0.1)
        self.downsampling_2 = BLSTMPoolModule(hidden_size=256, pool=2, dropout=0.1)
        self.transformer = Transformer(model_dim=self.downsampling_2.out_feature_dim, target_dim=target_vocab, **kwargs)

        #self.input_linear = nn.Linear(out_dim=self.transformer.model_dim, in_dim=audio_feature_dim)

    @nn.scoped
    def __call__(self,
                 *,
                 audio_features: nn.Tensor,
                 labels: nn.Tensor,
                 audio_time_dim: nn.Dim,
                 label_time_dim: nn.Dim,
                 label_dim: nn.Dim,
                 ):
        pool1_out, pool1_time_axis = self.downsampling_1(audio_features, time_axis=audio_time_dim)
        pool2_out, pool2_time_axis = self.downsampling_2(pool1_out, time_axis=pool1_time_axis)

        encoder_out, out_logits, out_labels, _ = self.transformer(
            pool2_out,
            source_spatial_axis=pool2_time_axis,
            target=labels,
            target_spatial_axis=label_time_dim
        )

        loss = nn.sparse_softmax_cross_entropy_with_logits(
            logits=out_logits,
            targets=labels,
            axis=label_dim,
        )
        loss.mark_as_loss()

        return out_logits


def get_network(dim_tags_proxy: nn.ReturnnDimTagsProxy, source_data: nn.Data, target_data: nn.Data, feature_dim, time_dim, label_dim, label_time_dim, **kwargs):

    start = time.time()
    with nn.NameCtx.new_root() as name_ctx_network:
        print("context: %f" % (time.time() - start))
        start = time.time()
        net = BLSTMDownsamplingTransformerASR(audio_feature_dim=feature_dim, target_vocab=label_dim)
        print("net building: %f" % (time.time() - start))
        start = time.time()
        out = net(
            audio_features=nn.get_extern_data(source_data),
            labels=nn.get_extern_data(target_data),
            audio_time_dim=time_dim,
            label_time_dim=label_time_dim,
            label_dim=label_dim,
        )
        print("net calling: %f" % (time.time() - start))
        start = time.time()
        out.mark_as_default_output()
        print("mark output: %f" % (time.time() - start))

        start = time.time()
        for param in net.parameters():
            param.weight_decay = 0.1
        print("weight decay: %f" % (time.time() - start))

        start = time.time()
        serializer = nn.ReturnnConfigSerializer(name_ctx_network)
        print("building serializer %f" % (time.time() - start))
        start = time.time()
        base_string = serializer.get_base_extern_data_py_code_str()
        print("extern data string: %f" % (time.time() - start))
        start = time.time()
        network_string = serializer.get_ext_net_dict_py_code_str(net, ref_extern_data_dims_via_global_config=True)
        print("network string: %f" % (time.time() - start))

    return network_string, base_string

    # config = name_ctx_network.get_returnn_config()
    # extern_data_dims = list(dim_tags_proxy.dim_refs_by_name.values())
    # dim_tags_proxy = dim_tags_proxy.copy()
    # config = dim_tags_proxy.collect_dim_tags_and_transform_config(config)

    # config = resolve_dim_proxies(config)

    # text_lines = [
    #     "from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim\n\n",
    #     "from returnn.config import get_global_config\n",
    #     "config = get_global_config()\n"]

    # for value in extern_data_dims:
    #     text_lines.append(f"{value.py_id_name()} = config.typed_dict[{value.py_id_name()!r}]\n")
    # text_lines.append(dim_tags_proxy.py_code_str(exclude_dims=extern_data_dims))

    # return config["network"], "".join(text_lines)
