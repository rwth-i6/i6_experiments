from typing import *

from i6_experiments.common.setups.returnn_common.util import resolve_dim_proxies
from i6_experiments.common.setups.returnn_common.config import get_network_config_and_prolog

from returnn_common.nn.transformer import Transformer

from returnn_common import nn

class TransformerASR(nn.Module):
    """
    Standard Transformer Module
    """
    def __init__(self,
                 audio_feature_dim: nn.Dim,
                 target_vocab: nn.Dim,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.transformer = Transformer(target_vocab=target_vocab, **kwargs)

        self.input_linear = nn.Linear(out_dim=self.transformer.out_dim, in_dim=audio_feature_dim)

    @nn.scoped
    def __call__(self,
                 *,
                 audio_features: nn.Tensor,
                 labels: nn.Tensor,
                 audio_time_dim: nn.Dim,
                 label_time_dim: nn.Dim,
                 label_dim: nn.Dim,
                 ):
        in_vector = self.input_linear(audio_features)

        encoder_out, out_logits, out_labels, _ = self.transformer(
            in_vector,
            source_spatial_axis=audio_time_dim,
            target=labels,
            target_spatial_axis=label_time_dim
        )

        loss = nn.sparse_softmax_cross_entropy_with_logits(
            logits=out_logits,
            targets=labels,
            axis=label_dim,
        )
        loss.mark_as_loss()

        nn.

        return out_logits


def get_network(dim_tags_proxy: nn.ReturnnDimTagsProxy, source_data: nn.Data, target_data: nn.Data, feature_dim, time_dim, label_dim, label_time_dim, **kwargs):

    with nn.NameCtx.new_root() as name_ctx_network:
        net = TransformerASR(audio_feature_dim=feature_dim, target_vocab=label_dim)
        out = net(
            audio_features=nn.get_extern_data(source_data),
            labels=nn.get_extern_data(target_data),
            audio_time_dim=time_dim,
            label_time_dim=label_time_dim,
            label_dim=label_dim,
            name=name_ctx_network,
        )
        out.mark_as_default_output()

        #out_dim = nn.FeatureDim("L2_dim", 1)
        #reductions = [nn.expand_dim(nn.reduce(param ** 2.0, mode="sum", axis=list(param.shape)), dim=out_dim) for param in net.parameters()]
        #concat_out_dim = len(reductions) * out_dim
        #l2loss = nn.concat(*tuple([[reduction, out_dim] for reduction in reductions]))
        #l2loss = nn.reduce(l2loss, axis=concat_out_dim, mode="sum")
        #l2loss = sum(reductions)
        #l2loss.mark_as_loss(loss_scale=1e-4)

        l2loss = sum(nn.reduce(param ** 2, axis=list(param.shape), mode="sum") for param in net.parameters())
        l2loss.mark_as_loss()

        #l2loss = sum(nn.reduce_sum(param ** 2) for param in net.parameters())
        #l2loss.mark_as_loss()

    config, prolog = get_network_config_and_prolog(dim_tags_proxy, name_ctx_network)

    return config, prolog

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