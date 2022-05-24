from typing import Optional
from returnn_common import nn

def construct_network(
        epoch: int,
        net_module: nn.Module,
        audio_data: nn.Data,
        label_data: nn.Data,
        audio_feature_dim: nn.Dim,
        audio_time_dim: nn.Dim,
        label_time_dim: nn.Dim,
        label_dim: nn.Dim,
        weight_decay: Optional[float],
        **kwargs):
    net = net_module(audio_feature_dim=audio_feature_dim, target_vocab=label_dim, **kwargs)
    out = net(
        audio_features=nn.get_extern_data(audio_data),
        labels=nn.get_extern_data(label_data),
        audio_time_dim=audio_time_dim,
        label_time_dim=label_time_dim,
        label_dim=label_dim,
    )
    out.mark_as_default_output()
    if weight_decay:
        for param in net.parameters():
            param.weight_decay = weight_decay

    return net