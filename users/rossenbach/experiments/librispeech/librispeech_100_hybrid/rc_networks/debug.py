from .default_hybrid import construct_hybrid_network, BLSTMEncoder
from returnn_common import nn
from returnn_common.asr.specaugment import random_mask_v2
from returnn_common.nn.encoder import ISeqFramewiseEncoder

from returnn.datasets import init_dataset
from returnn.config import Config
from returnn.tf.engine import Engine

import better_exchook
better_exchook.install()


def specaugment_v2(x: nn.Tensor, *,
                   spatial_dim: nn.Dim,
                   feature_dim: nn.Dim = nn.NotSpecified,
                   global_train_step_dependent: bool = True,
                   only_on_train: bool = True,
                   ) -> nn.Tensor:
    """
    SpecAugment reimplementation of :func:`specaugment_v1`
    """
    if feature_dim is nn.NotSpecified:
        assert x.feature_dim
        feature_dim = x.feature_dim
    if global_train_step_dependent:
        step = nn.global_train_step()
        step1 = nn.where(step >= 1000, 1, 0)
        step2 = nn.where(step >= 2000, 1, 0)
    else:
        step1 = step2 = 1
    time_factor = 1

    #with nn.Cond(nn.train_flag() | (not only_on_train)) as cond:
    #    x_masked = x
    #    spatial_len = nn.dim_value(spatial_dim)
    #    # time mask
    #    x_masked = random_mask_v2(
    #        x_masked, mask_axis=spatial_dim, broadcast_axis=feature_dim,
    #        min_num=nn.minimum(step1 + step2, spatial_len),
    #        max_num=nn.minimum(nn.maximum(spatial_len // 100, 2) * (1 + step1 + step2 * 2), spatial_len),
    #        max_dims=20 // time_factor)
    #    # feature mask
    #    # x_masked = random_mask_v2(
    #    #     x_masked, mask_axis=feature_dim, broadcast_axis=spatial_dim,
    #    #     min_num=step1 + step2, max_num=2 + step1 + step2 * 2,
    #    #     max_dims=feature_dim.dimension // 5)
    #    # cond.true = x_masked
    #    cond.false = x

    spatial_len = nn.dim_value(spatial_dim)
    # return cond.result
    x_masked = random_mask_v2(
        x, mask_axis=spatial_dim, broadcast_axis=feature_dim,
        min_num=nn.minimum(step1 + step2, spatial_len),
        max_num=nn.minimum(nn.maximum(spatial_len // 100, 2) * (1 + step1 + step2 * 2), spatial_len),
        max_dims=20 // time_factor)
    return x_masked





class BLSTMEncoderMinimal(ISeqFramewiseEncoder):
    """
    BLSTM encoder with specaugment
    """

    def __init__(self, label_feature_dim):
        super().__init__()

    def __call__(self, source: nn.Tensor, *, spatial_dim: nn.Dim) -> nn.Tensor:
        return specaugment_v2(source, spatial_dim=spatial_dim, feature_dim=source.feature_dim)


data_time = nn.SpatialDim("data_time", None)
data_feature = nn.FeatureDim("data_feature", 5)
classes_feature = nn.FeatureDim("classes_feature", 12)
data = nn.Data(
    name="data",
    dim_tags=[nn.batch_dim, data_time, data_feature],
    available_for_inference=True,
)
classes = nn.Data(
    name="classes",
    dim_tags=[nn.batch_dim, data_time],
    sparse_dim=classes_feature,
    available_for_inference=False,
)

def _config_get_network(epoch: int, **_kwargs) -> dict:
    nn.reset_default_root_name_ctx()
    net = construct_hybrid_network(
        epoch=0,
        train=True,
        encoder=BLSTMEncoderMinimal,
        audio_data=data,
        label_data=classes,
    )
    return nn.get_returnn_config().get_net_dict_raw_dict(net)

class DummyNet(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, source: nn.Tensor, spatial_dim: nn.Dim):
        return random_mask_v2(x=source, mask_axis=spatial_dim, broadcast_axis=source.feature_dim, min_num=1, max_num=2, max_dims=3)

def _config_get_spec_network(epoch: int) -> dict:
    nn.reset_default_root_name_ctx()
    net = DummyNet()
    data_tensor = nn.get_extern_data(data)
    spatial_dim = data.dim_tags[data.time_dim_axis]
    out = net(
        source=data_tensor,
        spatial_dim=spatial_dim
    )
    out.mark_as_default_output()
    return nn.get_returnn_config().get_net_dict_raw_dict(net)


data_args = data.get_kwargs()
data_args.pop("name")
classes_args = classes.get_kwargs()
classes_args.pop("name")

extern_data = {
    "data": data_args,
    "classes": classes_args,
}

config = Config({
    "task": "train", "num_epochs": 1, "start_epoch": 1,
    "get_network": _config_get_spec_network,
    "extern_data": extern_data,
})
train_dataset = init_dataset(
    {"class": "DummyDataset", "input_dim": 5, "output_dim": 12, "num_seqs": 3})
engine = Engine(config)
engine.init_train_from_config(config, train_data=train_dataset)
engine.train()