from returnn_common import nn
from returnn_common.asr.specaugment import random_mask_v2, specaugment_v2

from returnn.datasets import init_dataset
from returnn.config import Config
from returnn.tf.engine import Engine

import better_exchook
better_exchook.install()

data_time = nn.SpatialDim("data_time", None)
data_feature = nn.FeatureDim("data_feature", 5)
data = nn.Data(
    name="data",
    dim_tags=[nn.batch_dim, data_time, data_feature],
    available_for_inference=True,
)

class DummyNet(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, source: nn.Tensor, spatial_dim: nn.Dim):
        return specaugment_v2(x=source, spatial_dim=spatial_dim)
        # return random_mask_v2(x=source, mask_axis=spatial_dim, broadcast_axis=source.feature_dim, min_num=1, max_num=2, max_dims=3)


def _config_get_spec_network(epoch: int) -> dict:
    nn.reset_default_root_name_ctx()
    net = DummyNet()
    data_tensor = nn.get_extern_data(data)
    spatial_dim = data.dim_tags[data.time_dim_axis]
    out = net(source=data_tensor, spatial_dim=spatial_dim)
    out.mark_as_default_output()
    config_code = nn.get_returnn_config().get_complete_py_code_str(net)
    print(config_code)  # I will also provide this as gist
    return nn.get_returnn_config().get_net_dict_raw_dict(net)

data_args = data.get_kwargs()
data_args.pop("name")
extern_data = {"data": data_args,}

config = Config({
    "task": "train", "num_epochs": 1, "start_epoch": 1,
    "get_network": _config_get_spec_network,
    "extern_data": extern_data,
    "behavior_version": 12
})
train_dataset = init_dataset(
    {"class": "DummyDataset", "input_dim": 5, "output_dim": 12, "num_seqs": 3})
engine = Engine(config)
engine.init_train_from_config(config, train_data=train_dataset)
engine.train()
