from sisyphus import *

from i6_core import returnn

from returnn_common import nn, tests
from returnn_common.nn import conformer
from returnn_common.tests import returnn_helpers

from typing import Tuple, Dict, Any

net = conformer.ConformerEncoder(num_layers=12)
data_data = nn.Data(
    name="data",
    shape=(None, 40),
    batch_dim_axis=0,
    time_dim_axis=1,
    feature_dim_axis=2,
)
data = nn.get_extern_data(
    data_data
)
# print(net(data, in_spatial_dim=data_data.get_time_dim_tag()))

# print(net.name_ctx.children)

with nn.NameCtx.new_root() as name_ctx:
    data_data = nn.Data(
        name="data",
        shape=(None, 40),
        batch_dim_axis=0,
        time_dim_axis=1,
        feature_dim_axis=2,
    )


    # extern_data_dict = {
    #     "data": data_data,
    # }
    dim_tags_proxy = nn.ReturnnDimTagsProxy()
    # ed_config = dim_tags_proxy.collect_dim_tags_and_transform_config(extern_data_dict)
    data = nn.get_extern_data(
        data_data
    )
    net = conformer.ConformerEncoder(num_layers=12)
    config = name_ctx.get_returnn_config()

    # print(config.get_config_raw_dict(net))
    # print(dim_tags_proxy.collect_dim_tags_and_transform_config(config)["network"])

# net_dict = returnn_helpers.dummy_config_net_dict(net, with_axis=True)

# print(net_dict)

def config_net_dict_via_serialized(config_code: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """
  :param str config_code: via get_returnn_config_serialized
  :return: config, net_dict
  """
  from returnn.util import better_exchook
  print(config_code)
  scope = {}
  src_filename = "<config_net_dict_via_serialized>"
  better_exchook.set_linecache(src_filename, config_code)
  code_ = compile(config_code, src_filename, "exec")
  exec(code_, scope, scope)
  for tmp in ["__builtins__", "Dim", "batch_dim", "FeatureDim", "SpatialDim"]:
    scope.pop(tmp)
  config = scope
  net_dict = config["network"]
  return config, net_dict

def dummy_config_net_dict(
    net: nn.Module, *,
    with_axis=False, in_dim: int = 13, reset_name_ctx: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    :return: config, net_dict
    """
    if reset_name_ctx:
        nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("input", in_dim)
    data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
    opts = {}
    if with_axis:
        opts["in_spatial_dim"] = time_dim
    out = net(data, **opts)
    if isinstance(out, tuple):
        out = out[0]
    assert isinstance(out, nn.Tensor)
    out.mark_as_default_output()

    config_code = nn.get_returnn_config().get_complete_py_code_str(net)
    # return config_net_dict_via_serialized(config_code)
    return nn.get_returnn_config().get_config_raw_dict(net)

net_dict = dummy_config_net_dict(net, with_axis=True)
print(net_dict)

returnn_config = returnn.ReturnnConfig(net_dict)

# write_job = returnn.WriteReturnnConfigJob(returnn_config=returnn_config)
# tk.register_output("config/baseline_conformer.config", write_job.out_returnn_config_file)

# print(nn.get_returnn_config().get_complete_py_code_str(net))