from typing import TYPE_CHECKING

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import FramewiseTrainDef

# if TYPE_CHECKING:
from returnn.tensor import TensorDict


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  from returnn.tensor import Tensor, Dim
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")
  non_blank_vocab = config.typed_value("non_blank_vocab")
  data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
  targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
  non_blank_targets = Tensor(
    name="non_blank_targets",
    sparse_dim=Dim(description="non_blank_vocab", dimension=targets.sparse_dim.dimension - 1, kind=Dim.Types.Spatial),
    vocab=non_blank_vocab,
  )

  model_def = config.typed_value("_model_def")
  model = model_def(
    epoch=epoch, in_dim=data.feature_dim, align_target_dim=targets.sparse_dim, target_dim=non_blank_targets.sparse_dim)
  return model


def _returnn_v2_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()
  train_def: FramewiseTrainDef = config.typed_value("_train_def")
  train_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    align_targets=targets,
    align_targets_spatial_dim=targets_spatial_dim,
  )
