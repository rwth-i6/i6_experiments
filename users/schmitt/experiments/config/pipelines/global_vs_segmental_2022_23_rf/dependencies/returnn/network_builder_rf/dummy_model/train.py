from typing import Dict, List, Tuple, Optional, Union

from returnn.tensor import TensorDict
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.dummy_model.model import DummyModel


def _returnn_v2_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()
  train_def: TrainDef = config.typed_value("_train_def")
  train_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    targets=targets,
    targets_spatial_dim=targets_spatial_dim,
  )


def dummy_training(
        *,
        model: DummyModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        targets: rf.Tensor,
        targets_spatial_dim: Dim
):
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  config = get_global_config()  # noqa

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio
  batch_dims = data.remaining_dims(data_spatial_dim)

  loss = model(rf.ones(dims=batch_dims + [model.layer.in_dim], dtype="float32"))
  loss.mark_as_loss("dummy-loss")


dummy_training: TrainDef[DummyModel]
dummy_training.learning_rate_control_error_measure = "dev_score_full_sum"
