from returnn.tensor import Dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef


class DummyModel(rf.Module):
  def __init__(self):
    super(DummyModel, self).__init__()

    self.layer = rf.Linear(rf.Dim(name="dummy-input", dimension=1), rf.Dim(name="dummy-output", dimension=1))

  def __call__(self, x):
    return self.layer(x)


class MakeModel:
  """for import"""

  def __call__(self) -> DummyModel:
    return self.make_model()

  @classmethod
  def make_model(cls) -> DummyModel:
    return DummyModel()


def from_scratch_model_def() -> DummyModel:
  """Function is run within RETURNN."""

  return MakeModel.make_model()


from_scratch_model_def: ModelDef[DummyModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  from returnn.config import get_global_config
  config = get_global_config()

  model_def = config.typed_value("_model_def")
  model = model_def()
  return model
