"""
helpers for training.

- Uses ``unhashed_package_root`` now, via :func:`i6_experiments.users.zeyer.utils.sis_setup.get_base_module`.
"""
from __future__ import annotations

__all__ = [
    "encoder_forward",
    "get_model_generic",

]

from typing import TYPE_CHECKING, Protocol, Tuple, TypeVar, Union
import torch
from torch import nn

if TYPE_CHECKING:
    from returnn.tensor import TensorDict, Tensor, Dim
    import returnn.frontend as _rf
    from returnn_common import nn as _rc_nn

    ModelT = TypeVar("ModelT", bound=Union[torch.nn.Module, _rf.Module, _rc_nn.Module])
else:
    ModelT = TypeVar("ModelT")

def encoder_forward(*, model: torch.nn.Module, data: Tensor, data_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
    from returnn.config import get_global_config

    config = get_global_config()
    assert config is not None

    if config.typed_value("use_w2v_model", False):
        _, enc, spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
        return enc, spatial_dim
    else:
        raise NotImplementedError("Only W2V model is supported for now.")


class ForwardDef(Protocol[ModelT]):
    """
    Defines the losses (mark_as_loss).
    """

    def __call__(
        self,
        *,
        model: ModelT,
        data: Tensor,
        data_spatial_dim: Dim,
    ):
        raise NotImplementedError


def get_model_generic(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config

    config = get_global_config()
    assert config
    default_input_key = config.typed_value("default_input")
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])

    model_def = config.typed_value("_model_def")
    if model_def is None:
        model_def = config.typed_value("model_def")
    assert model_def is not None, "model_def must be provided in the config"
    model = model_def(epoch=epoch, in_dim=data.feature_dim)
    return model

def forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    from returnn.config import get_global_config
    import returnn.frontend as rf

    config = get_global_config()
    assert config is not None
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    forward_def: ForwardDef = config.typed_value("_forward_def")
    if forward_def is None:
        forward_def = config.typed_value("forward_def")
    assert forward_def is not None, "forward_def must be provided in the config"
    output, spatial_dim = forward_def(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
    )

    run_ctx = rf.get_run_ctx()
    assert run_ctx.expected_outputs is not None
    run_ctx.expected_outputs["output"]._dims = output.dims
    run_ctx.mark_as_default_output(output)


def get_dummy_model(*, epoch: int, **_kwargs):
  """Return empty torch module"""
  return nn.Module()

def forward_step_passthrough(*, model, extern_data, **_kwargs_unused):
  from returnn.config import get_global_config
  import returnn.frontend as rf

  config = get_global_config()
  assert config is not None
  default_input_key = config.typed_value("default_input")
  data = extern_data[default_input_key]

  run_ctx = rf.get_run_ctx()
  assert run_ctx.expected_outputs is not None
  run_ctx.expected_outputs["output"]._dims = data.dims
  run_ctx.mark_as_default_output(data)
