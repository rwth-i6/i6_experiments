"""
Generic interfaces to define models, training and recognition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any, Union, List, Protocol, TypeVar

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim
    import returnn.frontend as _rf
    from returnn_common import nn as _rc_nn
    import torch

    ModelT = TypeVar("ModelT", bound=Union[torch.nn.Module, _rf.Module, _rc_nn.Module])
else:
    ModelT = TypeVar("ModelT")

if TYPE_CHECKING:
    from i6_experiments.common.setups import serialization


class ModelDef(Protocol[ModelT]):
    """
    Creates the model, per epoch
    """

    def __call__(self, *, epoch: int, in_dim: Dim, target_dim: Dim, training: bool = False) -> ModelT:
        raise NotImplementedError

    behavior_version: int
    # Version 2 (i.e. attrib might not be available in all cases):
    backend: Optional[str] = None
    batch_size_factor: int  # according to some arbitrary reference point


class ModelDefWithCfg:
    """extended ModelDef"""

    def __init__(self, model_def: ModelDef[ModelT], config: Dict[str, Any]):
        assert callable(model_def) and not isinstance(model_def, ModelDefWithCfg)
        self.model_def = model_def
        self.behavior_version = model_def.behavior_version
        self.backend = model_def.backend
        self.batch_size_factor = model_def.batch_size_factor
        self.config = config

    def __call__(self, **kwargs) -> ModelT:
        return self.model_def(**kwargs)


def serialize_model_def(
    model_def: ModelDef,
    *,
    import_as: str = "_model_def",
    unhashed_package_root: Optional[str] = None,
) -> List[serialization.SerializerObject]:
    """
    serialize
    """
    from i6_experiments.common.setups import serialization

    if isinstance(model_def, ModelDefWithCfg):
        model_def = model_def.model_def
    return [
        serialization.Import(
            model_def, import_as=import_as, ignore_import_as_for_hash=True, unhashed_package_root=unhashed_package_root
        )
    ]
