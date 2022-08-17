"""
Model logic
"""

import types
from typing import Any
import dataclasses
from i6_core.returnn.training import Checkpoint


# TODO
ModelDefinition = Any


@dataclasses.dataclass(frozen=True)
class Model:
    """
    Model
    """
    definition: ModelDefinition
    checkpoint: Checkpoint


def get_model_definition_from_module(mod: types.ModuleType) -> ModelDefinition:
    pass  # TODO...
