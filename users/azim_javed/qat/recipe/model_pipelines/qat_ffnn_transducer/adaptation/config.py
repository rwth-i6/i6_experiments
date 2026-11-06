from dataclasses import dataclass
from typing import List

from i6_models.config import ModelConfiguration

from ...common.adapters import ErrorAdapterConfig
from ..mem_inited.pytorch_modules import QATFFNNTransducerConfig


@dataclass
class MemristorAdapterConfig(ModelConfiguration):
    base_config: QATFFNNTransducerConfig
    adapters: List[ErrorAdapterConfig]