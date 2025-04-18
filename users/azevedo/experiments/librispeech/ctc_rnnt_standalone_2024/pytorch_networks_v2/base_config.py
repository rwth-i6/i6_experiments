import importlib
from dataclasses import dataclass, asdict
import torch.nn as nn
from i6_models.config import ModelConfiguration

@dataclass
class BaseConfig(ModelConfiguration):
    config_class: str = ""
    module_class: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.config_class = self.__class__.__qualname__
        self.module_class = self.__class__.__module__
    
    @classmethod
    def load_config(cls, data: dict):
        # dynamically import config classes
        if "module_class" in data:
            module = importlib.import_module(data["module_class"])
            config_class: BaseConfig = getattr(module, data["config_class"])
            return config_class.from_dict(data)
        else:
            raise NotImplementedError
            # return data
    
    @classmethod
    def from_dict(cls, data: dict):
        raise NotImplementedError
    
    def module():
        raise NotImplementedError
