from dataclasses import dataclass
from i6_models.config import ModelConfiguration



@dataclass
class BaseConfig(ModelConfiguration):
    """Base class for all configs"""
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return cls(**d)