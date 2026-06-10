__all__ = ["HashedCode"]

from sisyphus.delayed_ops import DelayedBase, DelayedFormat
from sisyphus.hash import sis_hash_helper

from i6_core.serialization.base import SerializerObject

class HashedCode(SerializerObject):
    def __init__(self, code: DelayedFormat):
        self.code = code
    
    def get(self):
        return self.code.get()
    
    def _sis_hash(self):
        return sis_hash_helper(self.code)
