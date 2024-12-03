"""
Delayed sum
"""

from typing import Sequence
from sisyphus.delayed_ops import DelayedBase
from i6_core.util import instanciate_delayed


class DelayedSum(DelayedBase):
    def __init__(self, elements: Sequence):
        super().__init__(elements, None)

    def get(self):
        elements = instanciate_delayed(self.a)
        return sum(elements)
