__all__ = ["Int", "Float", "TDP"]

from typing import Union

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase


Int = Union[int, tk.Variable, DelayedBase]
Float = Union[float, tk.Variable, DelayedBase]
TDP = Union[Float, str]
