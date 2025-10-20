"""
This is a Dim where the dim is only computed when it is first accessed,
e.g. because it depends on a vocab that we don't want to load in the Sis manager.
"""

from __future__ import annotations
from typing import Any, Dict
from returnn.tensor import Dim
import functools


class DelayedReduceDim:
    """
    This is for pickle / serialization_v2.
    """

    def __init__(self, opts: Dict[str, Any]):
        self.opts = opts

    def _sis_hash(self) -> bytes:
        from sisyphus.hash import sis_hash_helper  # noqa

        return b"(DelayedReduceDim, " + sis_hash_helper(self.opts) + b")"

    def __reduce__(self):
        return functools.partial(Dim, **self.opts), ()
