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
        from i6_experiments.users.zeyer import serialization_v2

        if serialization_v2.in_serialize_config():
            return functools.partial(Dim, **self.opts), ()

        # Generic fallback: Serialize as-is (as DelayedReduceDim).
        # E.g. when we pickle the job state.
        # Otherwise, we would create the Dim already during unpickling of the job state,
        # which we don't want.
        # We basically only want the Dim to be created when we actually use it in RETURNN,
        # i.e. the reduce logic when we serialize for the RETURNN config.
        return DelayedReduceDim, (self.opts,)
