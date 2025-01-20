"""
A set but which behaves like dict about the order of elements when iterating over it,
i.e. deterministic and insertion order.
"""


from __future__ import annotations
from typing import Dict, Any, Sequence, Optional


class SetInsertOrder:
    """
    A set but which behaves like dict about the order of elements when iterating over it,
    i.e. deterministic and insertion order.

    Internally, we currently use a dict but with dummy values (None).
    """

    def __init__(self, values: Optional[Sequence[Any]] = None):
        self._dict: Dict[Any, None] = {}
        if values is not None:
            for value in values:
                self.add(value)

    def add(self, item):
        self._dict[item] = None

    def remove(self, item):
        del self._dict[item]

    def __contains__(self, item):
        return item in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f"SetInsertOrder({list(self)})"
