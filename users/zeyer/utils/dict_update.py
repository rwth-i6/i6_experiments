"""
Dict update utils
"""

from __future__ import annotations
from typing import Optional, Any, Dict, Sequence


def dict_update_deep(d: Dict[str, Any], deep_updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    :param d: dict to update
    :param deep_updates: might also contain "." in the key, for nested dicts
    :return: updated dict
    """
    if not deep_updates:
        return d
    d = d.copy()
    for k, v in deep_updates.items():
        assert isinstance(k, str)
        if "." in k:
            k1, k2 = k.split(".", 1)
            d[k1] = dict_update_deep(d[k1], {k2: v})
        else:
            d[k] = v
    return d


def dict_update_delete_deep(d: Dict[str, Any], deep_deletes: Optional[Sequence[str]]) -> Dict[str, Any]:
    """
    :param d: dict to update (to delete from)
    :param deep_deletes: might also contain "." in the key, for nested dicts
    :return: updated dict
    """
    if not deep_deletes:
        return d
    d = d.copy()
    for k in deep_deletes:
        assert isinstance(k, str)
        if "." in k:
            k1, k2 = k.split(".", 1)
            d[k1] = dict_update_delete_deep(d[k1], [k2])
        else:
            del d[k]
    return d
