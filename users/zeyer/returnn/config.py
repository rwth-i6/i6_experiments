"""
RETURNN config utils
"""

from __future__ import annotations
from typing import Optional, Any, Dict, Tuple


def config_dict_update_(config: Dict[str, Any], update: Dict[str, Any]) -> None:
    """
    Mostly just config.update(update), but some special handling for some entries like "behavior_version",
    which would take the max between both.

    :param config: dict to update. update is inplace
    :param update: dict to update from
    """
    for key, value in update.items():
        if key == "behavior_version" and value and config.get(key):
            config[key] = max(config[key], value)
        else:
            config[key] = value


def pop_from_config_post_config(
    config: Optional[Dict[str, Any]], post_config: Optional[Dict[str, Any]], *, key: str, prev: Any = None
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Any]:
    """
    Take key from config dict and merge it with prev value.
    Then take key from post_config dict and merge it with the result.
    Returns the updated config dicts (with key removed) and the merged value.
    """
    config, prev = pop_from_config(config, key=key, prev=prev)
    post_config, prev = pop_from_config(post_config, key=key, prev=prev)
    return config, post_config, prev


def pop_from_config(
    config: Optional[Dict[str, Any]], *, key: str, prev: Any = None
) -> Tuple[Optional[Dict[str, Any]], Any]:
    """
    Take key from config dict and merge it with prev value.
    Returns the updated config dict (with key removed) and the merged value.
    """
    if not config:
        return config, prev
    if key not in config:
        return config, prev
    config = config.copy()
    value = config.pop(key)
    return config, merge_option(prev, value)


def merge_option(prev: Any, new: Any) -> Any:
    """
    E.g. consider some option like env_updates or so.
    """
    if prev is None:
        return new
    if new is None:
        return prev
    if isinstance(prev, dict) and isinstance(new, dict):
        merged = prev.copy()
        merged.update(new)
        return merged
    if isinstance(prev, list) and isinstance(new, list):
        return prev + new
    return new
