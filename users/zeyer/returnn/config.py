"""
RETURNN config utils
"""

from typing import Dict, Any


def config_dict_update_(config: Dict[str, Any], update: Dict[str, Any]) -> None:
    """
    Mostly just config.update(update), but some special handling for some entries like "behavior_version",
    which would take the max between bot.

    :param config: dict to update. update is inplace
    :param update: dict to update from
    """
    for key, value in update.items():
        if key == "behavior_version" and value and config.get(key):
            config[key] = max(config[key], value)
        else:
            config[key] = value
