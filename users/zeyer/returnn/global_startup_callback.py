"""
Global startup func.
All RETURNN jobs here can use this (add `startup_callback=global_startup_callback` to your ``post_config``).
We might do some extra checks, fixes, etc (depending on environment).
This should be optional and usually not needed if everything is set up correctly (thus ``post_config``).
"""

from __future__ import annotations
from typing import Optional, Any, Dict, List
import textwrap
from i6_experiments.common.setups import serialization


def global_startup_callback(*_args, **_kwargs):
    print("RETURNN global startup callback.")


def serialize_global_startup_callback() -> serialization.NonhashedCode:
    """
    For old-style RETURNN config serialization.
    """
    return serialization.NonhashedCode(
        textwrap.dedent(f"""\
            def _global_startup_callback(*args, **kwargs):
                try:
                    from {__name__} import global_startup_callback
                except ImportError as exc:
                    print("Warning: could not import {__name__}.global_startup_callback: %s" % exc)
                else:
                    global_startup_callback()
                    
            startup_callback = _global_startup_callback
            """)
    )


def maybe_add_global_startup_callback_to_post_config(config: Dict[str, Any], post_config: Dict[str, Any]) -> None:
    """
    Maybe add the global startup callback to the given config, if no other startup_callback is already defined.
    """
    if "startup_callback" in config:
        return
    if "startup_callback" in post_config:
        return
    post_config["startup_callback"] = global_startup_callback


def maybe_serialize_global_startup_callback(*configs: Optional[Dict[str, Any]]) -> List[serialization.NonhashedCode]:
    """
    Maybe add the global startup callback serialization, if no other startup_callback is already defined.
    """
    for config in configs:
        if config and "startup_callback" in config:
            return []
    return [serialize_global_startup_callback()]
