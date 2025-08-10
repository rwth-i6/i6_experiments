from typing import Any, Dict, Optional, Sequence, Union
from .model import ModelDefWithCfg, ModelDef
from .model_with_checkpoints import ModelWithCheckpoint


def get_from_config(
    config_sources: Sequence[Optional[Union[Dict[str, Any], ModelWithCheckpoint, ModelDefWithCfg, ModelDef]]],
    attr: str,
    default: Any = None,
):
    for src in config_sources:
        if src is None:
            continue
        if isinstance(src, ModelWithCheckpoint):
            src = src.definition
        if isinstance(src, ModelDefWithCfg):
            src = src.config
            assert isinstance(src, dict)
        if not isinstance(src, dict):
            # Assume ModelDef. Just some sanity checks:
            assert hasattr(src, "behavior_version") and hasattr(src, "backend")
            continue
        assert isinstance(src, dict)
        if src.get(attr) is not None:
            return src[attr]
    return default
