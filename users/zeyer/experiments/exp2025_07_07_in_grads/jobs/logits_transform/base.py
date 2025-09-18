from __future__ import annotations
from typing import Union, Any, Sequence, List, Dict
import torch


def make_logits_transform(
    cfg: Union[None, str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]],
) -> List[BaseTransform]:
    if cfg is None:
        return []
    if not isinstance(cfg, (list, tuple)):
        cfg = [cfg]
    transformations = []
    for cfg_ in cfg:
        if isinstance(cfg_, str):
            cfg_ = {"type": cfg_}
        if not isinstance(cfg_, dict):
            raise TypeError(f"Invalid configuration: {cfg_}")
        if "type" not in cfg_:
            raise ValueError(f"Configuration must contain 'type': {cfg_}")
        cfg_ = cfg_.copy()
        cls_name = cfg_.pop("type")
        cls = _get_cls(cls_name)
        transformations.append(cls(**cfg_))
    return transformations


class BaseTransform:
    """
    Base class for all transformations.
    """

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        :param logits: [B,T,C]
        :return: transformed logits [B,T,C]
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


_classes = {}


def _get_cls(cls_name: str) -> type:
    if not _classes:
        from . import fwd_zero_bwd_id
        from . import fwd_scaled_bwd_id

        for mod in [fwd_zero_bwd_id, fwd_scaled_bwd_id]:
            for name, cls in vars(mod).items():
                if isinstance(cls, type) and issubclass(cls, BaseTransform):
                    _classes[name] = cls

        assert _classes

    if cls_name in _classes:
        return _classes[cls_name]
    raise ValueError(f"Class {cls_name} not found in globals.")
