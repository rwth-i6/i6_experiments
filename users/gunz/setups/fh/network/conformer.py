__all__ = ["get_best_model_config"]

import typing

from ...common.conformer.best_conformer import get_best_model_config as get_cfg, Size
from .augment import Network


def get_best_model_config(
    size: typing.Union[Size, int],
    num_classes: int,
    time_tag_name: str,
    *,
    int_loss_at_layer: typing.Optional[int] = None,
    int_loss_scale: typing.Optional[float] = None,
    label_smoothing: float = 0.2,
    target: str = "classes",
) -> Network:
    conformer_net = get_cfg(
        size=size,
        num_classes=num_classes,
        int_loss_at_layer=int_loss_at_layer,
        int_loss_scale=int_loss_scale,
        label_smoothing=label_smoothing,
        target=target,
        time_tag_name=time_tag_name,
    )

    conformer_net.network.pop("output", None)
    conformer_net.network["encoder-output"] = {
        "class": "copy",
        "from": "length_masked",
    }

    return conformer_net
