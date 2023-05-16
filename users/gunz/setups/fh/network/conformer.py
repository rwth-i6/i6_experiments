__all__ = ["get_best_model_config"]

import typing

from i6_core import returnn

from ...common.conformer import attention_for_hybrid
from ...common.conformer.best_conformer import get_best_model_config as get_cfg, Size


def get_best_model_config(
    size: typing.Union[Size, int],
    num_classes: int,
    time_tag_name: str,
    *,
    chunking: typing.Optional[str] = None,
    focal_loss_factor: typing.Optional[float] = None,
    int_loss_at_layer: typing.Optional[int] = None,
    int_loss_scale: typing.Optional[float] = None,
    label_smoothing: float = 0.2,
    leave_cart_output: bool = False,
    target: str = "classes",
    upsample_by_transposed_conv: bool = True,
    feature_stacking_size: int = 3,
) -> attention_for_hybrid:
    conformer_net = get_cfg(
        num_classes=num_classes,
        size=size,
        chunking=chunking,
        focal_loss_factor=focal_loss_factor,
        int_loss_at_layer=int_loss_at_layer,
        int_loss_scale=int_loss_scale,
        label_smoothing=label_smoothing,
        target=target,
        time_tag_name=time_tag_name,
        upsample_by_transposed_conv=upsample_by_transposed_conv,
        feature_stacking_size=feature_stacking_size,
    )

    if not leave_cart_output:
        cart_out = conformer_net.network.pop("output")
        last_layer = cart_out["from"][0]

        conformer_net.network["encoder-output"] = {"class": "copy", "from": last_layer}

    return conformer_net
