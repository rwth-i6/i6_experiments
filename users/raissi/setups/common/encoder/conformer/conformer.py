__all__ = ["get_best_conformer_network"]

from typing import Any, Optional, Union


from i6_experiments.users.raissi.setups.common.encoder.conformer.best_setup import get_best_model_config, Size
from i6_experiments.users.raissi.setups.common.encoder.conformer.transformer_network import attention_for_hybrid
from i6_experiments.users.raissi.setups.common.helpers.train import returnn_time_tag
from i6_experiments.users.raissi.setups.common.encoder.conformer.layers import DEFAULT_INIT


def get_best_conformer_network(
    size: Union[Size, int],
    num_classes: int,
    num_input_feature: int,
    *,
    time_tag_name: Optional[str] = None,
    chunking: Optional[str] = None,
    int_loss_at_layer: Optional[int] = None,
    int_loss_scale: Optional[float] = None,
    label_smoothing: float = 0.0,
    leave_cart_output: bool = False,
    target: str = "classes",
    upsample_by_transposed_conv: bool = True,
    feature_stacking_size: int = 3,
    clipping: Optional[int] = None,
    weights_init: str = DEFAULT_INIT,
    additional_args: Optional[Any] = None,
) -> attention_for_hybrid:
    conformer_net = get_best_model_config(
        size=size,
        num_classes=num_classes,
        num_input_feature=num_input_feature,
        chunking=chunking,
        int_loss_at_layer=int_loss_at_layer,
        int_loss_scale=int_loss_scale,
        label_smoothing=label_smoothing,
        target=target,
        time_tag_name=time_tag_name,
        upsample_by_transposed_conv=upsample_by_transposed_conv,
        feature_stacking_size=feature_stacking_size,
        clipping=clipping,
        weights_init=weights_init,
        additional_args=additional_args,
    )

    if not leave_cart_output:
        cart_out = conformer_net.network.pop("output")
        last_layer = cart_out["from"][0]

        conformer_net.network["encoder-output"] = {"class": "copy", "from": last_layer}

    return conformer_net
