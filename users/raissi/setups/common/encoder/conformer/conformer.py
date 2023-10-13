__all__ = ["get_best_conformer_network"]

from typing import Optional, Union


from i6_experiments.users.raissi.setups.common.encoder.conformer.best_setup import get_best_model_config, Size
from i6_experiments.users.raissi.setups.common.helpers.network.augment import Network
from i6_experiments.users.raissi.setups.common.helpers.train import returnn_time_tag


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
    additional_args: dict = None,
) -> Network:
    if time_tag_name is None:
        _, time_tag_name = returnn_time_tag.get_shared_time_tag()
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
        additional_args=additional_args,
    )

    if not leave_cart_output:
        conformer_net.network.pop("output", None)
        conformer_net.network["encoder-output"] = {
            "class": "copy",
            "from": "length_masked",
        }

    return conformer_net
