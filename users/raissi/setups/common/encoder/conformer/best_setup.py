__all__ = ["INT_LOSS_LAYER", "INT_LOSS_SCALE", "get_best_model_config", "Size"]


from enum import Enum
from typing import Union, Optional, Tuple
from i6_experiments.users.raissi.setups.common.encoder.conformer.get_network_args import (
    get_encoder_args,
    get_network_args,
)

from i6_experiments.users.raissi.setups.common.encoder.conformer.layers import DEFAULT_INIT
from i6_experiments.users.raissi.setups.common.encoder.conformer.transformer_network import attention_for_hybrid

INT_LOSS_LAYER = 6
INT_LOSS_SCALE = 0.5


class Size(Enum):
    S = 256
    M = 384
    L = 512

    def size(self) -> int:
        return self.value


def get_best_model_config(
    size: Union[Size, int],
    num_classes: int,
    num_input_feature: int,
    *,
    chunking: [str, Tuple] = None,
    int_loss_at_layer: Optional[int] = None,
    int_loss_scale: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    target: str = "classes",
    time_tag_name: Optional[str] = None,
    upsample_by_transposed_conv: bool = True,
    feature_stacking_size: int = 3,
    clipping: Optional[int] = None,
    weights_init: str = DEFAULT_INIT,
    additional_args: Optional[dict] = None,
) -> attention_for_hybrid:
    if int_loss_at_layer is None:
        int_loss_at_layer = INT_LOSS_LAYER
    if int_loss_scale is None:
        int_loss_scale = INT_LOSS_SCALE
    if label_smoothing is None:
        label_smoothing = 0.0

    att_dim = 64
    model_dim = size if isinstance(size, int) else size.size()

    assert model_dim % att_dim == 0, "model_dim must be divisible by number of att heads"

    if isinstance(chunking, tuple):
        [clip, overlap] = [ele["data"] for ele in chunking]
    else:
        clip, overlap = [int(v) for v in chunking.split(":")] if chunking is not None else (400, 200)
    if clipping is not None:
        clip = clipping

    enc_args = get_encoder_args(
        model_dim // att_dim,
        att_dim,
        att_dim,
        model_dim,
        model_dim * 4,
        32,
        0.1,
        0.0,
        clipping=clip,
        layer_norm_instead_of_batch_norm=True,
        relative_pe=True,
        initialization=weights_init,
    )

    args = {
        "add_blstm_block": False,
        "add_conv_block": True,
        "loss_layer_idx": int_loss_at_layer,
        "loss_scale": int_loss_scale,
        "feature_stacking": True,
        "feature_stacking_window": [feature_stacking_size - 1, 0],
        "feature_stacking_stride": feature_stacking_size,
        "transposed_conv": upsample_by_transposed_conv,
        "transposed_conv_args": {
            "time_tag_name": time_tag_name,
        },
    }

    if additional_args is not None:
        args.update(**additional_args)

    configured_args = get_network_args(
        num_enc_layers=12,
        type="conformer",
        enc_args=enc_args,
        target=target,
        num_classes=num_classes,
        num_input_feature=num_input_feature,
        label_smoothing=label_smoothing,
        **args,
    )

    conformer = attention_for_hybrid(**configured_args)
    conformer.get_network()

    return conformer
