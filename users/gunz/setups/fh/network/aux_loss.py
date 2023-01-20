__all__ = ["add_intermediate_loss"]

import copy

import i6_core.returnn as returnn

from ..factored import PhoneticContext, LabelInfo
from .augment import (
    augment_net_with_monophone_outputs,
    augment_net_with_diphone_outputs,
    augment_net_with_triphone_outputs,
    Network,
)


def add_intermediate_loss(
    network: Network,
    encoder_output_len: int,
    label_info: LabelInfo,
    context: PhoneticContext,
    time_tag_name: str,
    *,
    at_layer: int = 6,
    scale: float = 0.5,
    center_state_only: bool = False,
    final_ctx_type: PhoneticContext = PhoneticContext.triphone_forward,
    focal_loss_factor: float = 2.0,
    label_smoothing: float = 0.2,
    l2: float = 0.0,
) -> Network:
    assert (
        f"aux_{at_layer}_ff1" in network
    ), "network needs to be built w/ CART intermediate loss to add FH intermediate loss (upsampling)"

    network = copy.deepcopy(network)

    network.pop(f"aux_{at_layer}_ff1", None)
    network.pop(f"aux_{at_layer}_ff2", None)
    network.pop(f"aux_{at_layer}_output_prob", None)
    aux_length = network.pop(f"aux_{at_layer}_length_masked")

    input_layer = f"aux_{at_layer:03d}_length_masked"
    prefix = f"aux_{at_layer:03d}_"

    network[input_layer] = {
        "class": "slice_nd",
        "from": aux_length["from"],
        "start": 0,
        "size": returnn.CodeWrapper(time_tag_name),
        "axis": "T",
    }

    network = augment_net_with_monophone_outputs(
        network,
        label_info=label_info,
        add_mlps=True,
        final_ctx_type=final_ctx_type,
        encoder_output_len=encoder_output_len,
        encoder_output_layer=input_layer,
        focal_loss_factor=focal_loss_factor,
        label_smoothing=label_smoothing,
        l2=l2,
        use_multi_task=True,
        prefix=prefix,
        loss_scale=scale,
    )

    if context == PhoneticContext.monophone:
        pass  # already dealt with above
    elif context == PhoneticContext.diphone:
        network = augment_net_with_diphone_outputs(
            network,
            encoder_output_len=encoder_output_len,
            label_smoothing=label_smoothing,
            l2=l2,
            ph_emb_size=label_info.ph_emb_size,
            st_emb_size=label_info.st_emb_size,
            use_multi_task=True,
            encoder_output_layer=input_layer,
            prefix=prefix,
        )
    elif context == PhoneticContext.triphone_forward:
        network = augment_net_with_triphone_outputs(
            network,
            l2=l2,
            ph_emb_size=label_info.ph_emb_size,
            st_emb_size=label_info.st_emb_size,
            variant=PhoneticContext.triphone_forward,
            encoder_output_layer=input_layer,
            prefix=prefix,
        )
    else:
        raise AttributeError(
            f"context type {context} not implemented as intermediate loss"
        )

    if center_state_only:
        network.pop(f"{prefix}left-output", None)
        network.pop(f"{prefix}right-output", None)

    return network
