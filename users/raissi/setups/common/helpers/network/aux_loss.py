__all__ = ["add_intermediate_loss"]

import copy
from typing import List, Tuple

import i6_core.returnn as returnn

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo, PhoneticContext

from i6_experiments.users.raissi.setups.common.helpers.network.augment import (
    augment_net_with_monophone_outputs,
    augment_net_with_diphone_outputs,
    augment_net_with_triphone_outputs,
    Network,
)

from i6_experiments.users.raissi.setups.common.helpers.network.frame_rate import FrameRateReductionRatioinfo

DEFAULT_INIT = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)"

FH_LOSS_VARIANTS_MONO = [
    ([(6, PhoneticContext.monophone, True)], True),
    ([(6, PhoneticContext.monophone, False)], True),
    ([(6, PhoneticContext.triphone_forward, False)], False),
    ([(4, PhoneticContext.monophone, True), (8, PhoneticContext.monophone, True)], True),
    (
        [
            (3, PhoneticContext.monophone, True),
            (6, PhoneticContext.diphone, False),
            (9, PhoneticContext.triphone_forward, False),
        ],
        True,
    ),
    (
        [
            (3, PhoneticContext.triphone_forward, False),
            (6, PhoneticContext.diphone, False),
            (9, PhoneticContext.monophone, True),
        ],
        False,
    ),
]
FH_LOSS_VARIANTS_DI = [
    ([(6, PhoneticContext.monophone, False)], False),
    ([(6, PhoneticContext.diphone, False)], False),
    ([(4, PhoneticContext.monophone, True), (8, PhoneticContext.monophone, True)], False),
    ([(4, PhoneticContext.diphone, False), (8, PhoneticContext.diphone, False)], True),
    (
        [
            (3, PhoneticContext.monophone, True),
            (6, PhoneticContext.diphone, False),
            (9, PhoneticContext.triphone_forward, False),
        ],
        True,
    ),
    (
        [
            (3, PhoneticContext.triphone_forward, False),
            (6, PhoneticContext.diphone, False),
            (9, PhoneticContext.monophone, True),
        ],
        False,
    ),
    ([(6, PhoneticContext.triphone_forward, False)], False),
    (
        [
            (4, PhoneticContext.triphone_forward, False),
            (8, PhoneticContext.triphone_forward, False),
        ],
        False,
    ),
]
FH_LOSS_VARIANTS_DI_MULTISTAGE = [
    ([(6, PhoneticContext.diphone, False)], False, "6mono"),
    (
        [(4, PhoneticContext.diphone, False), (8, PhoneticContext.diphone, False)],
        False,
        "6mono_c",
    ),
]
FH_LOSS_VARIANTS_TRI = [
    ([(6, PhoneticContext.monophone, False)], True),
    ([(6, PhoneticContext.triphone_forward, False)], True),
    ([(4, PhoneticContext.monophone, True), (8, PhoneticContext.monophone, True)], False),
    (
        [
            (4, PhoneticContext.triphone_forward, False),
            (8, PhoneticContext.triphone_forward, False),
        ],
        False,
    ),
    (
        [
            (3, PhoneticContext.monophone, True),
            (6, PhoneticContext.diphone, False),
            (9, PhoneticContext.triphone_forward, False),
        ],
        False,
    ),
    (
        [
            (3, PhoneticContext.triphone_forward, False),
            (6, PhoneticContext.diphone, False),
            (9, PhoneticContext.monophone, True),
        ],
        True,
    ),
]
FH_LOSS_VARIANTS_TRI_MULTISTAGE = [
    ([(6, PhoneticContext.monophone, False)], False, "6mono"),
    (
        [
            (3, PhoneticContext.monophone, True),
            (6, PhoneticContext.diphone, False),
            (9, PhoneticContext.triphone_forward, False),
        ],
        False,
        "6mono",
    ),
    (
        [
            (3, PhoneticContext.triphone_forward, False),
            (6, PhoneticContext.diphone, False),
            (9, PhoneticContext.monophone, True),
        ],
        False,
        "6mono_c",
    ),
]


def get_int_loss_scale(
    num_final_losses: int,
    aux_losses: List[Tuple[int, PhoneticContext, bool]],
    final_loss_share: float = 0.6,
    final_loss_scale: float = 1.0,
) -> float:
    total_num_int_losses = float(sum([1 if center_only else 3 for _, _, center_only in aux_losses]))
    total_num_final_losses = float(num_final_losses)

    total_loss_weight = (1.0 / final_loss_share) * (total_num_final_losses * final_loss_scale)
    remaining_int_loss_weight = total_loss_weight - (total_num_final_losses * final_loss_scale)

    return remaining_int_loss_weight / total_num_int_losses if total_num_int_losses > 0 else remaining_int_loss_weight


def get_loss_variant_and_scale(
    int_loss_index, variants=FH_LOSS_VARIANTS_MONO, num_final_losses=3, final_loss_scale=1.0
):
    (opt_loss_variant, _) = variants[int_loss_index]
    int_loss_scale = get_int_loss_scale(
        num_final_losses=num_final_losses, aux_losses=opt_loss_variant, final_loss_scale=final_loss_scale
    )
    return opt_loss_variant, int_loss_scale


def add_intermediate_loss(
    network: Network,
    encoder_output_len: int,
    label_info: LabelInfo,
    context: PhoneticContext,
    time_tag_name: str,
    frame_rate_reduction_ratio_info: FrameRateReductionRatioinfo,
    *,
    at_layer: int = 6,
    scale: float = 0.5,
    center_state_only: bool = False,
    final_ctx_type: PhoneticContext = PhoneticContext.triphone_forward,
    focal_loss_factor: float = 2.0,
    label_smoothing: float = 0.0,
    l2: float = 0.0,
    upsampling: bool = True,
    weights_init: str = DEFAULT_INIT,
) -> Network:
    network = copy.deepcopy(network)
    prefix = f"aux_{at_layer:03d}_"

    old_ff1_layer = network.pop(f"aux_{at_layer}_ff1", None)
    network.pop(f"aux_{at_layer}_ff2", None)
    network.pop(f"aux_{at_layer}_output_prob", None)

    if upsampling:
        assert (
            old_ff1_layer is not None
        ), "network needs to be built w/ CART intermediate loss to add FH intermediate loss (upsampling)"

        aux_length = network.pop(f"aux_{at_layer}_length_masked")
        input_layer = f"aux_{at_layer:03d}_length_masked"
        network[input_layer] = {
            "class": "slice_nd",
            "from": aux_length["from"],
            "start": 0,
            "size": returnn.CodeWrapper(time_tag_name),
            "axis": "T",
        }
    else:
        input_layer = f"enc_{at_layer:03d}"

    network = augment_net_with_monophone_outputs(
        network,
        label_info=label_info,
        add_mlps=True,
        final_ctx_type=final_ctx_type,
        frame_rate_reduction_ratio_info=frame_rate_reduction_ratio_info,
        encoder_output_len=encoder_output_len,
        encoder_output_layer=input_layer,
        focal_loss_factor=focal_loss_factor,
        label_smoothing=label_smoothing,
        l2=l2,
        use_multi_task=True,
        prefix=prefix,
        loss_scale=scale,
        weights_init=weights_init,
    )

    if context == PhoneticContext.monophone:
        pass  # already dealt with above
    elif context == PhoneticContext.diphone:
        network = augment_net_with_diphone_outputs(
            network,
            encoder_output_len=encoder_output_len,
            label_info=label_info,
            frame_rate_reduction_ratio_info=frame_rate_reduction_ratio_info,
            label_smoothing=label_smoothing,
            l2=l2,
            use_multi_task=True,
            encoder_output_layer=input_layer,
            prefix=prefix,
            weights_init=weights_init,
        )
    elif context == PhoneticContext.triphone_forward:
        network = augment_net_with_triphone_outputs(
            network,
            frame_rate_reduction_ratio_info=frame_rate_reduction_ratio_info,
            label_info=label_info,
            l2=l2,
            variant=PhoneticContext.triphone_forward,
            encoder_output_layer=input_layer,
            prefix=prefix,
        )
    else:
        raise AttributeError(f"context type {context} not implemented as intermediate loss")

    if center_state_only:
        assert "aux" in prefix, "Are you deleting the main outputs?"
        network.pop(f"{prefix}left-output", None)
        network.pop(f"{prefix}right-output", None)

    return network


def add_intermediate_loss_v2(
    network: Network,
    encoder_output_len: int,
    label_info: LabelInfo,
    context: PhoneticContext,
    time_tag_name: str,
    frame_rate_reduction_ratio_info: FrameRateReductionRatioinfo,
    *,
    use_multi_task: bool = True,
    add_mlps: bool = True,
    center_state_only: bool = False,
    at_layer: list = [6],
    scale: float = 0.5,
    final_ctx_type: PhoneticContext = PhoneticContext.triphone_forward,
    focal_loss_factor: float = 2.0,
    label_smoothing: float = 0.0,
    l2: float = 0.0,
    upsampling: bool = True,
) -> Network:
    network = copy.deepcopy(network)

    for aux_l in at_layer:
        prefix = f"aux_{aux_l:03d}_"

        old_ff1_layer = network.pop(f"aux_{aux_l}_ff1", None)
        network.pop(f"aux_{aux_l}_ff2", None)
        network.pop(f"aux_{aux_l}_output_prob", None)

        if upsampling:
            assert (
                old_ff1_layer is not None
            ), "network needs to be built w/ CART intermediate loss to add FH intermediate loss (upsampling)"

            aux_length = network.pop(f"aux_{aux_l}_length_masked")
            input_layer = f"aux_{aux_l:03d}_length_masked"
            network[input_layer] = {
                "class": "slice_nd",
                "from": aux_length["from"],
                "start": 0,
                "size": returnn.CodeWrapper(time_tag_name),
                "axis": "T",
            }
        else:
            input_layer = f"enc_{aux_l:03d}"

        network = augment_net_with_monophone_outputs(
            network,
            label_info=label_info,
            add_mlps=add_mlps,
            final_ctx_type=final_ctx_type,
            frame_rate_reduction_ratio_info=frame_rate_reduction_ratio_info,
            encoder_output_len=encoder_output_len,
            encoder_output_layer=f"{prefix}encoder",
            focal_loss_factor=focal_loss_factor,
            label_smoothing=label_smoothing,
            l2=l2,
            use_multi_task=use_multi_task,
            prefix=prefix,
            loss_scale=scale,
        )

        if context == PhoneticContext.monophone or center_state_only:
            continue
        elif context == PhoneticContext.diphone:
            network = augment_net_with_diphone_outputs(
                network,
                encoder_output_len=encoder_output_len,
                frame_rate_reduction_ratio_info=frame_rate_reduction_ratio_info,
                label_info=label_info,
                label_smoothing=label_smoothing,
                l2=l2,
                use_multi_task=True,
                encoder_output_layer=input_layer,
                prefix=prefix,
            )
        elif context == PhoneticContext.triphone_forward:
            network = augment_net_with_triphone_outputs(
                network,
                label_info=label_info,
                l2=l2,
                variant=PhoneticContext.triphone_forward,
                encoder_output_layer=input_layer,
                prefix=prefix,
            )
        else:
            raise AttributeError(f"context type {context} not implemented as intermediate loss")

    return network
