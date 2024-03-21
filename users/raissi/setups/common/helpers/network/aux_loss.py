__all__ = ["add_intermediate_loss"]

import copy

import i6_core.returnn as returnn

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo, PhoneticContext

from i6_experiments.users.raissi.setups.common.helpers.network.augment import (
    augment_net_with_monophone_outputs,
    augment_net_with_diphone_outputs,
    augment_net_with_triphone_outputs,
    Network,
)

from i6_experiments.users.raissi.setups.common.helpers.network.frame_rate import FrameRateReductionRatioinfo


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
    )

    if context == PhoneticContext.monophone:
        pass  # already dealt with above
    elif context == PhoneticContext.diphone:
        network = augment_net_with_diphone_outputs(
            network,
            encoder_output_len=encoder_output_len,
            frame_rate_reduction_ratio_info=frame_rate_reduction_ratio_info,
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
            raise AttributeError(f"context type {context} not implemented as intermediate loss")

    return network
