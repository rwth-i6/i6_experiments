from typing import Any, Dict, List, Tuple
from i6_experiments.users.berger.network.helpers.feature_extraction import (
    add_gt_feature_extraction,
)
from i6_experiments.users.berger.network.helpers.speech_separation import (
    add_speech_separation,
)
from i6_experiments.users.berger.network.helpers.conformer import (
    get_variance_scaling_init,
    add_conformer_stack,
    add_initial_conv,
    add_transposed_conv,
)
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.output import add_softmax_output


def make_conformer_hybrid_dual_output_model(
    num_outputs: int,
    gt_args: Dict[str, Any],
    target_key: str = "data:target_rasr",
    use_mixed_input: bool = True,
    init_conv_args: Dict = {},
    conformer_01_args: Dict = {},
    conformer_mix_args: Dict = {},
    conformer_01_mix_args: Dict = {},
    transposed_conv_args: Dict = {},
    aux_loss_01_blocks: List[Tuple[int, float]] = [(4, 0.3), (8, 0.3)],
    aux_loss_01_mix_blocks: List[Tuple[int, float]] = [(4, 0.3), (8, 0.3)],
    ff_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, str]:

    for key, val in [
        ("num_layers", 1),
        ("size", 384),
        ("activation", None),
        ("dropout", None),
        ("l2", None),
        ("initializer", get_variance_scaling_init()),
    ]:
        ff_args.setdefault(key, val)

    output_args.setdefault("initializer", get_variance_scaling_init())

    network = {}
    python_code = []

    from_list = "data"

    if use_mixed_input:
        mix_features, python_code = add_gt_feature_extraction(
            network, from_list=from_list, name="gt_mix", **gt_args
        )

    from_list, dim_tags = add_speech_separation(network, from_list=from_list)
    sep_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_sep", **gt_args
    )

    network["targets_merged"] = {
        "class": "merge_dims",
        "from": target_key,
        "axes": ["B", dim_tags["speaker"]],
    }

    from_list = add_initial_conv(
        network, "vgg_conv_01", from_list=sep_features, **init_conv_args
    )

    enc_01, blocks = add_conformer_stack(
        network, from_list=from_list, name="conformer_01", **conformer_01_args
    )

    for block, scale in aux_loss_01_blocks:
        transp_conv = add_transposed_conv(
            network,
            from_list=blocks[block - 1],
            name=f"aux_01_{block}",
            **transposed_conv_args,
        )
        mlp_out = add_feed_forward_stack(
            network,
            from_list=transp_conv,
            name=f"aux_01_{block}_mlp",
            **ff_args,
        )
        network[f"aux_01_{block}_mlp_merge"] = {
            "class": "merge_dims",
            "from": mlp_out,
            "axes": ["B", dim_tags["speaker"]],
        }
        add_softmax_output(
            network,
            from_list=f"aux_01_{block}_mlp_merge",
            num_outputs=num_outputs,
            name=f"aux_01_output_{block}",
            target="layer:targets_merged",
            scale=scale,
            **output_args,
        )

    if use_mixed_input:
        from_list = add_initial_conv(
            network, "vgg_conv_mix", from_list=mix_features, **init_conv_args
        )

        enc_mix, _ = add_conformer_stack(
            network, from_list=from_list, name="conformer_mix", **conformer_mix_args
        )

        for speaker_idx in [0, 1]:
            network[f"encoder_{speaker_idx}"] = {
                "class": "slice",
                "from": enc_01,
                "axis": dim_tags["speaker"],
                "slice_start": speaker_idx,
                "slice_end": speaker_idx + 1,
            }

            network[f"encoder_{speaker_idx}+mix_input"] = {
                "class": "copy",
                "from": [f"encoder_{speaker_idx}", enc_mix],
            }

        network["encoder_01+mix_input"] = {
            "class": "split_dims",
            "from": ["encoder_0+mix_input", "encoder_1+mix_input"],
            "axis": "F",
            "dims": (dim_tags["speaker"], -1),
        }

        enc_01, blocks = add_conformer_stack(
            network,
            from_list="encoder_01+mix_input",
            name="conformer_01+mix",
            **conformer_01_mix_args,
        )

        for block, scale in aux_loss_01_mix_blocks:
            transp_conv = add_transposed_conv(
                network,
                from_list=blocks[block - 1],
                name=f"aux_01+mix_{block}",
                **transposed_conv_args,
            )
            mlp_out = add_feed_forward_stack(
                network,
                from_list=transp_conv,
                name=f"aux_01+mix_{block}_mlp",
                **ff_args,
            )
            network[f"aux_01+mix_{block}_mlp_merge"] = {
                "class": "merge_dims",
                "from": mlp_out,
                "axes": ["B", dim_tags["speaker"]],
            }
            add_softmax_output(
                network,
                from_list=f"aux_01+mix_{block}_mlp_merge",
                num_outputs=num_outputs,
                name=f"aux_01+mix_output_{block}",
                target="layer:targets_merged",
                scale=scale,
                **output_args,
            )

    transp_conv = add_transposed_conv(
        network,
        from_list=enc_01,
        name="output",
        **transposed_conv_args,
    )
    mlp_out = add_feed_forward_stack(
        network,
        from_list=transp_conv,
        name="output_mlp",
        **ff_args,
    )
    network["output_mlp_merge"] = {
        "class": "merge_dims",
        "from": mlp_out,
        "axes": ["B", dim_tags["speaker"]],
    }
    add_softmax_output(
        network,
        from_list="output_mlp_merge",
        target="layer:targets_merged",
        num_outputs=num_outputs,
        **output_args,
    )

    return network, python_code, dim_tags


def make_conformer_hybrid_dual_output_recog_model(
    num_outputs: int,
    speaker_idx: int,
    gt_args: Dict[str, Any],
    use_mixed_input: bool = True,
    init_conv_args: Dict = {},
    conformer_01_args: Dict = {},
    conformer_mix_args: Dict = {},
    conformer_01_mix_args: Dict = {},
    transposed_conv_args: Dict = {},
    ff_args: Dict = {},
) -> Tuple[Dict, str]:

    for key, val in [
        ("num_layers", 1),
        ("size", 384),
        ("activation", None),
    ]:
        ff_args.setdefault(key, val)

    network = {}
    python_code = []

    from_list = "data"

    if use_mixed_input:
        mix_features, python_code = add_gt_feature_extraction(
            network, from_list=from_list, name="gt_mix", **gt_args
        )

    from_list, dim_tags = add_speech_separation(network, from_list=from_list)

    network["speech_separator"]["subnetwork"]["separated_stft_pit"]["subnetwork"][
        "output"
    ]["position"] = speaker_idx

    sep_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_sep", **gt_args
    )

    network["squeeze_sep_features"] = {
        "class": "squeeze",
        "from": sep_features,
        "axis": "auto",
    }

    from_list = add_initial_conv(
        network, "vgg_conv_01", from_list="squeeze_sep_features", **init_conv_args
    )

    enc_01, _ = add_conformer_stack(
        network, from_list=from_list, name="conformer_01", **conformer_01_args
    )

    if use_mixed_input:
        from_list = add_initial_conv(
            network, "vgg_conv_mix", from_list=mix_features, **init_conv_args
        )

        enc_mix, _ = add_conformer_stack(
            network, from_list=from_list, name="conformer_mix", **conformer_mix_args
        )

        network["encoder_01+mix_input"] = {
            "class": "copy",
            "from": [enc_01, enc_mix],
        }

        enc_01, _ = add_conformer_stack(
            network,
            from_list="encoder_01+mix_input",
            name="conformer_01+mix",
            **conformer_01_mix_args,
        )

    transp_conv = add_transposed_conv(
        network,
        from_list=enc_01,
        name="output",
        **transposed_conv_args,
    )
    mlp_out = add_feed_forward_stack(
        network,
        from_list=transp_conv,
        name="output_mlp",
        **ff_args,
    )
    network["output"] = {
        "class": "linear",
        "from": mlp_out,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, python_code, dim_tags
