from typing import Any, Dict, List, Optional, Tuple
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
from returnn.tf.util.data import FeatureDim, batch_dim


def add_output(
    network: dict,
    from_list: str,
    name: str,
    target: str,
    num_outputs: int,
    scale: float = 1.0,
    reuse_from_name: Optional[str] = None,
    transposed_conv_args: dict = {},
    ff_args: dict = {},
    output_args: dict = {},
):
    transp_conv = add_transposed_conv(
        network,
        from_list=from_list,
        name=name,
        size_base=target,
        reuse_from_name=reuse_from_name,
        **transposed_conv_args,
    )
    mlp_out = add_feed_forward_stack(
        network,
        from_list=transp_conv,
        name=f"{name}_mlp",
        reuse_from_name=f"{reuse_from_name}_mlp" if reuse_from_name else None,
        **ff_args,
    )
    add_softmax_output(
        network,
        from_list=mlp_out,
        num_outputs=num_outputs,
        name=f"{name}_output",
        target=f"layer:{target}",
        scale=scale,
        reuse_from_name=f"{reuse_from_name}_output" if reuse_from_name else None,
        **output_args,
    )


def make_conformer_hybrid_dual_output_model(
    num_outputs: int,
    gt_args: Dict[str, Any],
    target_key: str = "data:target_classes",
    freeze_separator: bool = True,
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

    network = {}

    # *** Set some default parameters ***
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

    use_mixed_input = (
        conformer_mix_args.get("num_blocks", 1) > 0
        or conformer_01_mix_args.get("num_blocks", 1) > 0
    )

    # *** Sep features ***
    from_list, dim_tags = add_speech_separation(
        network, from_list="data", trainable=not freeze_separator
    )
    sep_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_sep", padding=(80, 0), **gt_args
    )
    network["gt_sep_merge"] = {
        "class": "merge_dims",
        "from": sep_features,
        "axes": ["B", dim_tags["speaker"]],
    }

    network["targets_merged"] = {
        "class": "merge_dims",
        "from": target_key,
        "axes": ["B", dim_tags["speaker"]],
    }
    for speaker_idx in [0, 1]:
        network[f"targets_{speaker_idx}"] = {
            "class": "slice",
            "from": target_key,
            "axis": dim_tags["speaker"],
            "slice_start": speaker_idx,
            "slice_end": speaker_idx + 1,
        }
        network[f"targets_{speaker_idx}_squeeze"] = {
            "class": "squeeze",
            "from": f"targets_{speaker_idx}",
            "axis": "auto",
        }

    # *** Sep encoder ***
    from_list = add_initial_conv(
        network,
        "vgg_conv_01",
        from_list="gt_sep_merge",
        **init_conv_args,
    )
    enc_01, blocks = add_conformer_stack(
        network,
        from_list=from_list,
        name="conformer_01",
        **conformer_01_args,
    )

    # *** Auxiliary loss ***
    for block, scale in aux_loss_01_blocks:
        network[f"conformer_01_block_{block}_split"] = {
            "class": "split_dims",
            "from": blocks[block-1],
            "axis": "B",
            "dims": (batch_dim, dim_tags["speaker"]),
        }
        for speaker_idx in [0, 1]:
            network[f"conformer_01_block_{block}_speaker_{speaker_idx}"] = {
                "class": "slice",
                "from": f"conformer_01_block_{block}_split",
                "axis": dim_tags["speaker"],
                "slice_start": speaker_idx,
                "slice_end": speaker_idx + 1,
            }
            network[f"conformer_01_block_{block}_speaker_{speaker_idx}_squeeze"] = {
                "class": "squeeze",
                "from": f"conformer_01_block_{block}_speaker_{speaker_idx}",
                "axis": "auto",
            }
            add_output(
                network=network,
                from_list=f"conformer_01_block_{block}_speaker_{speaker_idx}_squeeze",
                name=f"aux_loss_01_block_{block}_speaker_{speaker_idx}",
                target=f"targets_{speaker_idx}_squeeze",
                num_outputs=num_outputs,
                scale=scale,
                reuse_from_name=f"aux_loss_01_block_{block}_speaker_0" if speaker_idx == 1 else None,
                transposed_conv_args=transposed_conv_args,
                ff_args=ff_args,
                output_args=output_args,
            )

    # *** Mix + combination encoders ***
    if use_mixed_input:
        mix_features, _ = add_gt_feature_extraction(
            network, from_list="data", name="gt_mix", padding=(80, 0), **gt_args
        )
        from_list = add_initial_conv(
            network, "vgg_conv_mix", from_list=mix_features, **init_conv_args
        )

        enc_mix, _ = add_conformer_stack(
            network, from_list=from_list, name="conformer_mix", **conformer_mix_args
        )
        # network["encoder_01_split_dim"] = {
        #     "class": "split_dims",
        #     "from": enc_01,
        #     "axis": "B",
        #     "dims": (batch_dim, dim_tags["speaker"]),
        # }
        # for speaker_idx in [0, 1]:
        #     network[f"encoder_{speaker_idx}"] = {
        #         "class": "slice",
        #         "from": "encoder_01_split_dim",
        #         "axis": dim_tags["speaker"],
        #         "slice_start": speaker_idx,
        #         "slice_end": speaker_idx + 1,
        #     }
        #     network[f"encoder_{speaker_idx}_squeeze"] = {
        #         "class": "squeeze",
        #         "from": f"encoder_{speaker_idx}",
        #         "axis": "auto",
        #     }

        #     network[f"encoder_{speaker_idx}_mix_input"] = {
        #         "class": "combine",
        #         "from": [f"encoder_{speaker_idx}_squeeze", enc_mix],
        #         "kind": "add",
        #     }

        # dim_tags["enc_01_mix_input_feature"] = FeatureDim(
        #     "enc_01_mix_input_feature_dim", None
        # )
        # network["encoder_01_mix_input"] = {
        #     "class": "split_dims",
        #     "from": ["encoder_0_mix_input", "encoder_1_mix_input"],
        #     "axis": "F",
        #     "dims": (dim_tags["speaker"], dim_tags["enc_01_mix_input_feature"]),
        # }

        # network["encoder_01_mix_input_merged"] = {
        #     "class": "merge_dims",
        #     "from": "encoder_01_mix_input",
        #     "axes": ["B", dim_tags["speaker"]],
        # }
        network["encoder_01_mix_input"] = {
            "class": "combine",
            "from": [enc_01, enc_mix],
            "kind": "add",
        }

        enc_01, blocks = add_conformer_stack(
            network,
            from_list="encoder_01_mix_input",
            name="conformer_01_mix",
            **conformer_01_mix_args,
        )

        # *** Auxiliary loss ***
        for block, scale in aux_loss_01_mix_blocks:
            network[f"conformer_01_mix_block_{block}_split"] = {
                "class": "split_dims",
                "from": blocks[block-1],
                "axis": "B",
                "dims": (batch_dim, dim_tags["speaker"]),
            }
            for speaker_idx in [0, 1]:
                network[f"conformer_01_mix_block_{block}_speaker_{speaker_idx}"] = {
                    "class": "slice",
                    "from": f"conformer_01_mix_block_{block}_split",
                    "axis": dim_tags["speaker"],
                    "slice_start": speaker_idx,
                    "slice_end": speaker_idx + 1,
                }
                network[f"conformer_01_mix_block_{block}_speaker_{speaker_idx}_squeeze"] = {
                    "class": "squeeze",
                    "from": f"conformer_01_mix_block_{block}_speaker_{speaker_idx}",
                    "axis": "auto",
                }
                add_output(
                    network=network,
                    from_list=f"conformer_01_mix_block_{block}_speaker_{speaker_idx}_squeeze",
                    name=f"aux_loss_01_mix_block_{block}_speaker_{speaker_idx}",
                    target=f"targets_{speaker_idx}_squeeze",
                    num_outputs=num_outputs,
                    scale=scale,
                    reuse_from_name=f"aux_loss_01_mix_block_{block}_speaker_0" if speaker_idx == 1 else None,
                    transposed_conv_args=transposed_conv_args,
                    ff_args=ff_args,
                    output_args=output_args,
                )

    # *** Final loss ***
    network[f"conformer_output_split"] = {
        "class": "split_dims",
        "from": enc_01,
        "axis": "B",
        "dims": (batch_dim, dim_tags["speaker"]),
    }
    for speaker_idx in [0, 1]:
        network[f"conformer_output_speaker_{speaker_idx}"] = {
            "class": "slice",
            "from": "conformer_output_split",
            "axis": dim_tags["speaker"],
            "slice_start": speaker_idx,
            "slice_end": speaker_idx + 1,
        }
        network[f"conformer_output_speaker_{speaker_idx}_squeeze"] = {
            "class": "squeeze",
            "from": f"conformer_output_speaker_{speaker_idx}",
            "axis": "auto",
        }
        add_output(
            network=network,
            from_list=f"conformer_output_speaker_{speaker_idx}_squeeze",
            name=f"output_loss_speaker_{speaker_idx}",
            target=f"targets_{speaker_idx}_squeeze",
            num_outputs=num_outputs,
            scale=0.5,
            reuse_from_name="output_loss_speaker_0" if speaker_idx == 1 else None,
            transposed_conv_args=transposed_conv_args,
            ff_args=ff_args,
            output_args=output_args,
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

    if use_mixed_input:
        mix_features, python_code = add_gt_feature_extraction(
            network, from_list="data", name="gt_mix", **gt_args
        )

    from_list, dim_tags = add_speech_separation(network, from_list="data")

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

        network["encoder_01_mix_input"] = {
            "class": "combine",
            "from": [enc_01, enc_mix],
            "kind": "add",
        }

        enc_01, _ = add_conformer_stack(
            network,
            from_list="encoder_01_mix_input",
            name="conformer_01_mix",
            **conformer_01_mix_args,
        )

    transp_conv = add_transposed_conv(
        network,
        from_list=enc_01,
        name="output",
        size_base=None,
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
