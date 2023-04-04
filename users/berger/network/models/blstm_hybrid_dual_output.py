from typing import Any, Dict, List, Tuple, Union
from i6_core.returnn.config import CodeWrapper
from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack
from i6_experiments.users.berger.network.helpers.feature_extraction import (
    add_gt_feature_extraction,
)
from i6_experiments.users.berger.network.helpers.speech_separation import (
    add_speech_separation,
)
from returnn.tf.util.data import Dim
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from returnn.tf.util.data import FeatureDim, batch_dim


def make_blstm_hybrid_dual_output_model(
    num_outputs: int,
    gt_args: Dict[str, Any],
    target_key: str = "data:target_classes",
    freeze_separator: bool = True,
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
    aux_loss_01_layers: List[Tuple[int, float]] = [(3, 0.3)],
    aux_loss_01_mix_layers: List[Tuple[int, float]] = [(3, 0.3)],
    output_args: Dict = {},
) -> Tuple[Dict, Union[List, str], Dict[str, Dim]]:
    network = {}
    python_code = []

    from_list, dim_tags = add_speech_separation(
        network, from_list="data", trainable=not freeze_separator
    )
    sep_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_sep", padding=(80, 0), **gt_args
    )

    use_mixed_input = (
        blstm_mix_args.get("num_layers", 1) > 0
        or blstm_01_mix_args.get("num_layers", 1) > 0
    )
    if use_mixed_input:
        mix_features, python_code = add_gt_feature_extraction(
            network, from_list="data", name="gt_mix", padding=(80, 0), **gt_args
        )

    network["targets_merged"] = {
        "class": "merge_dims",
        "from": target_key,
        "axes": ["B", dim_tags["speaker"]],
    }

    enc_01, layers = add_blstm_stack(
        network, from_list=sep_features, name="lstm_01", **blstm_01_args
    )

    for layer, scale in aux_loss_01_layers:
        network[f"aux_01_{layer}_merge"] = {
            "class": "merge_dims",
            "from": layers[layer - 1],
            "axes": ["B", dim_tags["speaker"]],
        }
        add_softmax_output(
            network,
            from_list=f"aux_01_{layer}_merge",
            num_outputs=num_outputs,
            name=f"aux_01_output_{layer}",
            target="layer:targets_merged",
            scale=scale,
            **output_args,
        )

    if use_mixed_input:

        enc_mix, _ = add_blstm_stack(
            network, from_list=mix_features, name="lstm_mix", **blstm_mix_args
        )

        network["encoder_mix"] = {
            "class": "copy",
            "from": enc_mix,
        }
        enc_mix = "encoder_mix"

        for speaker_idx in [0, 1]:
            network[f"encoder_{speaker_idx}"] = {
                "class": "slice",
                "from": enc_01,
                "axis": dim_tags["speaker"],
                "slice_start": speaker_idx,
                "slice_end": speaker_idx + 1,
            }
            network[f"encoder_{speaker_idx}_squeeze"] = {
                "class": "squeeze",
                "from": f"encoder_{speaker_idx}",
                "axis": "auto",
            }

            if blstm_01_args.get("num_layers", 1) and blstm_mix_args.get(
                "num_layers", 1
            ):
                network[f"encoder_{speaker_idx}+mix_input"] = {
                    "class": "combine",
                    "from": [f"encoder_{speaker_idx}_squeeze", enc_mix],
                    "kind": "add",
                }
            else:
                network[f"encoder_{speaker_idx}+mix_input"] = {
                    "class": "copy",
                    "from": [f"encoder_{speaker_idx}_squeeze", enc_mix],
                }

        dim_tags["enc_01_mix_input_feature"] = FeatureDim(
            "enc_01_mix_input_feature_dim", None
        )
        network["encoder_01+mix_input"] = {
            "class": "split_dims",
            "from": ["encoder_0+mix_input", "encoder_1+mix_input"],
            "axis": "F",
            "dims": (dim_tags["speaker"], dim_tags["enc_01_mix_input_feature"]),
        }

        enc_01, layers = add_blstm_stack(
            network,
            from_list="encoder_01+mix_input",
            name="lstm_01+mix",
            **blstm_01_mix_args,
        )

        for layer, scale in aux_loss_01_mix_layers:
            network[f"aux_01+mix_{layer}_merge"] = {
                "class": "merge_dims",
                "from": layers[layer - 1],
                "axes": ["B", dim_tags["speaker"]],
            }
            add_softmax_output(
                network,
                from_list=f"aux_01+mix_{layer}_merge",
                num_outputs=num_outputs,
                name=f"aux_01+mix_output_{layer}",
                target="layer:targets_merged",
                scale=scale,
                **output_args,
            )

    network["output_merge"] = {
        "class": "merge_dims",
        "from": enc_01,
        "axes": ["B", dim_tags["speaker"]],
    }
    add_softmax_output(
        network,
        from_list="output_merge",
        target="layer:targets_merged",
        num_outputs=num_outputs,
        **output_args,
    )

    return network, python_code, dim_tags


def make_blstm_hybrid_dual_output_recog_model(
    num_outputs: int,
    speaker_idx: int,
    gt_args: Dict[str, Any],
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
) -> Tuple[Dict, Union[str, List], Dict[str, Dim]]:

    network = {}
    python_code = []

    from_list = "data"

    use_mixed_input = (
        blstm_mix_args.get("num_layers", 1) > 0
        or blstm_01_mix_args.get("num_layers", 1) > 0
    )
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

    enc_01, _ = add_blstm_stack(
        network, from_list="squeeze_sep_features", name="lstm_01", **blstm_01_args
    )

    if use_mixed_input:
        enc_mix, _ = add_blstm_stack(
            network, from_list=mix_features, name="lstm_mix", **blstm_mix_args
        )
        network["encoder_01"] = {
            "class": "copy",
            "from": enc_01,
        }
        enc_01 = "encoder_01"

        network["encoder_mix"] = {
            "class": "copy",
            "from": enc_mix,
        }
        enc_mix = "encoder_mix"

        if blstm_01_args.get("num_layers", 1) and blstm_mix_args.get("num_layers", 1):
            network[f"encoder_01+mix_input"] = {
                "class": "combine",
                "from": [enc_01, enc_mix],
                "kind": "add",
            }
        else:
            network[f"encoder_01+mix_input"] = {
                "class": "copy",
                "from": [enc_01, enc_mix],
            }

        enc_01, _ = add_blstm_stack(
            network,
            from_list="encoder_01+mix_input",
            name="lstm_01+mix",
            **blstm_01_mix_args,
        )

    network["output"] = {
        "class": "linear",
        "from": enc_01,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, python_code, dim_tags


def make_blstm_hybrid_dual_output_combine_enc_model(
    num_outputs: int,
    gt_args: Dict[str, Any],
    target_key: str = "data:target_classes",
    freeze_separator: bool = True,
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
    blstm_combine_args: Dict = {},
    aux_loss_01_layers: List[Tuple[int, float]] = [(3, 0.3)],
    aux_loss_01_mix_layers: List[Tuple[int, float]] = [(3, 0.3)],
    output_args: Dict = {},
) -> Tuple[Dict, Union[str, List], Dict[str, Dim]]:
    network = {}
    python_code = []

    from_list, dim_tags = add_speech_separation(
        network, from_list="data", trainable=not freeze_separator
    )
    sep_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_sep", padding=(80, 0), **gt_args
    )

    use_mixed_input = (
        blstm_mix_args.get("num_layers", 1) > 0
        or blstm_combine_args.get("num_layers", 1) > 0
    )
    assert use_mixed_input

    mix_features, python_code = add_gt_feature_extraction(
        network, from_list="data", name="gt_mix", padding=(80, 0), **gt_args
    )

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

    enc_01, layers = add_blstm_stack(
        network, from_list=sep_features, name="lstm_01", **blstm_01_args
    )

    for layer, scale in aux_loss_01_layers:
        network[f"aux_01_{layer}_merge"] = {
            "class": "merge_dims",
            "from": layers[layer - 1],
            "axes": ["B", dim_tags["speaker"]],
        }
        add_softmax_output(
            network,
            from_list=f"aux_01_{layer}_merge",
            num_outputs=num_outputs,
            name=f"aux_01_output_{layer}",
            target="layer:targets_merged",
            scale=scale,
            **output_args,
        )

    enc_mix, _ = add_blstm_stack(
        network, from_list=mix_features, name="lstm_mix", **blstm_mix_args
    )

    network["encoder_mix"] = {
        "class": "copy",
        "from": enc_mix,
    }
    enc_mix = "encoder_mix"

    for speaker_idx in [0, 1]:
        network[f"encoder_{speaker_idx}"] = {
            "class": "slice",
            "from": enc_01,
            "axis": dim_tags["speaker"],
            "slice_start": speaker_idx,
            "slice_end": speaker_idx + 1,
        }
        network[f"encoder_{speaker_idx}_squeeze"] = {
            "class": "squeeze",
            "from": f"encoder_{speaker_idx}",
            "axis": "auto",
        }

    if blstm_01_mix_args.get("num_layers", 0):
        for speaker_idx in [0, 1]:
            network[f"encoder_{speaker_idx}+mix_input"] = {
                "class": "combine",
                "from": [f"encoder_{speaker_idx}_squeeze", enc_mix],
                "kind": "add",
            }

        dim_tags["enc_01_mix_input_feature"] = FeatureDim(
            "enc_01_mix_input_feature_dim", None
        )
        network["encoder_01+mix_input"] = {
            "class": "split_dims",
            "from": ["encoder_0+mix_input", "encoder_1+mix_input"],
            "axis": "F",
            "dims": (dim_tags["speaker"], dim_tags["enc_01_mix_input_feature"]),
        }

        enc_01, layers = add_blstm_stack(
            network,
            from_list="encoder_01+mix_input",
            name="lstm_01+mix",
            **blstm_01_mix_args,
        )

        for layer, scale in aux_loss_01_mix_layers:
            network[f"aux_01+mix_{layer}_merge"] = {
                "class": "merge_dims",
                "from": layers[layer - 1],
                "axes": ["B", dim_tags["speaker"]],
            }
            add_softmax_output(
                network,
                from_list=f"aux_01+mix_{layer}_merge",
                num_outputs=num_outputs,
                name=f"aux_01+mix_output_{layer}",
                target="layer:targets_merged",
                scale=scale,
                **output_args,
            )
        for speaker_idx in [0, 1]:
            network[f"encoder_{speaker_idx}_mix"] = {
                "class": "slice",
                "from": enc_01,
                "axis": dim_tags["speaker"],
                "slice_start": speaker_idx,
                "slice_end": speaker_idx + 1,
            }
            network[f"encoder_{speaker_idx}_mix_squeeze"] = {
                "class": "squeeze",
                "from": f"encoder_{speaker_idx}_mix",
                "axis": "auto",
            }
        network[f"encoder_combine_input"] = {
            "class": "copy",
            "from": ["encoder_0_mix_squeeze", "encoder_1_mix_squeeze"],
        }
    else:
        network[f"encoder_combine_input"] = {
            "class": "copy",
            "from": ["encoder_0_squeeze", enc_mix, "encoder_1_squeeze"],
        }

    enc_combine, layers = add_blstm_stack(
        network,
        from_list="encoder_combine_input",
        name="lstm_combine",
        **blstm_combine_args,
    )

    for speaker_idx in [0, 1]:
        add_softmax_output(
            network,
            from_list=enc_combine,
            num_outputs=num_outputs,
            name=f"output_speaker_{speaker_idx}",
            target=f"layer:targets_{speaker_idx}_squeeze",
            scale=output_args.get("loss_scale", 1.0) / 2,
            **output_args,
        )

    network["output"] = {
        "class": "stack",
        "from": ["output_speaker_0", "output_speaker_1"],
    }

    return network, python_code, dim_tags


def make_blstm_hybrid_dual_output_combine_enc_recog_model(
    num_outputs: int,
    speaker_idx: int,
    gt_args: Dict[str, Any],
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
    blstm_combine_args: Dict = {},
) -> Tuple[Dict, Union[str, List], Dict[str, Dim]]:

    network = {}
    python_code = []

    from_list = "data"

    use_mixed_input = (
        blstm_mix_args.get("num_layers", 1) > 0
        or blstm_combine_args.get("num_layers", 1) > 0
    )
    assert use_mixed_input

    mix_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_mix", **gt_args
    )

    from_list, dim_tags = add_speech_separation(network, from_list=from_list)

    network["speech_separator"]["subnetwork"]["separated_stft_pit"]["subnetwork"][
        "default_permutation"
    ] = {
        "class": "constant",
        "value": CodeWrapper("np.array([0, 1])"),
        "shape": (dim_tags["speaker"],),
    }

    network["speech_separator"]["subnetwork"]["separated_stft_pit"]["subnetwork"][
        "output"
    ]["position"] = "default_permutation"

    sep_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_sep", **gt_args
    )

    enc_01, _ = add_blstm_stack(
        network, from_list=sep_features, name="lstm_01", **blstm_01_args
    )

    enc_mix, _ = add_blstm_stack(
        network, from_list=mix_features, name="lstm_mix", **blstm_mix_args
    )

    network["encoder_mix"] = {
        "class": "copy",
        "from": enc_mix,
    }
    enc_mix = "encoder_mix"

    for s in [0, 1]:
        network[f"encoder_{s}"] = {
            "class": "slice",
            "from": enc_01,
            "axis": dim_tags["speaker"],
            "slice_start": s,
            "slice_end": s + 1,
        }
        network[f"encoder_{s}_squeeze"] = {
            "class": "squeeze",
            "from": f"encoder_{s}",
            "axis": "auto",
        }

    if blstm_01_mix_args.get("num_layers", 0):
        for s in [0, 1]:
            network[f"encoder_{s}+mix_input"] = {
                "class": "combine",
                "from": [f"encoder_{s}_squeeze", enc_mix],
                "kind": "add",
            }

        dim_tags["enc_01_mix_input_feature"] = FeatureDim(
            "enc_01_mix_input_feature_dim", None
        )
        network["encoder_01+mix_input"] = {
            "class": "split_dims",
            "from": ["encoder_0+mix_input", "encoder_1+mix_input"],
            "axis": "F",
            "dims": (dim_tags["speaker"], dim_tags["enc_01_mix_input_feature"]),
        }

        enc_01, _ = add_blstm_stack(
            network,
            from_list="encoder_01+mix_input",
            name="lstm_01+mix",
            **blstm_01_mix_args,
        )

        for s in [0, 1]:
            network[f"encoder_{s}_mix"] = {
                "class": "slice",
                "from": enc_01,
                "axis": dim_tags["speaker"],
                "slice_start": s,
                "slice_end": s + 1,
            }
            network[f"encoder_{s}_mix_squeeze"] = {
                "class": "squeeze",
                "from": f"encoder_{s}_mix",
                "axis": "auto",
            }
        network[f"encoder_combine_input"] = {
            "class": "copy",
            "from": ["encoder_0_mix_squeeze", "encoder_1_mix_squeeze"],
        }
    else:
        network["encoder_combine_input"] = {
            "class": "copy",
            "from": ["encoder_0_squeeze", enc_mix, "encoder_1_squeeze"],
        }

    enc_combine, _ = add_blstm_stack(
        network,
        from_list="encoder_combine_input",
        name="lstm_combine",
        **blstm_combine_args,
    )

    network[f"output_speaker_{speaker_idx}"] = {
        "class": "linear",
        "from": enc_combine,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    network["output"] = {
        "class": "copy",
        "from": f"output_speaker_{speaker_idx}",
    }

    return network, python_code, dim_tags


def make_blstm_hybrid_dual_output_soft_context_model(
    num_outputs: int,
    gt_args: Dict[str, Any],
    target_key: str = "data:target_classes",
    freeze_separator: bool = True,
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
    blstm_context_args: Dict = {},
    aux_loss_01_layers: List[Tuple[int, float]] = [(3, 0.3)],
    aux_loss_01_mix_layers: List[Tuple[int, float]] = [(3, 0.3)],
    use_logits: bool = False,
    pre_context_loss_scale: float = 0.5,
    output_args: Dict = {},
) -> Tuple[Dict, Union[str, List], Dict[str, Dim]]:
    network = {}
    python_code = []

    from_list, dim_tags = add_speech_separation(
        network, from_list="data", trainable=not freeze_separator
    )
    sep_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_sep", padding=(80, 0), **gt_args
    )

    use_mixed_input = (
        blstm_mix_args.get("num_layers", 1) > 0
        or blstm_01_mix_args.get("num_layers", 1) > 0
    )
    if use_mixed_input:
        mix_features, python_code = add_gt_feature_extraction(
            network, from_list="data", name="gt_mix", padding=(80, 0), **gt_args
        )

    network["targets_merged"] = {
        "class": "merge_dims",
        "from": target_key,
        "axes": ["B", dim_tags["speaker"]],
    }

    enc_01, layers = add_blstm_stack(
        network, from_list=sep_features, name="lstm_01", **blstm_01_args
    )

    for layer, scale in aux_loss_01_layers:
        network[f"aux_01_{layer}_merge"] = {
            "class": "merge_dims",
            "from": layers[layer - 1],
            "axes": ["B", dim_tags["speaker"]],
        }
        add_softmax_output(
            network,
            from_list=f"aux_01_{layer}_merge",
            num_outputs=num_outputs,
            name=f"aux_01_output_{layer}",
            target="layer:targets_merged",
            scale=scale,
            **output_args,
        )

    if use_mixed_input:
        enc_mix, _ = add_blstm_stack(
            network, from_list=mix_features, name="lstm_mix", **blstm_mix_args
        )

        network["encoder_mix"] = {
            "class": "copy",
            "from": enc_mix,
        }
        enc_mix = "encoder_mix"

        for speaker_idx in [0, 1]:
            network[f"encoder_{speaker_idx}"] = {
                "class": "slice",
                "from": enc_01,
                "axis": dim_tags["speaker"],
                "slice_start": speaker_idx,
                "slice_end": speaker_idx + 1,
            }
            network[f"encoder_{speaker_idx}_squeeze"] = {
                "class": "squeeze",
                "from": f"encoder_{speaker_idx}",
                "axis": "auto",
            }

            if blstm_01_args.get("num_layers", 1) and blstm_mix_args.get(
                "num_layers", 1
            ):
                network[f"encoder_{speaker_idx}+mix_input"] = {
                    "class": "combine",
                    "from": [f"encoder_{speaker_idx}_squeeze", enc_mix],
                    "kind": "add",
                }
            else:
                network[f"encoder_{speaker_idx}+mix_input"] = {
                    "class": "copy",
                    "from": [f"encoder_{speaker_idx}_squeeze", enc_mix],
                }

        dim_tags["enc_01_mix_input_feature"] = FeatureDim(
            "enc_01_mix_input_feature_dim", None
        )
        network["encoder_01+mix_input"] = {
            "class": "split_dims",
            "from": ["encoder_0+mix_input", "encoder_1+mix_input"],
            "axis": "F",
            "dims": (dim_tags["speaker"], dim_tags["enc_01_mix_input_feature"]),
        }

        enc_01, layers = add_blstm_stack(
            network,
            from_list="encoder_01+mix_input",
            name="lstm_01+mix",
            **blstm_01_mix_args,
        )

        for layer, scale in aux_loss_01_mix_layers:
            network[f"aux_01+mix_{layer}_merge"] = {
                "class": "merge_dims",
                "from": layers[layer - 1],
                "axes": ["B", dim_tags["speaker"]],
            }
            add_softmax_output(
                network,
                from_list=f"aux_01+mix_{layer}_merge",
                num_outputs=num_outputs,
                name=f"aux_01+mix_output_{layer}",
                target="layer:targets_merged",
                scale=scale,
                **output_args,
            )

    network["output_merge"] = {
        "class": "merge_dims",
        "from": enc_01,
        "axes": ["B", dim_tags["speaker"]],
    }
    network["output"] = {
        "class": "linear",
        "from": "output_merge",
        "n_out": num_outputs,
        **output_args,
    }
    network["output_softmax"] = {
        "class": "activation",
        "from": "output",
        "activation": "softmax",
        "loss": "ce",
        "target": "layer:targets_merged",
        "loss_scale": pre_context_loss_scale,
    }

    if use_logits:
        context_layer = "output"
    else:
        context_layer = "output_softmax"

    network["output_split"] = {
        "class": "split_dims",
        "from": context_layer,
        "axis": "B",
        "dims": (batch_dim, dim_tags["speaker"]),
    }
    network["axis_switch"] = {
        "class": "constant",
        "value": CodeWrapper("np.array([1, 0])"),
    }
    network["output_permute"] = {
        "class": "gather",
        "from": "output_split",
        "axis": dim_tags["speaker"],
        "position": "axis_switch",
    }
    network["output_permute_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "output_permute",
        "size_base": "output_split",
    }
    network["context_encoder_input"] = {
        "class": "copy",
        "from": [*enc_01, "output_permute_reinterpret"],
    }
    context_enc, _ = add_blstm_stack(
        network,
        from_list="context_encoder_input",
        name="context_encoder",
        **blstm_context_args,
    )
    network["final_output_merge"] = {
        "class": "merge_dims",
        "from": context_enc,
        "axes": ["B", dim_tags["speaker"]],
    }
    add_softmax_output(
        network,
        from_list="final_output_merge",
        num_outputs=num_outputs,
        name="final_output",
        target="layer:targets_merged",
        **output_args,
    )

    return network, python_code, dim_tags


def make_blstm_hybrid_dual_output_soft_context_recog_model(
    num_outputs: int,
    speaker_idx: int,
    gt_args: Dict[str, Any],
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
    blstm_context_args: Dict = {},
    use_logits: bool = False,
) -> Tuple[Dict, Union[str, List], Dict[str, Dim]]:

    network = {}
    python_code = []

    from_list = "data"

    use_mixed_input = (
        blstm_mix_args.get("num_layers", 1) > 0
        or blstm_01_mix_args.get("num_layers", 1) > 0
    )
    if use_mixed_input:
        mix_features, python_code = add_gt_feature_extraction(
            network, from_list=from_list, name="gt_mix", **gt_args
        )

    from_list, dim_tags = add_speech_separation(network, from_list=from_list)

    network["speech_separator"]["subnetwork"]["separated_stft_pit"]["subnetwork"][
        "default_permutation"
    ] = {
        "class": "constant",
        "value": CodeWrapper("np.array([0, 1])"),
        "shape": (dim_tags["speaker"],),
    }

    network["speech_separator"]["subnetwork"]["separated_stft_pit"]["subnetwork"][
        "output"
    ]["position"] = "default_permutation"

    sep_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_sep", **gt_args
    )

    enc_01, _ = add_blstm_stack(
        network, from_list=sep_features, name="lstm_01", **blstm_01_args
    )

    if use_mixed_input:
        enc_mix, _ = add_blstm_stack(
            network, from_list=mix_features, name="lstm_mix", **blstm_mix_args
        )
        network["encoder_01"] = {
            "class": "copy",
            "from": enc_01,
        }
        enc_01 = "encoder_01"

        network["encoder_mix"] = {
            "class": "copy",
            "from": enc_mix,
        }
        enc_mix = "encoder_mix"

        for s in [0, 1]:
            network[f"encoder_{s}"] = {
                "class": "slice",
                "from": enc_01,
                "axis": dim_tags["speaker"],
                "slice_start": s,
                "slice_end": s + 1,
            }
            network[f"encoder_{s}_squeeze"] = {
                "class": "squeeze",
                "from": f"encoder_{s}",
                "axis": "auto",
            }

            if blstm_01_args.get("num_layers", 1) and blstm_mix_args.get(
                "num_layers", 1
            ):
                network[f"encoder_{s}+mix_input"] = {
                    "class": "combine",
                    "from": [f"encoder_{s}_squeeze", enc_mix],
                    "kind": "add",
                }
            else:
                network[f"encoder_{s}+mix_input"] = {
                    "class": "copy",
                    "from": [f"encoder_{s}_squeeze", enc_mix],
                }

        dim_tags["enc_01_mix_input_feature"] = FeatureDim(
            "enc_01_mix_input_feature_dim", None
        )
        network["encoder_01+mix_input"] = {
            "class": "split_dims",
            "from": ["encoder_0+mix_input", "encoder_1+mix_input"],
            "axis": "F",
            "dims": (dim_tags["speaker"], dim_tags["enc_01_mix_input_feature"]),
        }

        enc_01, _ = add_blstm_stack(
            network,
            from_list="encoder_01+mix_input",
            name="lstm_01+mix",
            **blstm_01_mix_args,
        )
    network["encoder_slice_main"] = {
        "class": "slice",
        "from": enc_01,
        "axis": dim_tags["speaker"],
        "slice_start": speaker_idx,
        "slice_end": speaker_idx + 1,
    }
    network["encoder_slice_main_squeeze"] = {
        "class": "squeeze",
        "from": "encoder_slice_main",
        "axis": "auto",
    }
    network["encoder_slice_other"] = {
        "class": "slice",
        "from": enc_01,
        "axis": dim_tags["speaker"],
        "slice_start": 1 - speaker_idx,
        "slice_end": 2 - speaker_idx,
    }
    network["encoder_slice_other_squeeze"] = {
        "class": "squeeze",
        "from": "encoder_slice_other",
        "axis": "auto",
    }

    network["output"] = {
        "class": "linear",
        "from": "encoder_slice_other_squeeze",
        "n_out": num_outputs,
    }
    context_layer = "output"
    if not use_logits:
        network["output_softmax"] = {
            "class": "activation",
            "from": "output",
            "activation": "softmax",
        }
        context_layer = "output_softmax"

    network["context_encoder_input"] = {
        "class": "copy",
        "from": ["encoder_slice_main_squeeze", context_layer],
    }
    context_enc, _ = add_blstm_stack(
        network,
        from_list="context_encoder_input",
        name="context_encoder",
        **blstm_context_args,
    )
    network["final_output"] = {
        "class": "linear",
        "from": context_enc,
        "activation": "log_softmax",
        "n_out": num_outputs,
        "is_output_layer": True,
    }

    return network, python_code, dim_tags
