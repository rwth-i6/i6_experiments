from typing import Any, Dict, List, Tuple
from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack
from i6_experiments.users.berger.network.helpers.feature_extraction import (
    add_gt_feature_extraction,
)
from i6_experiments.users.berger.network.helpers.speech_separation import (
    add_speech_separation,
)
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from returnn.tf.util.data import FeatureDim


def make_blstm_hybrid_dual_output_model(
    num_outputs: int,
    gt_args: Dict[str, Any],
    target_key: str = "data:target_rasr",
    use_mixed_input: bool = True,
    freeze_separator: bool = True,
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
    aux_loss_01_layers: List[Tuple[int, float]] = [(3, 0.3)],
    aux_loss_01_mix_layers: List[Tuple[int, float]] = [(3, 0.3)],
    output_args: Dict = {},
) -> Tuple[Dict, str]:
    network = {}
    python_code = []

    from_list, dim_tags = add_speech_separation(network, from_list="data", trainable=not freeze_separator)
    sep_features, python_code = add_gt_feature_extraction(
        network, from_list=from_list, name="gt_sep", padding=(80, 0), **gt_args
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
    use_mixed_input: bool = True,
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
) -> Tuple[Dict, str]:

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

    enc_01, _ = add_blstm_stack(
        network, from_list="squeeze_sep_features", name="lstm_01", **blstm_01_args
    )

    if use_mixed_input:
        enc_mix, _ = add_blstm_stack(
            network, from_list=mix_features, name="lstm_mix", **blstm_mix_args
        )

        network["encoder_01+mix_input"] = {
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
