"""
Helpers for speech separation system
"""


# noinspection PyUnresolvedReferences
def get_network(epoch: int):
    """
    Add separator to network
    """
    from returnn.config import get_global_config

    config = get_global_config()

    epoch  # noqa
    sys.path.append(os.path.dirname(mask_est_net_dict))
    from returnn_net_dict import network as speech_separator_network

    masks = speech_separator_network["output"]["from"]
    # need to set dim tags for unflattening speaker and stft bin dims
    for layer, layer_args in speech_separator_network.items():
        if layer_args["class"] == "split_dims" and layer_args["dims"] == [2, 257]:
            speech_separator_network[layer]["dims"] = (
                speaker_dim,
                _stft_output_feature_dim__2__1_dim,
            )
    network = config.typed_value("network")
    network["speech_separator"]["subnetwork"] = speech_separator_network
    network["speech_separator"]["subnetwork"].update(
        {
            "masks": {
                "class": "copy",
                "from": masks,
            },
            "separated_stft_pit": {
                "class": "subnetwork",
                "from": "masks",
                "subnetwork": {
                    "permutation_constant": {
                        "class": "constant",
                        "value": np.array([[0, 1], [1, 0]]),
                        "shape": (
                            permutation_dim,
                            speaker_dim,
                        ),
                    },
                    "separated_stft": {
                        "class": "combine",
                        "kind": "mul",
                        "from": [f"base:data", "base:masks"],
                    },
                    "separated_stft_permutations": {
                        "class": "gather",
                        "axis": speaker_dim,
                        "position": "permutation_constant",
                        "from": "separated_stft",
                    },
                    "permutation_mse": {
                        "class": "subnetwork",
                        "from": "separated_stft_permutations",
                        "subnetwork": {
                            "diff": {
                                "class": "combine",
                                "kind": "sub",
                                "from": ["data", f"base:base:base:target_signals_stft"],
                            },
                            "square": {
                                "class": "activation",
                                "activation": "square",
                                "from": "diff",
                            },
                            "output": {
                                "class": "reduce",
                                "mode": "sum",
                                "from": "square",
                                "axes": ["T", "F", speaker_dim],
                            },
                        },
                    },
                    "permutation_argmin": {
                        "class": "reduce",
                        "mode": "argmin",
                        "from": "permutation_mse",
                        "axes": permutation_dim,
                    },
                    "permutation_indices": {
                        "class": "gather",
                        "axis": permutation_dim,
                        "position": "permutation_argmin",
                        "from": "permutation_constant",
                    },
                    "output": {
                        "class": "gather",
                        "axis": speaker_dim,
                        "position": "permutation_indices",
                        "from": "separated_stft",
                    },
                },
            },
            "output": {"class": "copy", "from": "separated_stft_pit"},
        }
    )
    freeze = config.typed_value("freeze", [])
    if isinstance(freeze, str):
        freeze = [freeze]
    for freeze_part in freeze:
        assert freeze_part in network, "trying to freeze {freeze_part} which is not in network"
        network[freeze_part]["trainable"] = False
    return network
