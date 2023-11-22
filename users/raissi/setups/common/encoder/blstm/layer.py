from typing import Dict


def blstm_network(
    layers=6 * [512],
    dropout: float = 0.1,
    l2: float = 0.1,
    specaugment: bool = True,
    as_data: bool = False,
    transform_func_name: str = "transform",
):
    num_layers = len(layers)
    assert num_layers > 0

    result = {}

    if specaugment:
        if as_data:
            eval_str = f"self.network.get_config().typed_value('{transform_func_name}')(source(0, as_data={as_data}), network=self.network)"
        else:
            eval_str = (
                f"self.network.get_config().typed_value('{transform_func_name}')(source(0), network=self.network)"
            )

        result["source"] = {
            "class": "eval",
            "eval": eval_str,
        }
        input_first_layer = "source"
    else:
        input_first_layer = ["data"]

    for l, size in enumerate(layers):
        l += 1  # start counting from 1
        for direction, name in [(1, "fwd"), (-1, "bwd")]:
            if l == 1:
                from_layers = input_first_layer
            else:
                from_layers = ["fwd_%d" % (l - 1), "bwd_%d" % (l - 1)]
            result["%s_%d" % (name, l)] = {
                "class": "rec",
                "unit": "nativelstm2",
                "direction": direction,
                "n_out": size,
                "dropout": dropout,
                "L2": l2,
                "from": from_layers,
            }

    encoder_output_size = result[f"fwd_{num_layers}"]["n_out"] + result[f"bwd_{num_layers}"]["n_out"]

    result["encoder-output"] = {
        "class": "copy",
        "from": ["fwd_6", "bwd_6"],
        "n_out": encoder_output_size,
    }

    return result


def add_subsmapling_via_max_pooling(network_dict: Dict, pool_factor: int = 2, num_layers: int = None) -> Dict:
    num_layers_ = num_layers if num_layers is not None else min(pool_factor % 3 + 1, 2)
    for i in range(num_layers_):
        l_n = i + 2
        for pre_name in ["fwd", "bwd"]:
            network_dict[f"{pre_name}_{l_n}"]["from"] = f"max_pool_{i+1}"
        network_dict[f"max_pool_{i+1}"] = (
            {
                "class": "pool",
                "from": [f"fwd_{i+2}", f"bwd_{i+2}"],
                "mode": "max",
                "padding": "same",
                "pool_size": (pool_factor,),
                "trainable": False,
            },
        )

    return network_dict


def add_subsampling_via_feature_stacking(network_dict: Dict, stride_factor: int = 3):
    input_data = network_dict["fwd_1"]["from"]
    network_dict["feature_stacking"] = (
        {
            "class": "window",
            "from": input_data,
            "stride": stride_factor,
            "window_left": stride_factor - 1,
            "window_right": 0,
            "window_size": stride_factor,
        },
    )
    network_dict["feature_stacking_merged"] = (
        {
            "axes": (2, 3),
            "class": "merge_dims",
            "from": ["feature_stacking"],
            "keep_order": True,
        },
    )

    # "from": "data" -> "from": "feature_stacking_merged"
    for pre_name in ["fwd", "bwd"]:
        network_dict[f"{pre_name}_1"]["from"] = ["feature_stacking_merged"]

    return network_dict
