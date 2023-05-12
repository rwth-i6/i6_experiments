from typing import Array


def blstm_network(
    layers: Array = 6 * [512], dropout: float = 0.1, l2: float = 0.1, specaugment: bool = True, as_data: bool = False,
    transform_func_name: str = 'transform',
):
    num_layers = len(layers)
    assert num_layers > 0

    result = {}

    if specaugment:
        result["source"] = {
            "class": "eval",
            "eval": f"self.network.get_config().typed_value('{transform_func_name}')(source(0, as_data={as_data}), network=self.network)",
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

    result["encoder-output"] =  {
        "class": "copy",
        "from": ["fwd_6", "bwd_6"],
        "n_out": encoder_output_size,
    }

    return result


def 
