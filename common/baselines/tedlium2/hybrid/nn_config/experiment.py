from .helper import get_network
from .helper import make_nn_config


def get_wei_config(specaug=False):
    network = get_network(spec_augment=specaug)
    nn_config = make_nn_config(network)
    nn_config["extern_data"] = {
        "data": {
            "dim": 80,
            "shape": (None, 80),
            "available_for_inference": True,
        },  # input: 80-dimensional logmel features
        "classes": {"dim": 9001, "shape": (None,), "available_for_inference": True, "sparse": True, "dtype": "int16"},
    }

    return nn_config
