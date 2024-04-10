import copy

from i6_core.returnn.config import ReturnnConfig

from .helper import get_network
from .helper import make_nn_config


def get_wei_config(specaug=False, no_min_seq_len=False):
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
    if no_min_seq_len:
        nn_config = copy.deepcopy(nn_config)
        del nn_config["min_seq_len"]
        del nn_config["min_seq_length"]

    return nn_config
