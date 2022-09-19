from collections import ChainMap

from .networks import mlp_network
from .configs import feed_forward_config
from .constants import (
    BASE_CRNN_CONFIG, BASE_LSTM_CONFIG,
    BASE_VITERBI_LRS, BASE_NETWORK_CONFIG,
)

from i6_core.returnn.config import ReturnnConfig

BASE_FFNN_LAYERS = {
    "layers": 6 * [2048],
    "feature_window": 15,
    "activation": "relu",
}

def _network_wrapper(layers, activation, dropout, l2, feature_window, **_ignored):
    return mlp_network(layers, activation, dropout, l2, feature_window)

def viterbi_ffnn(num_input, network_kwargs={}, **kwargs):
    kwargs = ChainMap(kwargs, BASE_VITERBI_LRS, BASE_CRNN_CONFIG)
    network_kwargs = ChainMap(network_kwargs, BASE_FFNN_LAYERS, BASE_NETWORK_CONFIG)
    config = feed_forward_config(
        num_input,
        mlp_network(**network_kwargs),
        **kwargs
    )
    config['network']['output'].update(
        loss_opts = {"focal_loss_factor": 2.0},
        loss = "ce"
    )
    return ReturnnConfig(config)