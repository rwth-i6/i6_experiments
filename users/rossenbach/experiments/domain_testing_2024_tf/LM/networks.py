from dataclasses import dataclass
from typing import Optional

@dataclass
class LstmLmSettings:
    embedding_dim: int
    lstm_sizes: list[int]
    bottleneck: Optional[int]
    lstm_dropout: float
    output_dropout: float


def get_lstm_lm_network(settings: LstmLmSettings, for_export: bool = False) -> dict[str, any]:
    network = {
        "input": {"class": "linear", "n_out": settings.embedding_dim, "activation": "identity",
                  "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "from": ["data:delayed"]},
    }
    current_last_layer = "input"
    for i, lstm_size in enumerate(settings.lstm_sizes):
        layer = f"lstm{i}"
        network[layer] = {
            "class": "rec", "unit": "lstm",
            "forward_weights_init" : "random_normal_initializer(mean=0.0, stddev=0.1)",
            "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "n_out": 1024, "dropout": settings.lstm_dropout, "L2": 0.0, "direction": 1, "from": [current_last_layer]
        }
        if for_export:
            network[layer]["initial_state"] = "keep_over_epoch_no_init"
        current_last_layer = layer

    if settings.bottleneck is not None:
        network["bottleneck"] = {
            "class": "linear", "n_out": settings.bottleneck, "activation": "identity",
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "dropout": settings.lstm_dropout,
            "from": [current_last_layer]
        }
        current_last_layer = "bottleneck"

    if for_export:
        network["output"] = {
            "class": "linear", "activation": "log_softmax", "dropout": settings.output_dropout, "use_transposed_weights": True,
            "loss_opts": {'num_sampled': 16384, 'use_full_softmax': True, 'nce_loss': False},
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "loss": "sampling_loss", "target": "data", "from": [current_last_layer]
        }
    else:
        network["output"] = {
            "class": "softmax", "dropout": settings.output_dropout, "use_transposed_weights": True,
            "loss_opts": {'num_sampled': 16384, 'use_full_softmax': True, 'nce_loss': False},
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "loss": "sampling_loss", "target": "data", "from": [current_last_layer]
        }
    return network

def get_lstm_lm_network_for_shallow_fusion(settings: LstmLmSettings, out_dim) -> dict[str, any]:
    network = {
        "input": {"class": "linear", "n_out": settings.embedding_dim, "activation": "identity"},
    }
    current_last_layer = "input"
    for i, lstm_size in enumerate(settings.lstm_sizes):
        layer = f"lstm{i}"
        network[layer] = {
            "class": "rnn_cell", "unit": "LSTMBlock",
            "n_out": 1024, "dropout": 0.0, "L2": 0.0,  "unit_opts": {"forget_bias": 0.0},
            "from": [current_last_layer]
        }
        current_last_layer = layer
    if settings.bottleneck is not None:
        network["bottleneck"] = {
            "class": "linear", "n_out": settings.bottleneck, "activation": "identity",
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "dropout": settings.lstm_dropout,
            "from": [current_last_layer]
        }
        current_last_layer = "bottleneck"
    network["output"] = {
        "class": "linear", "activation": "identity", "dropout": 0.0, "use_transposed_weights": True,
        "from": [current_last_layer],
        "n_out": out_dim,
    }
    return network
