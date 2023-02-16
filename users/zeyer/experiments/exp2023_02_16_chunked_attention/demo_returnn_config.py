#!rnn.py

"""
Demo config for the RETURNN training of the chunked attention model.

This is assumed to be run within RETURNN.
"""


import copy

from returnn.config import get_global_config

from i6_experiments.users.zeineldeen.models.asr.encoder.conformer_encoder import (
    ConformerEncoder,
)
from i6_experiments.users.zeineldeen.models.asr.decoder.rnn_decoder import RNNDecoder


def _create_config():
    exp_config = {}

    # -------------------------- network -------------------------- #

    conformer_encoder = ConformerEncoder(
        target=target,
        input_layer=None,
        num_blocks=1,
        specaug=False,
        ff_dim=64,
        enc_key_dim=32,
    )
    conformer_encoder.create_network()

    transformer_decoder = RNNDecoder(
        base_model=conformer_encoder,
        target=target,
        embed_dim=32,
        lstm_num_units=32,
        output_num_units=32,
        enc_key_dim=32,
        lstm_lm_proj_dim=32,
    )
    transformer_decoder.create_network()

    decision_layer_name = transformer_decoder.decision_layer_name
    exp_config["search_output_layer"] = decision_layer_name

    # add full network
    exp_config["network"] = conformer_encoder.network.get_net()  # type: dict
    exp_config["network"].update(transformer_decoder.network.get_net())

    # -------------------------- end network -------------------------- #

    return exp_config


config = get_global_config()
task = config.value("task", "train")
use_tensorflow = True


train = {
    "class": "TaskNumberBaseConvertDataset",
    "input_base": 3,
    "output_base": 15,
    "max_input_seq_len": 100,
    "num_seqs": 1000,
}
dev = copy.deepcopy(train)
dev.update(
    {
        "num_seqs": 100,
        "fixed_random_seed": 42,
    }
)
target = "classes"
extern_data = {
    "data": {"dim": 3, "sparse": True},
    "classes": {"dim": 15, "sparse": True},
}

globals().update(_create_config())


batch_size = 1000
max_seq_length = {target: 100}
optimizer = {"class": "adam", "epsilon": 1e-8}
