from typing import Dict, List, Tuple, Optional
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.lstm import add_lstm_stack
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from i6_experiments.users.berger.network.helpers.pred_succ import (
    add_pred_succ_targets_noblank,
)


def make_lstm_lm_model(
    num_outputs: int,
    embedding_args: Dict = {},
    lstm_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, str]:
    network = {}

    from_list = ["data:delayed"]

    embedding_args["size"] = embedding_args.get("size", 128)

    from_list = add_feed_forward_stack(
        network,
        from_list,
        name="embedding",
        num_layers=1,
        dropout=0.0,
        **embedding_args,
    )

    from_list = add_lstm_stack(network, from_list, **lstm_args)

    if "target" not in output_args:
        output_args["target"] = "data"

    add_softmax_output(
        network, from_list=from_list, num_outputs=num_outputs, **output_args
    )

    return network


def make_lstm_lm_recog_model(**kwargs) -> Tuple[Dict, str]:

    network = make_lstm_lm_model(**kwargs)
    del network["embedding_1"]["from"]
    del network["output"]["target"]
    del network["output"]["loss"]
    network["output"].update(
        {
            "class": "linear",
            "activation": "log_softmax",
        }
    )

    return network
