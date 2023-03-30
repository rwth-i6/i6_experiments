from typing import Dict, Tuple
from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.ctc_loss import (
    add_rasr_fastbw_output_layer,
)
from i6_experiments.users.berger.network.helpers.specaug import (
    add_specaug_layer,
    get_specaug_funcs,
)


def make_blstm_fullsum_ctc_model(
    num_outputs: int,
    specaug_args: Dict = {},
    blstm_args: Dict = {},
    mlp_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, str]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network, **specaug_args)
    python_code += get_specaug_funcs()

    from_list, _ = add_blstm_stack(network, from_list, **blstm_args)

    network["encoder"] = {"class": "copy", "from": from_list}

    from_list = add_feed_forward_stack(network, "encoder", **mlp_args)

    add_rasr_fastbw_output_layer(
        network, from_list=from_list, num_outputs=num_outputs, **output_args
    )

    return network, python_code


def make_blstm_ctc_recog_model(
    num_outputs: int, blstm_args: Dict = {}, mlp_args: Dict = {}, **kwargs
) -> Tuple[Dict, str]:
    network = {}

    from_list = ["data"]

    from_list, _ = add_blstm_stack(network, from_list, **blstm_args)

    network["encoder"] = {"class": "copy", "from": from_list}

    from_list = add_feed_forward_stack(network, "encoder", **mlp_args)

    network["output"] = {
        "class": "linear",
        "from": from_list,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, []
