from typing import Dict, List, Tuple, Union
from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack
from i6_experiments.users.berger.network.helpers.conformer_wei import add_conformer_stack, add_initial_conv
import i6_experiments.users.berger.network.helpers.conformer_i6models as i6models_conformer
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.ctc_loss import (
    add_ctc_output_layer,
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

    add_ctc_output_layer(network=network, from_list=from_list, num_outputs=num_outputs, **output_args)

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


def make_conformer_fullsum_ctc_model(
    num_outputs: int,
    specaug_args: Dict = {},
    vgg_args: Dict = {},
    conformer_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, Union[str, List[str]]]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network, from_list=from_list, **specaug_args)
    python_code += get_specaug_funcs()

    from_list = add_initial_conv(network, from_list, **vgg_args)
    from_list, _ = add_conformer_stack(network, from_list, **conformer_args)

    network["encoder"] = {"class": "copy", "from": from_list}
    add_ctc_output_layer(network=network, from_list=["encoder"], num_outputs=num_outputs, **output_args)

    return network, python_code


def make_conformer_ctc_recog_model(
    num_outputs: int,
    vgg_args: Dict = {},
    conformer_args: Dict = {},
) -> Tuple[Dict, Union[str, List[str]]]:
    network = {}

    from_list = ["data"]

    from_list = add_initial_conv(network, from_list, **vgg_args)
    from_list, _ = add_conformer_stack(network, from_list, **conformer_args)
    network["encoder"] = {"class": "copy", "from": from_list}

    network["output"] = {
        "class": "linear",
        "from": ["encoder"],
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, []


def make_i6models_conformer_fullsum_ctc_model(
    num_outputs: int,
    specaug_args: Dict = {},
    vgg_args: Dict = {},
    conformer_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, Union[str, List[str]]]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network, **specaug_args)
    python_code += get_specaug_funcs()

    from_list = i6models_conformer.add_initial_conv(network, from_list, **vgg_args)
    from_list, _ = i6models_conformer.add_conformer_stack(network, from_list, **conformer_args)

    network["encoder"] = {"class": "copy", "from": from_list}
    add_ctc_output_layer(network=network, from_list=["encoder"], num_outputs=num_outputs, **output_args)

    return network, python_code


def make_i6models_conformer_ctc_recog_model(
    num_outputs: int,
    vgg_args: Dict = {},
    conformer_args: Dict = {},
) -> Tuple[Dict, Union[str, List[str]]]:
    network = {}

    from_list = ["data"]

    from_list = i6models_conformer.add_initial_conv(network, from_list, **vgg_args)
    from_list, _ = i6models_conformer.add_conformer_stack(network, from_list, **conformer_args)
    network["encoder"] = {"class": "copy", "from": from_list}

    network["output"] = {
        "class": "linear",
        "from": ["encoder"],
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, []
