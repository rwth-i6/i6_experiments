from typing import Dict, List, Tuple, Optional
from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from i6_experiments.users.berger.network.helpers.specaug import (
    add_specaug_layer,
    get_specaug_funcs,
)
from i6_experiments.users.berger.network.helpers.pred_succ import (
    add_pred_succ_targets_noblank,
)


def make_blstm_hybrid_model(
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

    add_softmax_output(
        network, from_list=from_list, num_outputs=num_outputs, **output_args
    )

    return network, python_code


def make_blstm_hybrid_multitask_model(
    num_outputs: int,
    nonword_labels: List[int] = [0],
    context_transformation_func: Optional[str] = None,
    context_label_dim: int = 0,
    blstm_args: Dict = {},
    mlp_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, str]:
    network, python_code = make_blstm_hybrid_model(
        num_outputs=num_outputs,
        blstm_args=blstm_args,
        mlp_args=mlp_args,
        output_args=output_args,
    )

    base_labels = "data:classes"
    if context_transformation_func is not None:
        network["base_labels"] = {
            "class": "eval",
            "from": base_labels,
            "eval": context_transformation_func,
            "out_type": {"dim": context_label_dim},
        }
        base_labels = "base_labels"

    pred_layer, succ_layer = add_pred_succ_targets_noblank(
        network, context_label_dim, nonword_labels, base_labels
    )

    # === PREDECESSOR ===
    from_list = add_feed_forward_stack(network, "encoder", name="pred_ff", **mlp_args)

    add_softmax_output(
        network,
        from_list=from_list,
        name="pred_output",
        num_outputs=context_label_dim + 1,
        target=f"layer:{pred_layer}",
        **output_args,
    )

    # === SUCCESSOR ===
    from_list = add_feed_forward_stack(network, "encoder", name="succ_ff", **mlp_args)

    add_softmax_output(
        network,
        from_list=from_list,
        name="succ_output",
        num_outputs=context_label_dim + 1,
        target=f"layer:{succ_layer}",
        **output_args,
    )

    return network, python_code


def make_blstm_hybrid_recog_model(
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
