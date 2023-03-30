from typing import Dict, List, Tuple, Optional
from i6_experiments.users.berger.network.helpers.conformer import (
    get_variance_scaling_init,
    add_conformer_stack,
    add_initial_conv,
    add_transposed_conv,
)
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from i6_experiments.users.berger.network.helpers.specaug import (
    add_specaug_layer,
    get_specaug_funcs,
)


def make_conformer_hybrid_model(
    num_outputs: int,
    specaug_args: Dict = {},
    init_conv_args: Dict = {},
    conformer_args: Dict = {},
    ff_args: Dict = {},
    output_args: Dict = {},
    transposed_conv_args: Dict = {},
    aux_loss_blocks: List[Tuple[int, float]] = [(4, 0.3), (8, 0.3)],
) -> Tuple[Dict, str]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network, **specaug_args)
    python_code += get_specaug_funcs()

    from_list = add_initial_conv(
        network, "vgg_conv", from_list=from_list, **init_conv_args
    )

    from_list, blocks = add_conformer_stack(
        network, from_list=from_list, name="conformer", **conformer_args
    )

    network["encoder"] = {"class": "copy", "from": from_list}

    for key, val in [
        ("num_layers", 1),
        ("size", 384),
        ("activation", None),
        ("dropout", None),
        ("l2", None),
        ("initializer", get_variance_scaling_init()),
    ]:
        ff_args.setdefault(key, val)

    output_args.setdefault("initializer", get_variance_scaling_init())

    for block, scale in aux_loss_blocks:
        transp_conv = add_transposed_conv(
            network,
            from_list=blocks[block - 1],
            name=f"aux_{block}",
            **transposed_conv_args,
        )
        mlp_out = add_feed_forward_stack(
            network,
            from_list=transp_conv,
            name=f"aux_{block}_mlp",
            **ff_args,
        )
        add_softmax_output(
            network,
            from_list=mlp_out,
            num_outputs=num_outputs,
            name=f"aux_output_{block}",
            scale=scale,
            **output_args,
        )

    transp_conv = add_transposed_conv(
        network,
        from_list="encoder",
        name="output",
        **transposed_conv_args,
    )
    mlp_out = add_feed_forward_stack(
        network,
        from_list=transp_conv,
        name="output_mlp",
        **ff_args,
    )
    add_softmax_output(
        network,
        from_list=mlp_out,
        num_outputs=num_outputs,
        **output_args,
    )

    return network, python_code


def make_conformer_hybrid_recog_model(
    num_outputs: int,
    init_conv_args: Dict = {},
    conformer_args: Dict = {},
    ff_args: Dict = {},
    transposed_conv_args: Dict = {},
) -> Tuple[Dict, str]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_initial_conv(
        network, "vgg_conv", from_list=from_list, **init_conv_args
    )

    from_list, _ = add_conformer_stack(
        network, from_list=from_list, name="conformer", **conformer_args
    )

    network["encoder"] = {"class": "copy", "from": from_list}

    for key, val in [
        ("num_layers", 1),
        ("size", 384),
        ("activation", None),
        ("dropout", None),
        ("l2", None),
        ("initializer", get_variance_scaling_init()),
    ]:
        ff_args.setdefault(key, val)

    transp_conv = add_transposed_conv(
        network,
        from_list="encoder",
        name="output",
        **transposed_conv_args,
    )
    mlp_out = add_feed_forward_stack(
        network,
        from_list=transp_conv,
        name="output_mlp",
        **ff_args,
    )
    network["output"] = {
        "class": "linear",
        "from": mlp_out,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, python_code