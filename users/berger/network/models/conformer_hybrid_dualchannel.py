from typing import Dict, List, Tuple
from i6_core.returnn.config import CodeWrapper

from i6_experiments.users.berger.network.helpers.conformer import (
    add_conformer_stack,
    add_initial_conv,
    add_transposed_conv,
    get_variance_scaling_init,
)
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from i6_experiments.users.berger.network.helpers.specaug_2 import (
    add_specaug_layer,
    get_specaug_funcs,
)


def make_conformer_hybrid_dualchannel_model(
    num_inputs: int,
    num_outputs: int,
    specaug_args: Dict = {},
    vgg_args: Dict = {},
    conformer_args: Dict = {},
    prim_blocks: int = 8,
    sec_blocks: int = 4,
    mas_blocks: int = 4,
    ff_args_aux: Dict = {},
    ff_args_out: Dict = {},
    output_args: Dict = {},
    transposed_conv_args: Dict = {},
    aux_prim_scale: float = 0.5,
    use_secondary_audio: bool = False,
    use_prim_identity_init: bool = False,
    emulate_single_speaker: bool = False,
    with_init: bool = True,
) -> Tuple[Dict, List[str]]:
    network = {}
    python_code = []

    from_prim = "data:features_primary"

    if use_secondary_audio:
        from_sec = "data:features_secondary"
    else:
        from_sec = "data:features_mix"

    # network["concat_features"] = {
    #     "class": "concat",
    #     "from": [(data, "F") for data in from_list],
    # }
    # from_list = "concat_features"

    python_code += get_specaug_funcs()

    from_prim = add_specaug_layer(network, name="specaug_prim", from_list=from_prim, **specaug_args)
    from_prim = add_initial_conv(network, "vgg_conv_prim", from_list=from_prim, with_init=with_init, **vgg_args)
    from_prim, _ = add_conformer_stack(
        network,
        from_list=from_prim,
        name="conformer_prim",
        num_blocks=prim_blocks,
        with_init=with_init,
        **conformer_args,
    )

    if emulate_single_speaker:
        network["combine_encoders"] = {
            "class": "copy",
            "from": from_prim,
        }
    else:
        from_sec = add_specaug_layer(network, name="specaug_sec", from_list=from_sec, **specaug_args)
        from_sec = add_initial_conv(network, "vgg_conv_sec", from_list=from_sec, with_init=with_init, **vgg_args)
        from_sec, _ = add_conformer_stack(
            network,
            from_list=from_sec,
            name="conformer_sec",
            num_blocks=sec_blocks,
            with_init=with_init,
            **conformer_args,
        )

        base_size = conformer_args.get("size", 384)
        network["combine_encoders"] = {
            "class": "linear",
            "from": [from_prim, from_sec],
            "n_out": base_size,
            "activation": None,
        }

        if use_prim_identity_init:
            network["combine_encoders"]["forward_weights_init"] = CodeWrapper(
                f"np.vstack((0.99 * np.eye({base_size}), 0.01 * np.eye({base_size})))"
            )

    from_list, _ = add_conformer_stack(
        network,
        name="conformer_mas",
        from_list="combine_encoders",
        num_blocks=mas_blocks,
        with_init=with_init,
        **conformer_args,
    )

    network["encoder"] = {"class": "copy", "from": from_list}

    for key, val in [
        ("num_layers", 1),
        ("size", 384),
        ("activation", "relu"),
        ("dropout", None),
        ("l2", None),
    ]:
        ff_args_aux.setdefault(key, val)
        ff_args_out.setdefault(key, val)

    if with_init:
        ff_args_aux.setdefault("initializer", get_variance_scaling_init())
        ff_args_out.setdefault("initializer", get_variance_scaling_init())
        output_args.setdefault("initializer", get_variance_scaling_init())

    network["classes_int"] = {
        "class": "cast",
        "from": "data:classes",
        "dtype": "int16",
    }

    network["classes_squeeze"] = {
        "class": "squeeze",
        "from": "classes_int",
        "axis": "F",
    }

    network["classes_sparse"] = {
        "class": "reinterpret_data",
        "from": "classes_squeeze",
        "set_sparse": True,
        "set_sparse_dim": num_outputs,
    }
    target = "layer:classes_sparse"

    if aux_prim_scale != 0:
        transp_conv = add_transposed_conv(
            network,
            from_list=from_prim,
            name="aux_prim",
            with_init=with_init,
            **transposed_conv_args,
        )
        mlp_out = add_feed_forward_stack(
            network,
            from_list=transp_conv,
            name="aux_prim_mlp",
            **ff_args_aux,
        )
        add_softmax_output(
            network,
            from_list=mlp_out,
            num_outputs=num_outputs,
            name="aux_output_prim",
            scale=aux_prim_scale,
            target=target,
            **output_args,
        )

    transp_conv = add_transposed_conv(
        network,
        from_list="encoder",
        name="output",
        with_init=with_init,
        **transposed_conv_args,
    )
    mlp_out = add_feed_forward_stack(
        network,
        from_list=transp_conv,
        name="output_mlp",
        **ff_args_out,
    )
    add_softmax_output(
        network,
        from_list=mlp_out,
        num_outputs=num_outputs,
        target=target,
        **output_args,
    )

    return network, python_code


def make_conformer_hybrid_dualchannel_recog_model(
    num_inputs: int,
    num_outputs: int,
    vgg_args: Dict = {},
    conformer_args: Dict = {},
    prim_blocks: int = 8,
    sec_blocks: int = 4,
    mas_blocks: int = 4,
    ff_args_out: Dict = {},
    output_args: Dict = {},
    transposed_conv_args: Dict = {},
    use_secondary_audio: bool = False,
    emulate_single_speaker: bool = False,
) -> Tuple[Dict, List[str]]:
    network = {}
    python_code = []

    from_list = ["data"]

    network["features_prim"] = {
        "class": "slice",
        "from": from_list,
        "axis": "F",
        "slice_start": 0,
        "slice_end": num_inputs,
    }
    from_prim = "features_prim"
    from_prim = add_initial_conv(network, "vgg_conv_prim", from_list=from_prim, **vgg_args)
    from_prim, _ = add_conformer_stack(
        network, from_list=from_prim, name="conformer_prim", num_blocks=prim_blocks, **conformer_args
    )

    if emulate_single_speaker:
        network["combine_encoders"] = {
            "class": "copy",
            "from": from_prim,
        }
    else:
        network["features_sec"] = {
            "class": "slice",
            "from": from_list,
            "axis": "F",
            "slice_start": num_inputs if use_secondary_audio else 2 * num_inputs,
            "slice_end": 2 * num_inputs if use_secondary_audio else 3 * num_inputs,
        }
        from_sec = "features_sec"

        from_sec = add_initial_conv(network, "vgg_conv_sec", from_list=from_sec, **vgg_args)
        from_sec, _ = add_conformer_stack(
            network, from_list=from_sec, name="conformer_sec", num_blocks=sec_blocks, **conformer_args
        )

        network["combine_encoders"] = {
            "class": "linear",
            "from": [from_prim, from_sec],
            "n_out": conformer_args.get("size", 384),
            "activation": None,
        }

    from_list, _ = add_conformer_stack(
        network, name="conformer_mas", from_list="combine_encoders", num_blocks=mas_blocks, **conformer_args
    )

    network["encoder"] = {"class": "copy", "from": from_list}

    for key, val in [
        ("num_layers", 1),
        ("size", 384),
        ("activation", "relu"),
        ("dropout", None),
        ("l2", None),
        ("initializer", get_variance_scaling_init()),
    ]:
        ff_args_out.setdefault(key, val)

    output_args.setdefault("initializer", get_variance_scaling_init())

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
        **ff_args_out,
    )
    network["output"] = {
        "class": "linear",
        "from": mlp_out,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, python_code
