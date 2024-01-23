from typing import Dict, List, Tuple
from i6_core.returnn.config import CodeWrapper

from i6_experiments.users.berger.network.helpers.conformer_moritz_exact import (
    add_frontend,
    add_conformer_block,
    add_aux_output,
    add_output,
)
from i6_experiments.users.berger.network.helpers.specaug_2 import (
    add_specaug_layer,
    get_specaug_funcs,
)


def make_conformer_hybrid_dualchannel_model(
    specaug_args: Dict = {},
    use_secondary_audio: bool = False,
    use_comb_init: bool = False,
    sep_comb_diag: float = 0.9,
    mix_comb_diag: float = 0.1,
    comb_noise: float = 0.01,
    emulate_single_speaker: bool = False,
) -> Tuple[Dict, List[str]]:
    network = {}
    python_code = []

    from_prim = "data:features_primary"

    if use_secondary_audio:
        from_sec = "data:features_secondary"
    else:
        from_sec = "data:features_mix"

    python_code += get_specaug_funcs()

    from_prim = add_specaug_layer(network, name="specaug_prim", from_list=from_prim, **specaug_args)
    from_prim = add_frontend(network, from_list=from_prim, prefix="prim_")
    for block_idx in range(1, 7):
        from_prim = add_conformer_block(network, from_list=from_prim, block_idx=block_idx, prefix="prim_")

    if emulate_single_speaker:
        network["combine_encoders"] = {
            "class": "copy",
            "from": from_prim,
        }
    else:
        from_sec = add_specaug_layer(network, name="specaug_sec", from_list=from_sec, **specaug_args)
        from_sec = add_frontend(network, from_list=from_sec, prefix="sec_")
        for block_idx in range(1, 7):
            from_sec = add_conformer_block(network, from_list=from_sec, block_idx=block_idx, prefix="sec_")

        network["combine_encoders"] = {
            "class": "linear",
            "from": [from_prim, from_sec],
            "n_out": 512,
            "activation": None,
        }

        if use_comb_init:
            network["combine_encoders"]["forward_weights_init"] = CodeWrapper(
                f"np.vstack(({sep_comb_diag} * np.eye(512), {mix_comb_diag} * np.eye(512))) + np.random.uniform(low=-{comb_noise}, high={comb_noise}, size=(1024,512))"
            )

    from_list = "combine_encoders"

    for block_idx in range(7, 13):
        from_list = add_conformer_block(network, from_list=from_list, block_idx=block_idx, prefix="mas_")

    network["encoder"] = {"class": "layer_norm", "from": from_list}

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
        "set_sparse_dim": 12001,
    }
    target = "layer:classes_sparse"

    add_aux_output(network, from_list=from_prim, target=target)
    add_output(network, from_list="encoder", target=target)

    return network, python_code


def make_conformer_hybrid_dualchannel_recog_model(
    specaug_args: Dict = {},
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
        "slice_end": 50,
    }
    from_prim = "features_prim"

    python_code += get_specaug_funcs()

    from_prim = add_specaug_layer(network, name="specaug_prim", from_list=from_prim, **specaug_args)
    from_prim = add_frontend(network, from_list=from_prim, prefix="prim_")
    for block_idx in range(1, 7):
        from_prim = add_conformer_block(network, from_list=from_prim, block_idx=block_idx, prefix="prim_")

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
            "slice_start": 50 if use_secondary_audio else 100,
            "slice_end": 100 if use_secondary_audio else 150,
        }
        from_sec = "features_sec"
        from_sec = add_specaug_layer(network, name="specaug_sec", from_list=from_sec, **specaug_args)
        from_sec = add_frontend(network, from_list=from_sec, prefix="sec_")
        for block_idx in range(1, 7):
            from_sec = add_conformer_block(network, from_list=from_sec, block_idx=block_idx, prefix="sec_")

        network["combine_encoders"] = {
            "class": "linear",
            "from": [from_prim, from_sec],
            "n_out": 512,
            "activation": None,
        }
    from_list = "combine_encoders"

    for block_idx in range(7, 13):
        from_list = add_conformer_block(network, from_list=from_list, block_idx=block_idx, prefix="mas_")

    network["encoder"] = {"class": "layer_norm", "from": from_list}

    add_output(network, from_list="encoder", target="")
    network.pop("length_masked")
    network["output"].pop("loss")
    network["output"].pop("loss_opts")
    network["output"].pop("target")
    network["output"]["from"] = ["upsampled0"]
    network["output"]["class"] = "linear"
    network["output"]["activation"] = "log_softmax"
    network["output"]["n_out"] = 12001

    return network, python_code
