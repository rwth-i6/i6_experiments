from typing import Dict

"""
     network['source_ivec_reinterpret'] = { 'class': 'reinterpret_data',
                                            'enforce_batch_major': True,
                                            'from': 'source_ivec_pooled',
                                            'set_axes': {'F': 2, 'T': 1},
                                            'size_base': 'conformer_block_01_self_att_ln'}"""


def add_ivectors_to_conformer_encoder(
    network: Dict,
    num_feature_input: int,
    num_ivec_input: int,
    subsampling_factor: int = 4,
    n_ivec_transform: int = 512,
):
    """
    The method follows: https://arxiv.org/pdf/2206.12955
    """
    network["source_features"] = {
        "axis": "F",
        "class": "slice",
        "from": "data",
        "slice_end": num_feature_input,
        "slice_start": 0,
        "trainable": False,
    }
    network["source_ivec"] = {
        "axis": "F",
        "class": "slice",
        "from": "data",
        "slice_end": num_feature_input + num_ivec_input,
        "slice_start": num_feature_input,
        "trainable": False,
    }
    network["source_ivec_pooled"] = {
        "class": "pool",
        "from": "source_ivec",
        "mode": "max",
        "padding": "same",
        "pool_size": (subsampling_factor,),
        "trainable": False,
    }
    network["source_ivec_norm"] = {
        "class": "eval",
        "eval": "tf.math.l2_normalize(source(0), axis=-1)",
        "from": "source_ivec_pooled",
    }
    network["source_ivec_reinterpret"] = {
        "class": "reinterpret_data",
        "enforce_batch_major": True,
        "from": "source_ivec_norm",
        "size_base": "conformer_1_mhsamod_ln",
    }
    ###
    network["ivec_transform"] = {
        "activation": None,
        "class": "linear",
        "from": "source_ivec_reinterpret",
        "n_out": n_ivec_transform,
    }

    network["ivec_transform_2"] = {
        "activation": "tanh",
        "class": "linear",
        "from": "source_ivec_reinterpret",
        "n_out": n_ivec_transform,
    }
    network["ivec_weight_wo_sigmoid"] = {
        "class": "eval",
        "eval": "tf.reduce_sum(source(0, auto_convert=False) * source(1, auto_convert=False), axis=-1, keepdims=True)",
        "from": ["conformer_1_mhsamod_ln", "ivec_transform_2"],
        "out_type": {"dim": 1, "shape": (None, 1)},
    }
    network["ivec_weight"] = {"activation": "sigmoid", "class": "activation", "from": "ivec_weight_wo_sigmoid"}
    network["weighted_ivec"] = {"class": "combine", "from": ["ivec_transform", "ivec_weight"], "kind": "mul"}
    #
    network["mhsa_ivec_input"] = {
        "class": "combine",
        "from": ["conformer_1_mhsamod_ln", "weighted_ivec"],
        "kind": "add",
    }

    network["conformer_1_mhsamod_self_attention"]["from"] = "mhsa_ivec_input"
    network["specaug"]["from"] = "source_features"

    return network
