from .conformer_returnn_dict_network_generator import prefix_all_keys

def se_block( # TODO naming convention off here, should have '_00' cause inital implementation
    net=None,
    in_l=None,
    prefix = None,

    # Shared args:
    model_dim = None,
):

    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"

    net_add = {
        "_SE_reduce": {
            "class" : "reduce",
            "mode" : "mean",
            "from"  : in_l,
            "axes" : "T"},
        "_SE_linear1": {
            "class" : "linear",
            "from" : "_SE_reduce",
            "n_out" : 32},
        "_SE_act1" : {
            "class" : "activation",
            "activation" : "swish",
            "from" : "_SE_linear1"},
        "_SE_linear2" : {
            "class" : "linear",
            "from" :  "_SE_act1",
            "n_out" : model_dim},
        "_SE_act2" : {
            "class" : "activation",
            "activation" : "swish",
            "from" : "_SE_linear2" },
        "_SE_elm_mul" :  {
            "class" : "eval",
            "eval" : "source(0) * source(1)",
            "from" : ["_SE_act2", in_l]},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_SE_elm_mul"


def se_block_scale( 
    net=None,
    in_l=None,
    prefix = None,

    se_dim = None,
    se_act = "swish",

    # Shared args:
    model_dim = None,
):

    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"

    net_add = {
        "_SE_reduce": {
            "class" : "reduce",
            "mode" : "mean",
            "from"  : in_l,
            "axes" : "T"},
        "_SE_linear1": {
            "class" : "linear",
            "from" : "_SE_reduce",
            "n_out" : se_dim},
        "_SE_act1" : {
            "class" : "activation",
            "activation" : "swish",
            "from" : "_SE_linear1"},
        "_SE_linear2" : {
            "class" : "linear",
            "from" :  "_SE_act1",
            "n_out" : model_dim},
        "_SE_act2" : {
            "class" : "activation",
            "activation" : se_act,
            "from" : "_SE_linear2" },
        "_SE_elm_mul" :  {
            "class" : "eval",
            "eval" : "source(0) * source(1)",
            "from" : ["_SE_act2", in_l]},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_SE_elm_mul"

def se_block_02_not_prefixed(
    net=None,
    in_l=None,

    # Shared args:
    model_dim = None,
):

    assert net, "no net"
    assert in_l, "no input layer"

    net_add = {
        "_SE_reduce": {
            "class" : "reduce",
            "mode" : "mean",
            "from"  : in_l,
            "axes" : "T"},
        "_SE_linear1": {
            "class" : "linear",
            "from" : "_SE_reduce",
            "n_out" : 32},
        "_SE_act1" : {
            "class" : "activation",
            "activation" : "swish",
            "from" : "_SE_linear1"},
        "_SE_linear2" : {
            "class" : "linear",
            "from" :  "_SE_act1",
            "n_out" : model_dim},
        "_SE_act2" : {
            "class" : "activation",
            "activation" : "swish",
            "from" : "_SE_linear2" },
        "_SE_elm_mul" :  {
            "class" : "eval",
            "eval" : "source(0) * source(1)",
            "from" : ["_SE_act2", in_l]},
    }

    net.update(net_add)

    return net, "_SE_elm_mul"

def stochatic_depth_00(
    subnetwork = None,
    survival_prob = None,
    subnet_last = None,

    in_l = None,

    prefix = None,

):

    random_bernulli = f"tf.compat.v1.distributions.Bernoulli(probs={survival_prob}).sample(sample_shape=())"
    switch = f"tf.equal({random_bernulli}, 0)"

    net_add = {
        f"{prefix}_train_flag" : {
            "class": "train_flag"},
        f"{prefix}_switch_train" : {
            "class" : "switch",
            "condition" : f"{prefix}_train_flag",
            "true_from" : f"{prefix}_stoch_depth_in_train",
            "false_from" : f"{prefix}_stoch_depth_in_eval"},
        f"{prefix}_stoch_depth_in_eval" : { # Then only multipy bu surival prob
            "class": "subnetwork",
            "from" : in_l,
            "subnetwork" : {
                in_l : { # We just copy this overunder the same name ( need it cause also used as input from the subnet )
                    "class" : "copy",
                    "from" : "data"},
                **subnetwork,
                "output" : {
                    "class": "eval",
                    "from" : [subnet_last, in_l], # TODO subnet last
                    "eval" : f"source(0) * {survival_prob} + source(1)"}}},
        f"{prefix}_stoch_depth_in_train": {
            "class": "cond", 
            "from": [],
            "condition": { # First condition only checks if we are in train using TrainFlagLayer
                "class": "eval", 
                "from": [], 
                "out_type": {
                    "batch_dim_axis": None, 
                    "shape": (), 
                    "dtype": "bool"},
                "eval": switch }, # In training generate random bernulli with 'surival_prob' if 0, then skip layer, if 1 the use layer ...
            "true_layer": { # TRUE add subnetwork output to redidual ( in_l )
                "class": "subnetwork", 
                "from": in_l, 
                "subnetwork": {
                    in_l : { # We just copy this overunder the same name
                        "class" : "copy",
                        "from" : "data"}, # TODO: we 
                    **subnetwork, # Most likely a full confore module
                    "output" : {
                        "class": "eval",
                        "from" : [subnet_last, in_l],
                        "eval" : "source(0) + source(1)"}}}, 
            "false_layer": { # FALSE: only add the residual i.e.: only 1 * input
                "class": "copy", 
                "from": in_l}},
    }


    return net_add, f"{prefix}_switch_train"


# + also put the train condition in a condition layer
def stochatic_depth_02_speedup(
    subnetwork = None,
    survival_prob = None,
    subnet_last = None,

    in_l = None,

    prefix = None,

):

    random_bernulli = f"tf.compat.v1.distributions.Bernoulli(probs={survival_prob}).sample(sample_shape=())"
    switch = f"tf.equal({random_bernulli}, 0)"

    net_train = {
            "class": "cond", 
            "from": [],
            "condition": { # First condition only checks if we are in train using TrainFlagLayer
                "class": "eval", 
                "from": [], 
                "out_type": {
                    "batch_dim_axis": None, 
                    "shape": (), 
                    "dtype": "bool"},
                "eval": switch }, # In training generate random bernulli with 'surival_prob' if 0, then skip layer, if 1 the use layer ...
            "true_layer": { # TRUE add subnetwork output to redidual ( in_l )
                "class": "subnetwork", 
                "from": in_l, 
                "subnetwork": {
                    in_l : { # We just copy this overunder the same name
                        "class" : "copy",
                        "from" : "data"}, # TODO: we 
                    **subnetwork, # Most likely a full confore module
                    "output" : {
                        "class": "eval",
                        "from" : [subnet_last, in_l],
                        "eval" : "source(0) + source(1)"}}}, 
            "false_layer": { # FALSE: only add the residual i.e.: only 1 * input
                "class": "copy", 
                "from": in_l}
    }

    net_eval = {
            "class": "subnetwork",
            "from" : in_l,
            "subnetwork" : {
                in_l : { # We just copy this overunder the same name ( need it cause also used as input from the subnet )
                    "class" : "copy",
                    "from" : "data"},
                **subnetwork,
                "output" : {
                    "class": "eval",
                    "from" : [subnet_last, in_l], # TODO subnet last
                    "eval" : f"source(0) * {survival_prob} + source(1)"}}
    }

    net_add = {
        f"{prefix}_train_flag" : {
            "class": "train_flag"},
        f"{prefix}_cond_train" : {
            "class" : "cond",
            "from" : [],
            "condition" : {
                "class" : "copy",  # TODO: can use train frag here
                "from": [f"{prefix}_train_flag"], 
                "out_type": {
                    "batch_dim_axis": None, 
                    "shape": (), 
                    "dtype": "bool"},
            },
            "true_layer" : {
                **net_train
            },
            "false_layer" : {
                **net_eval
            }
        }
    }


    return net_add, f"{prefix}_cond_train"


def stochatic_depth_03_namescopes(
    subnetwork = None,
    survival_prob = None,
    subnet_last = None,

    multipy_by_surivial_prob_ineval = True,

    in_l = None,

    prefix = None,

):

    # Subnetworks need to share the namescopes to share params:
    # -> so we just add a root namescope with the specific layers name:
    for layer_name in subnetwork:
        subnetwork[layer_name]["name_scope"] = f"/{layer_name}"
    # TODO: verfiy tensorflow namescopes

    random_bernulli = f"tf.compat.v1.distributions.Bernoulli(probs={survival_prob}).sample(sample_shape=())"
    switch = f"tf.equal({random_bernulli}, 1)"

    eval_case = f"source(0) * {survival_prob} + source(1)"
    if not multipy_by_surivial_prob_ineval:
        eval_case = f"source(0) + source(1)"

    net_train = {
            "class": "cond", 
            "from": [],
            "condition": { # First condition only checks if we are in train using TrainFlagLayer
                "class": "eval", 
                "from": [], 
                "out_type": {
                    "batch_dim_axis": None, 
                    "shape": (), 
                    "dtype": "bool"},
                "eval": switch }, # In training generate random bernulli with 'surival_prob' if 0, then skip layer, if 1 the use layer ...
            "true_layer": { # TRUE add subnetwork output to redidual ( in_l )
                "class": "subnetwork", 
                "from": in_l, 
                "subnetwork": {
                    in_l : { # We just copy this overunder the same name
                        "class" : "copy",
                        "from" : "data"}, 
                    **subnetwork, # Most likely a full confore module
                    "output" : {
                        "class": "eval",
                        "from" : [subnet_last, in_l],
                        "eval" : "source(0) + source(1)"}}}, 
            "false_layer": { # FALSE: only add the residual i.e.: only 1 * input
                "class": "copy", 
                "from": in_l}
    }

    net_eval = {
            "class": "subnetwork",
            "from" : in_l,
            "subnetwork" : {
                in_l : { # We just copy this overunder the same name ( need it cause also used as input from the subnet )
                    "class" : "copy",
                    "from" : "data"},
                **subnetwork,
                "output" : {
                    "class": "eval",
                    "from" : [subnet_last, in_l],
                    "eval" : eval_case}}
    }

    net_add = {
        f"{prefix}_cond_train" : {
            "class" : "cond",
            "from" : [],
            "condition" : {
                "class": "train_flag"
            },
            "true_layer" : {
                **net_train
            },
            "false_layer" : {
                **net_eval
            }
        }
    }


    return net_add, f"{prefix}_cond_train"

def stochatic_depth_03_namescopes(
    subnetwork = None,
    survival_prob = None,
    subnet_last = None,

    multipy_by_surivial_prob_ineval = True,

    in_l = None,

    prefix = None,

):

    # Subnetworks need to share the namescopes to share params:
    # -> so we just add a root namescope with the specific layers name:
    for layer_name in subnetwork:
        subnetwork[layer_name]["name_scope"] = f"/{layer_name}"
    # TODO: verfiy tensorflow namescopes

    random_bernulli = f"tf.compat.v1.distributions.Bernoulli(probs={survival_prob}).sample(sample_shape=())"
    switch = f"tf.equal({random_bernulli}, 1)"

    eval_case = f"source(0) * {survival_prob} + source(1)"
    if not multipy_by_surivial_prob_ineval:
        eval_case = f"source(0) + source(1)"

    net_train = {
            "class": "cond", 
            "from": [],
            "condition": { # First condition only checks if we are in train using TrainFlagLayer
                "class": "eval", 
                "from": [], 
                "out_type": {
                    "batch_dim_axis": None, 
                    "shape": (), 
                    "dtype": "bool"},
                "eval": switch }, # In training generate random bernulli with 'surival_prob' if 0, then skip layer, if 1 the use layer ...
            "true_layer": { # TRUE add subnetwork output to redidual ( in_l )
                "class": "subnetwork", 
                "from": in_l, 
                "subnetwork": {
                    in_l : { # We just copy this overunder the same name
                        "class" : "copy",
                        "from" : "data"}, 
                    **subnetwork, # Most likely a full confore module
                    "output" : {
                        "class": "eval",
                        "from" : [subnet_last, in_l],
                        "eval" : "source(0) + source(1)"}}}, 
            "false_layer": { # FALSE: only add the residual i.e.: only 1 * input
                "class": "copy", 
                "from": in_l}
    }

    net_eval = {
            "class": "subnetwork",
            "from" : in_l,
            "subnetwork" : {
                in_l : { # We just copy this overunder the same name ( need it cause also used as input from the subnet )
                    "class" : "copy",
                    "from" : "data"},
                **subnetwork,
                "output" : {
                    "class": "eval",
                    "from" : [subnet_last, in_l],
                    "eval" : eval_case}}
    }

    net_add = {
        f"{prefix}_cond_train" : {
            "class" : "cond",
            "from" : [],
            "condition" : {
                "class": "train_flag"
            },
            "true_layer" : {
                **net_train
            },
            "false_layer" : {
                **net_eval
            }
        }
    }


    return net_add, f"{prefix}_cond_train"


def stochatic_depth_04_no_copy(
    subnetwork = None,
    survival_prob = None,
    subnet_last = None,

    multipy_by_surivial_prob_ineval = True,

    in_l = None,
    
    extra_multiply = 1.0,

    prefix = None,

):

    # Subnetworks need to share the namescopes to share params:
    # -> so we just add a root namescope with the specific layers name:
    for layer_name in subnetwork:
        subnetwork[layer_name]["name_scope"] = f"/{layer_name}"

        if isinstance(subnetwork[layer_name]["from"], list):
            if in_l in subnetwork[layer_name]["from"]:
                index = subnetwork[layer_name]["from"].index(in_l)
                subnetwork[layer_name]["from"][index] = "data"
        elif isinstance(subnetwork[layer_name]["from"], str):
            if subnetwork[layer_name]["from"] == in_l:
                subnetwork[layer_name]["from"] = "data"
    # TODO: verfiy tensorflow namescopes

    random_bernulli = f"tf.compat.v1.distributions.Bernoulli(probs={survival_prob}).sample(sample_shape=())"
    switch = f"tf.equal({random_bernulli}, 1)"

    eval_case = f"source(0) * {extra_multiply} * {survival_prob} + source(1)"
    if not multipy_by_surivial_prob_ineval:
        eval_case = f"source(0) * {extra_multiply} + source(1)"

    net_train = {
            "class": "cond", 
            "from": [],
            "condition": { # First condition only checks if we are in train using TrainFlagLayer
                "class": "eval", 
                "from": [], 
                "out_type": {
                    "batch_dim_axis": None, 
                    "shape": (), 
                    "dtype": "bool"},
                "eval": switch }, # In training generate random bernulli with 'surival_prob' if 0, then skip layer, if 1 the use layer ...
            "true_layer": { # TRUE add subnetwork output to redidual ( in_l )
                "class": "subnetwork", 
                "from": in_l, 
                "subnetwork": {
                    **subnetwork, # Most likely a full confore module
                    "output" : {
                        "class": "eval",
                        "from" : [subnet_last, "data"],
                        "eval" : f"source(0) * {extra_multiply} + source(1)"}}}, 
            "false_layer": { # FALSE: only add the residual i.e.: only 1 * input
                "class": "copy", 
                "from": in_l}
    }

    net_eval = {
            "class": "subnetwork",
            "from" : in_l,
            "subnetwork" : {
                in_l : { # We just copy this overunder the same name ( need it cause also used as input from the subnet )
                    "class" : "copy",
                    "from" : "data"},
                **subnetwork,
                "output" : {
                    "class": "eval",
                    "from" : [subnet_last, in_l],
                    "eval" : eval_case}}
    }

    net_add = {
        f"{prefix}_cond_train" : {
            "class" : "cond",
            "from" : [],
            "condition" : {
                "class": "train_flag"
            },
            "true_layer" : {
                **net_train
            },
            "false_layer" : {
                **net_eval
            }
        }
    }


    return net_add, f"{prefix}_cond_train"

def add_feature_stacking(
    net=None,
    in_l=None,

    stacking_stride = None,
    window_size = None,
    window_left = None,
    window_right = None

):
    net_add = {
        'feature_stacking_merged': { # 'feature_stacking_window': [B,T|'ceildiv_right(conv1p:conv:s0, 3)'[?],'feature_stacking_window:window'(3),F|F'conv0p:conv:s1*conv1_1:channel'(1600)]
            'axes': ["dim:3", "F"],  # adapt bhv=12  #TODO: make dynamic, get axes as input?
            'class': "merge_dims",  # Should be conv merged
            'from': ['feature_stacking_window']},
        'feature_stacking_window': {
            'class': 'window', 
            'from': [in_l], 
            'stride': stacking_stride, 
            'window_left': window_left, 
            'window_right': window_right, 
            'window_size': window_size},
    }

    net.update(net_add)
    return net, 'feature_stacking_merged'

def tf_group_norm(x, G=32, eps=1e-5):
    import tensorflow as tf
    B, T, C = x.get_shape().as_list()
    G = min(G, C)

    x = tf.reshape(x, [B, T, G, C // G])
    mean, var = tf.nn.moments(x, [1, 3], keep_dims=True) # Batch and Feature?
    x = (x - mean) / tf.sqrt(var + eps)

    gamma = tf.get_variable('gamma', [ 1, 1, C], initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable('beta', [ 1, 1, C], initializer=tf.constant_initializer(0.0))

    x = tf.reshape(x, [B, T, C]) * gamma + beta

def group_normalization_old( # TODO: unfinished
    net=None,
    in_l=None,
    prefix=None,

    groups = None,
    epsilon = None,

    model_dim = None,
):
    # Per: https://github.com/taki0112/Group_Normalization-Tensorflow
    # "in_lengh" : {"class" : "length", "from": in_l, "axes" : "F"},
    net_add = {
        "gamma" : {"class" : "variable", "init": 1.0, "trainable": True},
        "beta"  : {"class" : "variable", "init": 0.0, "trainable": True},
        "mean" : {"class" : "reduce", "axes": "B", "mode" : "mean", "from": in_l}, # Or only batch?
        "var" : {"class" : "combine", "kind": "squared_difference", "from" : ["mean", in_l]}, # TODO: stop gradients here?
        "split" : {"class" : "split_dims", "from" : in_l, "axis": "F", "dims" : (groups, model_dim // groups)},
        "sub_mean" : {
            "class" : "eval",
            "from" : ["split", "mean"],
            "eval" : "source(0) - source(1)"
        },
        "diff_var" : {
            "class" : "eval",
            "from" : ["sub_mean", "var"],
            "eval" : f"source(0)/tf.sqrt(source(1) - {epsilon})"
        },
        "merge" : {"class" : "merge_dims", "from" : "diff_var", "axes" : [f"dim:{groups}", f"dim:{model_dim//groups}"]},
        "apply" : {"class" : "eval", "from" : ["merge", "gamma", "beta"], "eval": "source(0) * source(1) + source(2)"}
    }

    net.update(net_add)
    return net, "apply"


def group_normalization( 
    net=None,
    in_l=None,
    prefix=None,

    groups = None,
    epsilon = None,

    model_dim = None,
):
    # Per: https://github.com/taki0112/Group_Normalization-Tensorflow
    # "in_lengh" : {"class" : "length", "from": in_l, "axes" : "F"},
    if prefix and prefix != "":
        prefix += "_" # If we have a prefix add the under score other whise keep same for compatability

    net_add = {
        f"{prefix}split" : {"class" : "split_dims", "from" : in_l, "axis": "F", "dims" : (groups, model_dim // groups)},
        f"{prefix}norm" : {"class" : "norm", "from" : f"{prefix}split", "axes" : "TF"},
        f"{prefix}merge" : {"class" : "merge_dims", "from" : f"{prefix}norm", "axes" : [f"dim:{groups}", f"dim:{model_dim//groups}"]},
    }

    net.update(net_add)
    return net, f"{prefix}merge"

def add_auxilary_loss(
    net=None,
    in_l=None,
    prefix=None,

    # specific:
    aux_dim = None,
    aux_strides = None,

    # general:
    initialization = None,
    model_dim = None
):

    net_add = {
        '_ff1': { 
            'activation': 'relu',
            'class': 'linear',
            'forward_weights_init': initialization,
            'from': ['_length_masked'],
            'n_out': aux_dim,
            'with_bias': True},
        '_ff2': { 
            'activation': None,
            'class': 'linear',
            'forward_weights_init': initialization,
            'from': ['_ff1'],
            'n_out': aux_dim,
            'with_bias': True},
        '_length_masked': {
            'class': 'reinterpret_data', 
            'from': ['_upsampled0'], 
            'size_base': 'data:classes'},
        '_output_prob': { 
            'class': 'softmax',
            'dropout': 0.0,
            'from': ['_ff2'],
            'loss': 'ce',
            'loss_opts': {
                'focal_loss_factor': 0.0, 
                'label_smoothing': 0.0, 
                'use_normalized_loss': False},
            'loss_scale': 0.5,
            'target': 'classes'},
        '_upsampled0': { 
            'activation': 'relu',
            'class': 'transposed_conv',
            'filter_size': (3,),
            'from': [in_l],
            'n_out': model_dim,
            'strides': (aux_strides,),
            'with_bias': True},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )
    return net, f"{prefix}_output_prob" # But this should never be used as input
