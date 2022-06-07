def make_subsampling_000_old_bhv(
    net=None,
    in_l="source0",
    time_reduction=None,

    #specific:
    embed_dropout = None,
    embed_l2 = None,

    # General
    initialization = None,
    model_dim = None
):
    assert net, "need network"
    net.update({
        "conv0_0" : {
            "class": "conv", 
            "from": in_l, 
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": None, 
            "with_bias": True, 
            },  # (T,50,32)
        "conv0_1" : {
            "class": "conv", 
            "from": f"conv0_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": 'relu', 
            "with_bias": True, 
            },  # (T,50,32)
        "conv0p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (1, 2), 
            'strides': (1, 2),
            "from": "conv0_1", 
             }, # (T, 25, 32)
        "conv1_0" : {
            "class": "conv", 
            "from": "conv0p", 
            "padding": "same", 
            "filter_size": (3, 3), 
            "n_out": 64,
            "activation": None, 
            "with_bias": True,
             }, # (T, 25, 64)
        "conv1_1" : {
            "class": "conv", 
            "from": "conv1_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 64, 
            "activation": 'relu', 
            "with_bias": True, 
            }, # (T,25,64)
        "conv1p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (time_reduction, 1), 
            'strides': (time_reduction, 1),
            "from": "conv1_1", 
            },
        "conv_merged" : {
            "class": "merge_dims", 
            "from": "conv1p", 
            "axes": ["dim:25", "dim:64"]},
        'embedding': { 
            'L2': embed_l2,
            'activation': None,
            'class': 'linear',
            'forward_weights_init': initialization,
            'from': ['conv_merged'],
            'n_out': model_dim,
            'with_bias': True},
        'embedding_dropout': {
            'class': 'dropout', 
            'dropout': embed_dropout, 
            'from': ['embedding']},
    })
    return net, "embedding_dropout"

def make_subsampling_001(
    net=None,
    in_l="source0",
    time_reduction=None,

    #specific:
    embed_dropout = None,
    embed_l2 = None,

    # General
    initialization = None,
    model_dim = None
):
    assert net, "need network"
    net.update({
        "conv0_0" : {
            "class": "conv", 
            "from": in_l, 
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": None, 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0_1" : {
            "class": "conv", 
            "from": f"conv0_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": 'relu', 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (1, 2), 
            'strides': (1, 2),
            "from": "conv0_1", 
            "in_spatial_dims": ["T", "dim:50"] }, # (T, 25, 32)
        "conv1_0" : {
            "class": "conv", 
            "from": "conv0p", 
            "padding": "same", 
            "filter_size": (3, 3), 
            "n_out": 64,
            "activation": None, 
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:25"] }, # (T, 25, 64)
        "conv1_1" : {
            "class": "conv", 
            "from": "conv1_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 64, 
            "activation": 'relu', 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:25"]}, # (T,25,64)
        "conv1p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (time_reduction, 1), 
            'strides': (time_reduction, 1),
            "from": "conv1_1", 
            "in_spatial_dims": ["T", "dim:25"]},
        "conv_merged" : {
            "class": "merge_dims", 
            "from": "conv1p", 
            "axes": ["dim:25", "dim:64"]},
        'embedding': { 
            'L2': embed_l2,
            'activation': None,
            'class': 'linear',
            'forward_weights_init': initialization,
            'from': ['conv_merged'],
            'n_out': model_dim,
            'with_bias': True},
        'embedding_dropout': {
            'class': 'dropout', 
            'dropout': embed_dropout, 
            'from': ['embedding']},
    })
    return net, "embedding_dropout"


def make_unsampling_001(
    net=None,
    in_l=None,

    time_reduction=None
):
    net.update({
        'encoder' : {
            'class': 'layer_norm',
            'from': [in_l]},
        'upsampled2': { 
            'activation': 'relu',
            'class': 'transposed_conv',
            'filter_size': (3,),
            'from': ['encoder'],
            'n_out': 256, #TODO: I think this should use model_dim
            'strides': (time_reduction,),
            'with_bias': True}
    })

    return net, "upsampled2"

# The espnet subsampling:
def make_subsampling_002(
    net=None,
    in_l="source0",
    time_reduction=None,

    #specific:
    embed_dropout = None,
    embed_l2 = None,

    conv0p_pools = None,  # [2, 2]
    conv1p_pools = None,  # [2, 2]

    # General
    initialization = None,
    model_dim = None
):
    assert net, "need network"

    # Only two concolution blocks
    new_net = {
        "conv0": {
            "activation": "relu",
            "class": "conv",
            "filter_size": [3, 3],
            "from": "source0",
            "n_out": 32,
            "padding": "same",
            "in_spatial_dims": ["T", "dim:50"],
            "with_bias": True },
        "conv0p": {
            "class": "pool",
            "from": "conv0",
            "mode": "max",
            "padding": "same",
            "pool_size": conv0p_pools,
            "in_spatial_dims": ["T", "dim:50"],
            "trainable": False },
        "conv1": {
            "activation": "relu",
            "class": "conv",
            "filter_size": [3, 3],
            "from": "conv0p",
            "n_out": 64,
            "padding": "same",
            "in_spatial_dims": ["T", "dim:50"],
            "with_bias": True },
        "conv1p": {
            "class": "pool",
            "from": "conv1",
            "mode": "max",
            "padding": "same",
            "in_spatial_dims": ["T", "dim:50"],
            "pool_size": conv1p_pools,
            "trainable": False },
        "conv_merged" : {
            "class": "merge_dims", 
            "from": "conv1p", 
            "axes": ["dim:25", "dim:64"]}, # TODO: This depends on pooling, make this adaptive
        'embedding': { 
            'L2': embed_l2,
            'activation': None,
            'class': 'linear',
            'forward_weights_init': initialization,
            'from': ['conv_merged'],
            'n_out': model_dim,
            'with_bias': True},
        'embedding_dropout': {
            'class': 'dropout', 
            'dropout': embed_dropout, 
            'from': ['embedding']},
    }

    net.update(new_net)

    return net, 'embedding_dropout'

def make_unsampling_002( # For no vgg convoultions unsampling
    net=None,
    in_l=None,

    time_reduction=None,

    model_dim = None
):
    net.update({
        'encoder' : {
            'class': 'layer_norm',
            'from': [in_l]},
        'upsampled2': { 
            'activation': 'relu',
            'class': 'transposed_conv',
            'filter_size': (3,),
            'from': ['encoder'],
            'n_out': model_dim,
            'strides': (time_reduction,),
            'with_bias': True}
    })

    return net, "upsampled2"



# Includes SE block in the end of downsampling
def make_subsampling_003(
    net=None,
    in_l="source0",
    time_reduction=None,

    #specific:
    embed_dropout = None,
    embed_l2 = None,

    # General
    initialization = None,
    model_dim = None
):
    assert net, "need network"
    net.update({
        "conv0_0" : {
            "class": "conv", 
            "from": in_l, 
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": None, 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0_1" : {
            "class": "conv", 
            "from": f"conv0_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": 'relu', 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (1, 2), 
            'strides': (1, 2),
            "from": "conv0_1", 
            "in_spatial_dims": ["T", "dim:50"] }, # (T, 25, 32)
        "conv1_0" : {
            "class": "conv", 
            "from": "conv0p", 
            "padding": "same", 
            "filter_size": (3, 3), 
            "n_out": 64,
            "activation": None, 
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:25"] }, # (T, 25, 64)
        "conv1_1" : {
            "class": "conv", 
            "from": "conv1_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 64, 
            "activation": 'relu', 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:25"]}, # (T,25,64)
        "conv1p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (time_reduction, 1), 
            'strides': (time_reduction, 1),
            "from": "conv1_1", 
            "in_spatial_dims": ["T", "dim:25"]},
        "conv_merged" : {
            "class": "merge_dims", 
            "from": "conv1p", 
            "axes": ["dim:25", "dim:64"]},
        'embedding': { 
            'L2': embed_l2,
            'activation': None,
            'class': 'linear',
            'forward_weights_init': initialization,
            'from': ['conv_merged'],
            'n_out': model_dim,
            'with_bias': True},
    })

    from .network_additions import se_block

    net, se_out = se_block( # Now we ann an se block, bacause those are cool
        net = net,
        in_l =  "embedding",
        prefix = "emb",
        
        model_dim = model_dim
    )

    net.update({
        'embedding_dropout': {
            'class': 'dropout', 
            'dropout': embed_dropout, 
            'from': [se_out]},
    })
    return net, "embedding_dropout"


def make_unsampling_003( # for the version with an SE block in the end of downsampling
    net=None,
    in_l=None,

    time_reduction=None
):
    net.update({
        'encoder' : {
            'class': 'layer_norm',
            'from': [in_l]},
        'upsampled2': { 
            'activation': 'relu',
            'class': 'transposed_conv',
            'filter_size': (3,),
            'from': ['encoder'],
            'n_out': 256, #TODO: I think this should use model_dim
            'strides': (time_reduction,),
            'with_bias': True}
    })

    return net, "upsampled2"


def make_subsampling_004_feature_stacking(
    net=None,
    in_l="source0",
    time_reduction=None,

    #specific:
    embed_dropout = None,
    embed_l2 = None,
    stacking_stride = None,
    window_size = None,
    window_left = None,
    window_right = None,
    unsampling_strides = None,
    sampling_activation = "relu",

    # General
    initialization = None,
    model_dim = None
):
    assert net, "need network"
    net_add = {
        "conv0_0" : {
            "class": "conv", 
            "from": in_l, 
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": None, 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0_1" : {
            "class": "conv", 
            "from": f"conv0_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": sampling_activation, 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (1, 2), 
            'strides': (1, 2),
            "from": "conv0_1", 
            "in_spatial_dims": ["T", "dim:50"] }, # (T, 25, 32)
        "conv1_0" : {
            "class": "conv", 
            "from": "conv0p", 
            "padding": "same", 
            "filter_size": (3, 3), 
            "n_out": 64,
            "activation": None, 
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:25"] }, # (T, 25, 64)
        "conv1_1" : {
            "class": "conv", 
            "from": "conv1_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 64, 
            "activation": sampling_activation, 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:25"]}, # (T,25,64)
        "conv1p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (time_reduction, 1), 
            'strides': (time_reduction, 1),
            "from": "conv1_1", 
            "in_spatial_dims": ["T", "dim:25"]},
        "conv_merged" : {
            "class": "merge_dims", 
            "from": "conv1p", 
            "axes": ["dim:25", "dim:64"]}}

    from .network_additions import add_feature_stacking

    net.update(net_add)
    net, next = add_feature_stacking(
        net = net,
        in_l = "conv_merged",

        stacking_stride = stacking_stride,
        window_size = window_size,
        window_left = window_left,
        window_right = window_right
    )

    net_end = {
        'embedding': { 
            'activation': None,
            'class': 'linear',
            'forward_weights_init': initialization,
            'from': [next],
            'n_out': model_dim,
            'with_bias': True},
        'embedding_dropout': {
            'class': 'dropout', 
            'dropout': embed_dropout, 
            'from': ['embedding']},
    }

    net.update(net_end)

    return net, "embedding_dropout"

def make_unsampling_004_feature_stacking( # for the version with an SE block in the end of downsampling
    net=None,
    in_l=None,

    time_reduction=None,
    unsampling_strides = None,

    model_dim = None,
):
    net.update({
        'encoder' : {
            'class': 'layer_norm',
            'from': [in_l]},
        'upsampled0': {  # Renamed from upsampling2 don't know why but just following orders
            'activation': 'relu',
            'class': 'transposed_conv',
            'filter_size': (3,),
            'from': ['encoder'],
            'n_out': model_dim, 
            'strides': (unsampling_strides,),
            'with_bias': True}
    })

    return net, "upsampled0"

def make_subsampling_005_fstack_dyn_act(
    net=None,
    in_l="source0",
    time_reduction=None,

    #specific:
    embed_dropout = None,
    embed_l2 = None,
    stacking_stride = None,
    window_size = None,
    window_left = None,
    window_right = None,
    unsampling_strides = None,
    sampling_activation = "relu",

    # General
    initialization = None,
    model_dim = None
):
    assert net, "need network"
    net_add = {
        "conv0_0" : {
            "class": "conv", 
            "from": in_l, 
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": None, 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0_1" : {
            "class": "conv", 
            "from": f"conv0_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": sampling_activation,
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (1, 2), 
            'strides': (1, 2),
            "from": "conv0_1", 
            "in_spatial_dims": ["T", "dim:50"] }, # (T, 25, 32)
        "conv1_0" : {
            "class": "conv", 
            "from": "conv0p", 
            "padding": "same", 
            "filter_size": (3, 3), 
            "n_out": 64,
            "activation": None, 
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:25"] }, # (T, 25, 64)
        "conv1_1" : {
            "class": "conv", 
            "from": "conv1_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 64, 
            "activation": sampling_activation, 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:25"]}, # (T,25,64)
        "conv1p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (time_reduction, 1), 
            'strides': (time_reduction, 1),
            "from": "conv1_1", 
            "in_spatial_dims": ["T", "dim:25"]},
        "conv_merged" : {
            "class": "merge_dims", 
            "from": "conv1p", 
            "axes": ["dim:25", "dim:64"]}}

    from .network_additions import add_feature_stacking

    net.update(net_add)
    net, next = add_feature_stacking(
        net = net,
        in_l = "conv_merged",

        stacking_stride = stacking_stride,
        window_size = window_size,
        window_left = window_left,
        window_right = window_right
    )

    net_end = {
        'embedding': { 
            'activation': None,
            'class': 'linear',
            'forward_weights_init': initialization,
            'from': [next],
            'n_out': model_dim,
            'with_bias': True},
        'embedding_dropout': {
            'class': 'dropout', 
            'dropout': embed_dropout, 
            'from': ['embedding']},
    }

    net.update(net_end)

    return net, "embedding_dropout"