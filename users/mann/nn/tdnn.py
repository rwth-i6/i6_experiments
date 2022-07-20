import copy

from collections import UserDict
from functools import partial
from i6_core.returnn import CodeWrapper, ReturnnConfig

NotSpecified = object()

class TDNN(UserDict):
    def __init__(self, *args):
        self.tdnn_layers = []

def conv_layer(width, dilation, activation='relu', filter_size=2,
        padding='same', dropout=0.1, l2=0.02, batch_norm=True,
        input=None, forward_weights_init=None):
    assert isinstance(input, str) or input is None
    if forward_weights_init is None:
        forward_weights_init = "glorot_uniform"
    res = {
        "class": "conv",
        "n_out": width,
        "activation": activation,
        "with_bias": True,
        "filter_size": (filter_size,),
        "padding": padding,
        "strides": 1,
        # "L2": l2,
        "dilation_rate": dilation,
        "batch_norm": batch_norm,
        "forward_weights_init": forward_weights_init,
        "dropout": dropout,}
    if input is not None:
        res["from"] = input
    return res

def linear_layer(width, activation='relu',
        dropout=0.1, l2=0.02, batch_norm=True,
        input=None, forward_weights_init=None):
    # assert isinstance(input, str) or input is None
    if forward_weights_init is None:
        forward_weights_init = "glorot_uniform"
    res = {
        "class": "linear",
        "n_out": width,
        "activation": activation,
        "with_bias": True,
        "L2": l2,
        "batch_norm": batch_norm,
        "forward_weights_init": forward_weights_init,
        "dropout": dropout,}
    if input is not None:
        res["from"] = input
    return res


prolog = lambda data_dim, classes_dim: """
import numpy
from returnn.tf.util.data import Data, DimensionTag
time_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="time")
extern_data = {
  "data": {"dim": %d, "same_dim_tags_as": {"t": time_tag}}, 
  "classes": {"dim": %d, "sparse": True, "same_dim_tags_as": {"t": time_tag}},
}

def custom_gather(source, **kwargs):
  import tensorflow as tf
  params = source(0, auto_convert=False, enforce_batch_major=True)
  positions = source(1, auto_convert=False, enforce_batch_major=True)
  print("Custom_gather/params_shape: ", params.shape)
  print("Custom_gather/positions_shape: ", positions.shape)
  res = tf.gather(params, positions, axis=1, batch_dims=1)
  print("Custom_gather/output_shape: ", res.shape)
  return res

def out_lambda(sources, **kwargs):
  dims = tuple(source.output.dim for source in reversed(sources))
  print("Out_lambda dims: ", dims)
  return Data("test", time_dim_axis=1, same_dim_tags_as={"T": time_tag}, shape=(None,) + dims)
""" % (data_dim, classes_dim)

def custom_gather(source, **kwargs):
    import tensorflow as tf
    params = source(0, auto_convert=False, enforce_batch_major=True)
    positions = source(1, auto_convert=False, enforce_batch_major=True)
    print("Custom_gather/params_shape: ", params.shape)
    print("Custom_gather/positions_shape: ", positions.shape)
    res = tf.gather(params, positions, axis=1, batch_dims=1)
    print("Custom_gather/output_shape: ", res.shape)
    return res

def out_lambda(sources, **kwargs):
    dims = tuple(source.output.dim for source in reversed(sources))
    print("Out_lambda dims: ", dims)
    return Data("test", time_dim_axis=1, same_dim_tags_as={"T": time_tag}, shape=(None,) + dims)

class GatherTdnnConfig(ReturnnConfig):

    imports = [
        "import numpy",
        "from returnn.tf.util.data import Data, DimensionTag",
        "time_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description='time')"
    ]

    def __init__(self, base_config, data_dim, class_dim):
        base_config = copy.deepcopy(base_config)
        time_tag_code = CodeWrapper("time_tag")
        base_config["extern_data"] = {
            "data": {"dim": data_dim, "same_dim_tags_as": {"t": time_tag_code}}, 
            "classes": {"dim": class_dim, "sparse": True, "same_dim_tags_as": {"t": time_tag_code}},
        }
        prolog = self.imports + [ custom_gather, out_lambda ]
        super().__init__(base_config, python_prolog=prolog)
    
    def build_compile_config(self):
        conf = copy.deepcopy(self)
        del conf["extern_data"]
        conf.python_prolog.append("""
extern_data = {
  "data": {"dim": %d, "same_dim_tags_as": {"t": time_tag}}, 
  "classes": {"dim": %d, "sparse": True, "same_dim_tags_as": {"t": time_tag}},
} """ % (40, 9001))
        return conf


default_conv_config = {
    "batch_norm": True,
    "L2": 0.02,
    "dropout": 0.1,
    "activation": "relu"
}

def true_tdnn_layer(width, context, input, **kwargs):
    conv_config = copy.copy(default_conv_config)
    conv_config.update(**kwargs)
    padding_left = -context[0]
    padding_right = context[-1]
    assert padding_left >= 0 and padding_right >= 0, "Provided context not supported yet"
    context_rep = CodeWrapper("numpy.array({})".format(context))
    gather_func = CodeWrapper("custom_gather")
    out_lambda  = CodeWrapper("out_lambda")
    res = {
        'combined_features': {'axes': 'except_time', 'class': 'merge_dims', 'from': 'select_context_times'},
        'context': {'class': 'constant', 'value': context_rep, 'dtype': "int32", "with_batch_dim": True},
        'padded_input': {'axes': 'T', 'class': 'pad', 'from': 'data', 'padding': [padding_left, padding_right]},
        'range': {'axis': 'T', 'class': 'range_in_axis', 'from': 'data', 'keepdims': False},
        'select_context_times': {
            'class': 'eval', 'eval': gather_func, 'from': ['padded_input', 'selected_indices'],
            'out_type': out_lambda,
        },
        'selected_indices': { 'class': 'eval', 'eval': 'source(0) + source(1) + offset', 'eval_locals': {'offset': padding_left}, 'from': ['context', 'range']},
        "output": {"class": "linear", "from": "combined_features", "n_out": width, **conv_config},
    }
    return {"class": "subnetwork", "subnetwork": res, "from": [input]}

def true_tdnn_layer_comp(width, dilation, filter_size, input, **kwargs):
    return true_tdnn_layer(width, dilation, input, **kwargs)

# gated_res_bottleneck_layer = {
#     "conv"      : conv_layer(bottleneck, dilation, activation=None, input="data", **kwargs),
#     "gating"    : { "class": "gating", "activation": "tanh", "from": ["conv"] },
#     "linear"    : { "class": "linear", "activation": None, "from": ["gating"], "n_out": width},
#     "projection": { "class": "linear", "activation": None, "from": ["data"], "n_out": width},
#     "output"    : { "class": "combine", "kind": "add", "from": ["projection", "linear"]}
# }

def gated_conv_layer_bottleneck(width, dilation, input_layer, bottleneck, **kwargs):
    res = {
        "conv"      : conv_layer(bottleneck, dilation, activation=None, input="data", **kwargs),
        "gating"    : { "class": "gating", "activation": "tanh", "from": ["conv"] },
        "linear"    : { "class": "linear", "activation": None, "from": ["gating"], "n_out": width},
        "projection": { "class": "linear", "activation": None, "from": ["data"], "n_out": width},
        "output"    : { "class": "combine", "kind": "add", "from": ["projection", "linear"]}
    }
    res = {"class": "subnetwork", "subnetwork": res, "from": [input_layer]}
    return res

class ConfigBuilder:
    def __init__(self, layer_func, *extra_layers, default_topology=None, **create_args):
        # self.layer_func = layer_func
        self.set_layer_func(layer_func, **create_args)
        self.extra_layers = extra_layers
        self.default_topology = default_topology or {}
    
    @staticmethod
    def topology_tuple_to_dict(top):
        return {
            "layers" : top[0] * top[1],
            "filters": top[2],
            "dilations": top[3],
            "padding": "same" if len(top) < 5 else top[4]
        }

    @staticmethod
    def topology_tuple_to_str(top):
        fd = [','.join(map(str, x)) for x in top[2:4]]
        res = f"{top[0]}x{top[1][0]}x{fd[0]}x{fd[1]}"
        if len(top) == 4: return res + "-same"
        if isinstance(top[4], str): return f"{res}-{top[4][0]}"
        pad = ''.join(map(lambda x: x[0], top[4]))
        return f"{res}x{pad}"
    
    @staticmethod
    def build_net(layer_func, layers, dilations, filters=None, input_layer=None, prefix=None, **constants):
        if filters is None:
            filters = len(layers) * [2]
        assert len(layers) == len(dilations)
        if input_layer is None:
            input_layer = "data"
        prefix = prefix or "gated"
        res = {}
        for i, (width, dil, fil) in enumerate(zip(layers, dilations, filters)):
            layer_name = f"{prefix}_{i}"
            kwargs = dict(width=width, dilation=dil, filter_size=fil, input=input_layer)
            kwargs.update(**constants)
            res[layer_name] = layer_func(**kwargs)
            input_layer = layer_name
        return res, input_layer
    
    def apply_topology(self, config, **topology):
        self.set_topology(config, topology)
    
    def set_topology(self, config, topology):
        topology = {**self.default_topology, **topology}
        old_net = config['network']
        hidden_layers = list(filter(lambda x: x.startswith('tdnn'), old_net.keys()))
        for layer in hidden_layers:
            del old_net[layer]
        # create network
        old_net['input_conv'] = conv_layer(
            1700, 1, 'relu', 5, input='data', forward_weights_init=None
        )
        network, output = ConfigBuilder.build_net(self.layer_func, **topology, input_layer="input_conv")
        if "extra_layers" in topology:
            from collections import OrderedDict
            extra_layer_top_dict = topology["extra_layers"]
            assert isinstance(extra_layer_top_dict, OrderedDict)
            for layer_func, (prefix, top) in zip(self.extra_layers, extra_layer_top_dict.items()):
                extra_net, output = ConfigBuilder.build_net(layer_func, **top, input_layer=output, prefix=prefix)
                network.update(extra_net)
        config['network'].update(network)
        config['network']['output']['from'] = output
    
    def __getattr__(self, name):
        if name.startswith("set_"):
            constant_name = name[4:]
            def setter(config, constant):
                topology = copy.deepcopy(self.default_topology)
                topology[constant_name] = constant
                self.set_topology(config, topology)
            return setter
        raise AttributeError(name)
    
    def build(self, config, layer_sequence=None, layer_constructors=None, layer_func=None, **topology):
        layer_sequence_wo_bottleneck = assign_bottleneck(layer_sequence)
        assert not (layer_constructors and layer_func), "layer func and layer stack given, please decide for exactly one"
        layer_func = ConfigBuilder.create_layer_func_from_stack(layer_constructors, layer_sequence_wo_bottleneck)
        self.layer_func = layer_func
        self.set_topology(config, topology)
    
    def set_layer_func(self, layer_func=None, **create_args):
        if layer_func is not None:
            self.layer_func = layer_func
            return
        self.layer_func = self.create_layer_func_from_stack(
            **create_args
        )

    @staticmethod
    def create_layer_func_from_stack(
            layer_constructors: dict,
            layer_sequence: list,
            auto_add_layers=True
        ):
        if auto_add_layers:
            auto_add_layers = ("projection", "output")
        def layer_func(width, dilation, input, bottleneck, filter_size=None, **kwargs):
            conv = layer_constructors["conv"]
            layer_constructors["conv"] = partial(conv, dilation=dilation, filter_size=filter_size)
            res = {}
            src = "data"
            last_width = None
            for layer_name in layer_sequence:
                w = width
                if layer_name.split("_")[-1] == "b":
                    layer_name = layer_name[:-2]
                    w = bottleneck
                res[layer_name] = layer_constructors[layer_name](src, w)
                src = layer_name
                last_width = res[src].get("n_out", last_width)
            if src == "gating": last_width /= 2
            for ln in auto_add_layers:
                res[ln] = layer_constructors[ln](src, last_width)
            # pprint(res)
            return {"class": "subnetwork", "subnetwork": res, "from": [input]}
        return layer_func

def assign_bottleneck(seq):
    res = []
    for s in seq:
        if s == "bottleneck":
            res[-1] += "_b"
            continue
        res.append(s)
    return res

layer_funcs = {
    "conv"      : lambda src, width, dilation, filter_size: conv_layer(width, dilation, input=src, activation=None, filter_size=filter_size),
    "gating"    : lambda src, width: { "class": "gating", "activation": "tanh", "from": [src] },
    "linear"    : lambda src, width: { "class": "linear", "activation": None, "from": [src], "n_out": width},
    "projection": lambda src, width: { "class": "linear", "activation": None, "from": ["data"], "n_out": width},
    "output"    : lambda src, width: { "class": "combine", "kind": "add", "from": ["projection", src]}
}

def make_baseline(num_input):
    from .configs import blstm_config
    from .lstm import BASE_VITERBI_LRS, BASE_CRNN_CONFIG
    from i6_core import returnn
    crnn_kwargs = BASE_CRNN_CONFIG.copy()
    del crnn_kwargs["l2"], crnn_kwargs["dropout"]
    config = blstm_config(
        num_input,
        network={"output": {"class": "softmax"}},
        learning_rate=BASE_VITERBI_LRS["lr"],
        learning_rates=BASE_VITERBI_LRS["learning_rates"],
        **crnn_kwargs
    )
    base_config = config.config
    base_config['network']['output']['loss_opts'] = {"focal_loss_factor": 2.0}
    base_config["network"]["output"]["loss"] = "ce"

    reduced_topology = (6, [1700], 5 * [2] + [1], [1, 2, 4, 8, 16, 1])
    default_layer_sequence = ("conv", "gating", "linear", "bottleneck")
    layer_factory = ConfigBuilder.create_layer_func_from_stack(
        layer_funcs,
        assign_bottleneck(default_layer_sequence),
    )
    fbuilder = ConfigBuilder(layer_factory)
    fbuilder.default_topology["bottleneck"] = 200
    fbuilder.set_topology(base_config, ConfigBuilder.topology_tuple_to_dict(reduced_topology))

    return config


