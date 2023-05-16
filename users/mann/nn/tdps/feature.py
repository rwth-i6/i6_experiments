from .base import BaseTdpModel, TdpModelBuilder
from ..util import DelayedCodeWrapper

TDP_FFNN_LAYER = lambda num_classes: {
    "class": "subnetwork",
    "from": ["fwd_6", "bwd_6"],
    "subnetwork": {
        "fwd_prob": {
            "class": "linear",
            "activation": "log_sigmoid",
            "n_out": num_classes,
        },
        "loop_prob": {
            "class": "eval",
            "eval": "safe_log(1 - tf.exp(source(0)))",
            "from": ["fwd_prob"],
        },
        "output": {
            "class": "stack",
            "from": ["fwd_prob", "loop_prob"],
            "axis": -1,
        }
    }
}

BLSTM_CONFIG = {
    "dropout": 0.1,
    "L2": 0.0001
}

def TDP_BLSTM_NO_LABEL_LAYER(num_classes, n_out=512):
    return {
        "class": "subnetwork",
        "from": ["fwd_6", "bwd_6"],
        "subnetwork": {
            "lstm_fwd": {
                'class'     : 'rec',
                'unit'      : 'lstmp',
                'direction' : 1,
                'n_out'     : n_out,
                **BLSTM_CONFIG,
            },
            "lstm_bwd": {
                'class'     : 'rec',
                'unit'      : 'lstmp',
                'direction' : -1,
                'n_out'     : n_out,
                **BLSTM_CONFIG,
            },
            "fwd_prob": {
                "activation": "log_sigmoid",
                "class": "linear",
                "from": ["lstm_fwd", "lstm_bwd"],
                "n_out": 1,
            },
            "loop_prob": {
                "class": "eval",
                "eval": "safe_log(1 - tf.exp(source(0)))",
                "from": ["fwd_prob"],
            },
            "output": {
                "class": "expand_dims",
                "from": ["fwd_prob", "loop_prob"],
                "axis": "spatial",
                "dim": num_classes,
            },
        }
    }

TDP_BLSTM_NO_LABEL_LAYER_SMALL = lambda num_classes: TDP_BLSTM_NO_LABEL_LAYER(num_classes, 1)

TDP_SIGMOID_NO_LABEL = lambda num_classes: {
    "for_all_labels": {
        "class": "tile",
        "multiples": {"F": num_classes},
        "from": ["fwd_prob"],
    },
    "loop_prob": {
        "class": "eval",
        "eval": "safe_log(1 - tf.exp(source(0)))",
        "from": ["for_all_labels"],
    },
    "output": {
        "class": "stack",
        "from": ["for_all_labels", "loop_prob"],
        "axis": -1,
    }
}

TDP_SIGMOID = {
    "loop_prob": {
        "class": "eval",
        "eval": "safe_log(1 - tf.exp(source(0)))",
        "from": ["fwd_prob"],
    },
    "output": {
        "class": "stack",
        "from": ["fwd_prob", "loop_prob"],
        "axis": -1,
    }
}

TPD_BLSTM_NO_LABEL_SIGMOID_LAYER = lambda num_classes: {
    "class": "subnetwork",
    "from": ["fwd_6", "bwd_6"],
    "subnetwork": {
        "lstm_fwd": {
            'class'     : 'rec',
            'unit'      : 'lstmp',
            'direction' : 1,
            'n_out'     : 1,
            **BLSTM_CONFIG,
        },
        "lstm_bwd": {
            'class'     : 'rec',
            'unit'      : 'lstmp',
            'direction' : -1,
            'n_out'     : 1,
            **BLSTM_CONFIG,
        },
        "fwd_prob": {
            "class": "linear",
            "activation": "log_sigmoid",
            "from": ["lstm_fwd", "lstm_bwd"],
            "n_out": 1,
        },
        **TDP_SIGMOID_NO_LABEL(num_classes),
    }
}

def TDP_BLSTM_LAYER_SIGMOID(num_classes, n_out=512):
    return {
        "class": "subnetwork",
        "from": ["fwd_6", "bwd_6"],
        "subnetwork": {
            "lstm_fwd": {
                'class'     : 'rec',
                'unit'      : 'lstmp',
                'direction' : 1,
                'n_out'     : n_out,
                **BLSTM_CONFIG,
            },
            "lstm_bwd": {
                'class'     : 'rec',
                'unit'      : 'lstmp',
                'direction' : -1,
                'n_out'     : n_out,
                **BLSTM_CONFIG,
            },
            "fwd_prob": {
                "class": "linear",
                "activation": "log_sigmoid",
                "from": ["lstm_fwd", "lstm_bwd"],
                "n_out": num_classes,
            },
            **TDP_SIGMOID.copy(),
        }
    }

#--------------------------------------- initializer ----------------------------------------------

def get_simple_tdps(num_classes, silence_idx, speech_fwd, silence_fwd, as_logit=False):
    import numpy as np
    tdp_array = np.full((num_classes,), speech_fwd)
    tdp_array[silence_idx] = silence_fwd
    if as_logit:
        tdp_array = np.log(tdp_array / (1 - tdp_array))
    return tdp_array

class FeatureInitializer:

    def __init__(self, n_classes, init, init_args):
        self.n_classes = n_classes
        self.init = init
        self.init_args = init_args
    
    def build_tdp_code(self, num_classes, silence_idx, speech_fwd, silence_fwd, as_logit=True):
        fmt_string = "{}({}, {}, {}, {})"
        if as_logit:
            fmt_string = "{}({}, {}, {}, {}, as_logit=True)"
        return DelayedCodeWrapper(
            fmt_string,
            get_simple_tdps.__name__,
            num_classes, silence_idx, speech_fwd, silence_fwd, 
        )

    def set_layer(self, layer, value):
        layer["subnetwork"]["fwd_prob"].update({
            "bias_init": value,
            "forward_weights_init": 0,
        })
        return None
    
    def set_smart_init(self, layer, silence_idx, speech_fwd, silence_fwd, **_ignored):
        code = self.build_tdp_code(self.n_classes, silence_idx, speech_fwd, silence_fwd)
        self.set_layer(layer, code)
        return get_simple_tdps
    
    def apply(self, layer, config):
        init = self.init
        init_args = self.init_args
        assert init in {"flat", "smart", "random"}, "Unknown init method: {}".format(init)
        if init == "smart":
            assert all(key in init_args for key in {"speech_fwd", "silence_fwd", "silence_idx"})
        else:
            init_args = {}
        extra_imports = {
            "flat": lambda: self.set_layer(layer, 0),
            "smart": lambda: self.set_smart_init(layer, **init_args),
            "random": lambda: None,
        }[init]()
        config.maybe_add_dependencies(extra_imports)
        

TDP_BLSTM_LAYER_SIGMOID_SMALL = lambda num_classes: TDP_BLSTM_LAYER_SIGMOID(num_classes, n_out=1)

FEATURE_ARCHS = {
    'blstm_no_label': TDP_BLSTM_NO_LABEL_LAYER,
    "ffnn": TDP_FFNN_LAYER,
    "blstm_no_label_sigmoid": TPD_BLSTM_NO_LABEL_SIGMOID_LAYER,
    "blstm": TDP_BLSTM_LAYER_SIGMOID,
    "blstm_large": TDP_BLSTM_LAYER_SIGMOID,
}

def get_tdp_layer(num_classes, arch="ffnn"):
    return FEATURE_ARCHS[arch](num_classes)


TDP_OUTPUT_LAYER = {
    "class": "eval",
    "eval": "-source(0)",
    "from": "tdps",
    "loss": "via_layer",
    "loss_opts": {'error_signal_layer': 'fast_bw/tdps'}
}

TDP_OUTPUT_LAYER_AS_FUNC = lambda num_classes: TDP_OUTPUT_LAYER

TDP_OUTPUT_LAYER_W_SOFTMAX = {
    "class": "copy",
    "from": "tdps",
    "loss": "via_layer",
    "loss_opts": {'align_layer': 'fast_bw/tdps', 'loss_wrt_to_act_in': 'log_softmax'}
}

from functools import partial

FEATURE_MODEL_BUILDERS = {
    'ffnn': TdpModelBuilder(TDP_FFNN_LAYER, TDP_OUTPUT_LAYER_AS_FUNC, initializer=FeatureInitializer),
    'blstm_no_label_sigmoid': TdpModelBuilder(TPD_BLSTM_NO_LABEL_SIGMOID_LAYER, TDP_OUTPUT_LAYER_AS_FUNC),
    'blstm': TdpModelBuilder(TDP_BLSTM_LAYER_SIGMOID_SMALL, TDP_OUTPUT_LAYER_AS_FUNC),
    'blstm_no_label': TdpModelBuilder(TDP_BLSTM_NO_LABEL_LAYER_SMALL, TDP_OUTPUT_LAYER_AS_FUNC),
    'blstm_large': TdpModelBuilder(
        TDP_BLSTM_LAYER_SIGMOID,
        TDP_OUTPUT_LAYER_AS_FUNC,
        initializer=FeatureInitializer,
    ),
    'blstm_no_label_large': TdpModelBuilder(TDP_BLSTM_NO_LABEL_LAYER, TDP_OUTPUT_LAYER_AS_FUNC),
}

SOFTMAX_ARCHS = [
    'blstm_no_label',
]

def get_model(num_classes, arch):
    return BaseTdpModel(
        tdp_layer=get_tdp_layer(num_classes, arch),
        output_layer=TDP_OUTPUT_LAYER_W_SOFTMAX if arch in SOFTMAX_ARCHS else TDP_OUTPUT_LAYER,
    )
