from .base import BaseTdpModel, TdpModelBuilder
from .feature import TDP_SIGMOID
from ..util import DelayedCodeWrapper

TDP_LAYER = lambda num_classes: {
    "class": "subnetwork",
    "subnetwork" : {
        "fwd_prob_var": {
            "class": "variable",
            "shape": (num_classes,),
            "add_batch_axis": False,
        },
        "fwd_prob": {
            "class": "activation",
            "activation": "log_sigmoid",
            "from": ["fwd_prob_var"],
        },
        **TDP_SIGMOID.copy(),
    }
}

LABEL_ARCHS = {
    "label": TDP_LAYER,
}

def sublabel_tdp_layer(n_vars, selector_code):
    return {
        "class": "subnetwork",
        "subnetwork" : {
            "base_vars": {
                "add_batch_axis": False,
                "class": "variable",
                "shape": (n_vars,),
            },
            "base_fwd_probs": {
                "activation": "log_sigmoid",
                "class": "activation",
                "from": ["base_vars"],
            },
            "var_selector": {"value": selector_code, "class": "constant"},
            "fwd_prob": {"class": "gather", "position": "var_selector", "from": "base_fwd_probs", "axis": "F"},
            **TDP_SIGMOID.copy(),
        }
    }


TDP_OUTPUT_LAYER_ADD_BATCH_DIM = lambda num_classes:{
    'class': 'eval', 'out_type': {'batch_dim_axis': 0, 'shape': (num_classes, 2)},
    'eval': '-source(0, auto_convert=False, as_data=True).copy_add_batch_dim(0).placeholder', 
    'from': 'tdps',
    'loss': 'via_layer', 'loss_opts': {'error_signal_layer': 'fast_bw/tdps'}
}


def factorize_tying_code(n_classes, n_subclasses, div=None, silence_idx=None):
    extra_silence = "{m} * np.eye({n}, dtype=np.int32)[{silence_idxs}]"
    n_vars = n_subclasses
    if isinstance(div, int) and div > 1:
        tmp = "np.array(range({n})) // {div} % {m}"
    else:
        assert div is None
        tmp = "np.array(range({n})) % {m}"
    if silence_idx is not None:
        tmp = " + ".join([tmp, extra_silence])
        n_vars += 1
    return n_vars, DelayedCodeWrapper(tmp, n=n_classes, m=n_subclasses, div=div, silence_idxs=silence_idx)

def speech_silence_code(n_classes, silence_idx=None):
    extra_silence = "np.eye({n}, dtype=np.int32)[{silence_idxs}]"
    return 2, DelayedCodeWrapper(extra_silence, n=n_classes, silence_idxs=silence_idx)

code_builders = {
    "factorize": factorize_tying_code,
    "speech_silence": speech_silence_code,
}


#------------------------------------- initializers -----------------------------------------------

def logit(arr):
    return np.log(arr / (1 - arr))

def get_simple_tdps(num_classes, silence_idx, speech_fwd, silence_fwd, as_logit=False):
    import numpy as np
    tdp_array = np.full((num_classes,), speech_fwd)
    tdp_array[silence_idx] = silence_fwd
    if as_logit:
        tdp_array = np.log(tdp_array / (1 - tdp_array))
    return tdp_array

class FullLabelInitializer:
    def __init__(self, n_classes, init, init_args):
        self.n_classes = n_classes
        self.init = init
        self.init_args = init_args

    def build_tdp_code(self, num_classes, silence_idx, speech_fwd, silence_fwd, as_logit=True):
        assert num_classes is not None
        assert silence_idx is not None
        assert speech_fwd is not None
        assert silence_fwd is not None
        fmt_string = "{}({}, {}, {}, {})"
        if as_logit:
            fmt_string = "{}({}, {}, {}, {}, as_logit=True)"
        return DelayedCodeWrapper(
            fmt_string,
            get_simple_tdps.__name__,
            num_classes, silence_idx, speech_fwd, silence_fwd, 
        )
    
    def set_smart_init(self, layer, silence_idx, speech_fwd, silence_fwd):
        tdp_code = self.build_tdp_code(self.n_classes, silence_idx, speech_fwd, silence_fwd)
        self.set_layer(layer, tdp_code)
        return get_simple_tdps

    def set_layer(self, layer, value):
        layer["subnetwork"]["fwd_prob_var"]["init"] = value
        return None
    
    def apply(self, layer, config):
        init = self.init
        init_args = self.init_args
        assert init in {"flat", "smart", "random"}
        if init == "smart":
            assert all(key in init_args for key in {"speech_fwd", "silence_fwd", "silence_idx"})
        else:
            init_args = {}
        extra_imports = {
            "flat": lambda: None,
            "smart": lambda: self.set_smart_init(layer, **init_args),
            "random": lambda: self.set_layer(layer, {"class": "RandomNormal"}),
        }[init]()
        config.maybe_add_dependencies(extra_imports)

def logit(arr):
    return np.log(arr / (1 - arr))

def get_logit_code(n_subclasses, speech_fwd, silence_fwd, **_ignored):
    fmt_string = "{}(np.array({}))"
    init_arr = [speech_fwd] * n_subclasses + [silence_fwd]
    return DelayedCodeWrapper(fmt_string, logit.__name__, init_arr)

def init_factorize(n_subclasses, silence_idx, speech_fwd, silence_fwd, **_ignored):
    assert silence_idx is not None
    return (
        get_logit_code(n_subclasses, speech_fwd, silence_fwd),
        logit
    )

def init_speech_silence(silence_idx, speech_fwd, silence_fwd, **_ignored):
    assert silence_idx is not None
    return (
        get_logit_code(1, speech_fwd, silence_fwd),
        logit
    )

class SublabelInitializer:
    def __init__(self, n_classes, n_subclasses, init, init_args, init_func, **_ignored):
        self.n_classes = n_classes
        self.n_subclasses = n_subclasses
        self.init = init
        self.init_args = init_args
        self.init_func = init_func
    
    def set_layer(self, layer, value):
        layer["subnetwork"]["base_vars"]["init"] = value
        return None
    
    def set_smart_init(self, layer, silence_idx, speech_fwd, silence_fwd, **_ignored):
        init_code, init_fn = self.init_func(n_subclasses=self.n_subclasses, **self.init_args)
        self.set_layer(layer, init_code)
        return init_fn
    
    def apply(self, layer, config):
        init = self.init
        init_args = self.init_args
        assert init in {"smart", "flat", "random"}
        extra_imports = {
            "flat": lambda: None,
            "random": lambda: self.set_layer(layer, {"class": "RandomNormal"}),
            "smart": lambda: self.set_smart_init(layer, **init_args),
        }[init]()
        config.maybe_add_dependencies(extra_imports)
    

def build_sublabel_initializer(reduce):
    initializer_cls = SublabelInitializer
    init_fn = {
        "factorize": init_factorize,
        "speech_silence": init_speech_silence,
    }[arch]
    initializer_cls.init_fn = init_fn


#------------------------------------- collect builders--------------------------------------------

FULL_LABEL_BUILDER = TdpModelBuilder(
    tdp_layer_func=TDP_LAYER,
    output_layer_func=TDP_OUTPUT_LAYER_ADD_BATCH_DIM,
    initializer=FullLabelInitializer
)

LABEL_MODEL_BUILDER = {
    "label": FULL_LABEL_BUILDER,
}

sublabel_tdp_builders = {
    "factorize": lambda n_subclasses, div=None, silence_idx=None: TdpModelBuilder(
        tdp_layer_func=lambda n: sublabel_tdp_layer(*factorize_tying_code(n, n_subclasses, div, silence_idx)),
        output_layer_func=TDP_OUTPUT_LAYER_ADD_BATCH_DIM,
        imports="import numpy as np",
        initializer=lambda n_classes, init, init_args: SublabelInitializer(
            n_classes, n_subclasses, init, init_args, init_factorize
        )
    ),
    "substate_and_silence": lambda n_subclasses, div=None, silence_idx=None: TdpModelBuilder(
        tdp_layer_func=lambda n: sublabel_tdp_layer(*factorize_tying_code(n, n_subclasses, div, silence_idx)),
        output_layer_func=TDP_OUTPUT_LAYER_ADD_BATCH_DIM,
        imports="import numpy as np",
        initializer=lambda n_classes, init, init_args: SublabelInitializer(
            n_classes, n_subclasses, init, init_args, init_factorize
        )
    ),
    "speech_silence": lambda silence_idx=None, **_ignored: TdpModelBuilder(
        tdp_layer_func=lambda n: sublabel_tdp_layer(*speech_silence_code(n, silence_idx)),
        output_layer_func=TDP_OUTPUT_LAYER_ADD_BATCH_DIM,
        imports="import numpy as np",
        initializer=lambda n_classes, init, init_args: SublabelInitializer(
            n_classes, 1, init, init_args, init_speech_silence
        )
    )
}
