from i6_core import returnn
from i6_core.returnn import CodeWrapper

from sisyphus.delayed_ops import DelayedFormat

class DelayedCodeWrapper(DelayedFormat):
    def get(self):
        return CodeWrapper(super().get())

def get_simple_tdps(num_classes, silence_idx, speech_fwd, silence_fwd, as_logit=False):
    import numpy as np
    tdp_array = np.full((num_classes,), speech_fwd)
    tdp_array[silence_idx] = silence_fwd
    if as_logit:
        tdp_array = np.log(tdp_array / (1 - tdp_array))
    return tdp_array.view(SimpleEqualityArray)

def set_init(tdp_layer, init_tdps):
    tdp_layer["subnetwork"]["fwd_prob"].update({
        "bias_init": init_tdps,
        "forward_weights_init": 0,
    })
    return tdp_layer

class InitializerBuilder:
    def __init__(self, tdp_getter, init_setter):
        self.tdp_getter = tdp_getter
        self.init_setter = init_setter
    
    def build_tdp_code(self, num_classes, silence_idx, speech_fwd, silence_fwd, as_logit):
        fmt_string = "{}({}, {}, {}, {})"
        if as_logit:
            fmt_string = "{}({}, {}, {}, {}, as_logit=True)"
        return DelayedCodeWrapper(
            fmt_string,
            self.tdp_getter.__name__,
            num_classes, silence_idx, speech_fwd, silence_fwd, 
        )
    
    def build(self, num_classes, silence_idx, speech_fwd, silence_fwd, as_logit=False):
        return TdpInitializer(
            tdp_getter=self.tdp_getter,
            init_setter=self.init_setter,
            tdp_code=self.build_tdp_code(
                num_classes, silence_idx, speech_fwd, silence_fwd, as_logit
            )
        )

class TdpInitializer:
    def __init__(self, tdp_getter, init_setter, tdp_code):
        self.tdp_getter = tdp_getter
        self.init_setter = init_setter
        self.tdp_code = tdp_code
    
    def apply(self, tdp_layer: dict, config: returnn.ReturnnConfig):
        self.init_setter(tdp_layer, self.tdp_code)
        config.maybe_add_dependencies(self.tdp_getter)


INITIALIZATION_BUILDERS = {
    "simple_tdp_init": InitializerBuilder(
        tdp_getter=get_simple_tdps,
        init_setter=set_init
    )
}