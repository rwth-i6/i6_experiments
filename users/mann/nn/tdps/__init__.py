from .base import *
from .feature import *
from .label import *

_archs = {
    **FEATURE_ARCHS,
    # **LABEL_ARCHS,
}

_builders = {
    **LABEL_MODEL_BUILDER,
}

_softmax_archs = SOFTMAX_ARCHS

def get_tdp_layer(num_classes, arch):
    return _archs[arch](num_classes)
    

def get_model(num_classes, arch):
    try:
        return BaseTdpModel(
            tdp_layer=get_tdp_layer(num_classes, arch),
            output_layer=TDP_OUTPUT_LAYER_W_SOFTMAX if arch in _softmax_archs else TDP_OUTPUT_LAYER,
        )
    except KeyError:
        return _builders[arch].build(num_classes)
