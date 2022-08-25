from .base import BaseTdpModel, Arch, TdpModelBuilder
from .feature import TDP_SIGMOID

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


TDP_OUTPUT_LAYER_ADD_BATCH_DIM = lambda num_classes:{
    'class': 'eval', 'out_type': {'batch_dim_axis': 0, 'shape': (num_classes, 2)},
    'eval': '-source(0, auto_convert=False, as_data=True).copy_add_batch_dim(0).placeholder', 
    'from': 'tdps',
    'loss': 'via_layer', 'loss_opts': {'error_signal_layer': 'fast_bw/tdps'}
}

FULL_LABEL_BUILDER = TdpModelBuilder(
    tdp_layer_func=TDP_LAYER,
    output_layer_func=TDP_OUTPUT_LAYER_ADD_BATCH_DIM,
)

LABEL_MODEL_BUILDER = {
    "label": FULL_LABEL_BUILDER,
}
