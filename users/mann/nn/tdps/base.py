from sisyphus import *

import copy
import enum

from i6_core import returnn
from .. import bw
from ..inspect import InspectTFCheckpointJob
from i6_core.rasr import RasrCommand
from i6_experiments.users.mann.setups.tdps import SimpleTransitionModel

learning_rate_update = dict(
    learning_rate = 0.0008,
    learning_rate_control = "newbob_abs",
    learning_rate_control_error_measure = 'dev_score_output/output_tdps',
    learning_rate_control_min_num_epochs_per_new_lr = 4,
    newbob_learning_rate_decay = 0.9,
    newbob_error_threshold = -10.0,
    gradient_clip = 10.0,
    cleanup_old_models = {'keep': [12, 24, 32, 80, 160, 240, 320]}
)

def make_non_trainable(config):
    for layer in config["network"].values():
        if not layer["class"] == "hidden":
            continue
        layer["trainable"] = False
    config["network"]["output"]["trainable"] = False
    config.update(learning_rate_update)

class BaseTdpModel:
    def __init__(
        self,
        tdp_layer,
        output_layer,
        init_tdps=None,
        imports=None
    ):
        self.tdp_layer = tdp_layer
        self.output_layer = output_layer
        self.init_tdps = init_tdps
        self.imports = imports

    def set_config(self, config: returnn.ReturnnConfig):
        net = config.config["network"]
        net["tdps"] = layer = self.tdp_layer
        if self.init_tdps is not None:
            self.init_tdps.apply(layer, config)
        net["fast_bw"]["tdps"] = "tdps"
        net["output_tdps"] = self.output_layer
        config.maybe_add_dependencies(self.imports)

class TdpModelBuilder:
    def __init__(
        self,
        tdp_layer_func,
        output_layer_func,
        initializer=None,
        imports=None
    ):
        self.tdp_layer_func = tdp_layer_func
        self.output_layer_func = output_layer_func
        self.tdp_initializer = initializer
        self.imports = imports
    
    def build(self, num_classes, init_tdps=None, init_args=None, **kwargs):
        assert (init_args is None) or (init_tdps is None)
        if init_args:
            init_args = init_args.copy()
            type_ = init_args.pop("type")
            init_tdps = self.tdp_initializer(num_classes, type_, init_args)
        return BaseTdpModel(
            tdp_layer=self.tdp_layer_func(num_classes),
            output_layer=self.output_layer_func(num_classes),
            init_tdps=init_tdps,
            imports=self.imports,
        )

def bw_tdps_output():
    default_arch = {
        'class': 'eval', 'out_type': {'batch_dim_axis': 0, 'shape': (211, 2)},
        'eval': '- source(0, auto_convert=False, as_data=True).copy_add_batch_dim(0).placeholder', 
        'from': 'tdps_normed',
        'loss': 'via_layer', 'loss_opts': {'error_signal_layer': 'fast_bw/tdps'}
    }
    pass

def get_simple_tdps(num_classes, silence_idx, speech_fwd, silence_fwd):
    import numpy as np
    tdp_array = np.full((num_classes,), speech_fwd)
    tdp_array[silence_idx] = silence_fwd
    return tdp_array.view(SimpleEqualityArray)

def read_model_from_checkpoint(
        chkpt: returnn.training.Checkpoint
    ) -> SimpleTransitionModel:
    tensors = InspectTFCheckpointJob(
        chkpt, 
        all_tensors=True,
    ).tensors
    tdps = {}
    for s in ["speech", "silence"]:
        tdps[s] = tensors[f"tdps/{s}/{s}"]
    return SimpleTransitionModel.from_log_probs(**tdps)
