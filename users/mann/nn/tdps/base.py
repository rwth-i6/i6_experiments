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
    def __init__(self, tdp_layer, output_layer):
        self.tdp_layer = tdp_layer
        self.output_layer = output_layer

    def set_config(self, config: returnn.ReturnnConfig):
        net = config.config["network"]
        net["tdps"] = self.tdp_layer
        net["fast_bw"]["tdps"] = "tdps"
        net["output_tdps"] = self.output_layer

class TdpModelBuilder:
    def __init__(self, tdp_layer_func, output_layer_func):
        self.tdp_layer_func = tdp_layer_func
        self.output_layer_func = output_layer_func
    
    def build(self, num_classes):
        return BaseTdpModel(
            tdp_layer=self.tdp_layer_func(num_classes),
            output_layer=self.output_layer_func(num_classes),
        )

class Arch(enum.Enum):
    BLSTM = "rec"
    FFNN = "hidden"
    TDNN = "conv"

    @classmethod
    def recognize(cls, config):
        layer_types = set()
        for layer in config["network"].values():
            value = layer["class"]
            try:
                arch = cls(value)
            except ValueError:
                continue
            layer_types.add(arch)
        assert len(layer_types) == 1, "No layer type found or ambiguous"
        return list(layer_types)[0]

class NonTrainable:
    @classmethod
    def make(cls, arch, config):
        for layer in config["network"].values():
            if not layer["class"] == arch.value:
                continue
            layer["trainable"] = False
        config["network"]["output"]["trainable"] = False


class SimpleTdpsArch:
    default_arch = {
        'class': 'subnetwork',
        'subnetwork': {
            'speech':  {'class': 'variable', 'shape': (2,), 'add_batch_axis': False},
            'silence': {'class': 'variable', 'shape': (2,), 'add_batch_axis': False},
            'silence_id': {'class': 'constant', 'value': 207, 'dtype': 'int32'},
            'silence_mask': { 'class': 'eval',
                                'eval': 'tf.expand_dims(tf.one_hot(source(0, auto_convert=False), 211), -1)',
                                'from': ['silence_id'],
                                'out_type': { 'dim': 1, 'shape': (211, 1), 'dtype': 'float32'}},
            'output': {'class': 'eval', 'eval': 'source(1) * source(0) + source(2) * (1 - source(0))',
                        'from': ['silence_mask', 'silence', 'speech'],
                        'out_type': {'shape': (211, 2), 'dim': 2}
            },
        }
    }

    def __init__(self, updt=None):
        updt = updt or {}
        self.arch = copy.deepcopy(self.default_arch)
        self.arch.update(updt)
    
    def set_config(self, config):
        config["network"]["tdps"] = copy.deepcopy(self.arch)

class TdpNorm:
    default_layer = {
        'class': 'activation', 'activation': 'log_softmax', 'from': 'tdps'
    }

    def __init__(self, updt=None):
        updt = updt or {}
        self.arch = copy.deepcopy(self.default_layer)
        self.arch.update(updt)

    def set_config(cls, config):
        config["network"]["tdps_normed"] = copy.deepcopy(cls.arch)


class BWTdpsOutput:
    default_arch = {
        'class': 'eval', 'out_type': {'batch_dim_axis': 0, 'shape': (211, 2)},
        'eval': '- source(0, auto_convert=False, as_data=True).copy_add_batch_dim(0).placeholder', 
        'from': 'tdps_normed',
        'loss': 'via_layer', 'loss_opts': {'error_signal_layer': 'fast_bw/tdps'}
    }

    def __init__(self, updt=None):
        updt = updt or {}
        self.arch = copy.deepcopy(self.default_arch)
        self.arch.update(updt)
    
    def set_config(self, config):
        config["network"]["output_tdps"] = copy.deepcopy(self.arch)
        config["network"]["fast_bw"]["tdps"] = "tdps_normed"

def bw_tdps_output():
    default_arch = {
        'class': 'eval', 'out_type': {'batch_dim_axis': 0, 'shape': (211, 2)},
        'eval': '- source(0, auto_convert=False, as_data=True).copy_add_batch_dim(0).placeholder', 
        'from': 'tdps_normed',
        'loss': 'via_layer', 'loss_opts': {'error_signal_layer': 'fast_bw/tdps'}
    }
    pass

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


# NonTrainable = object()

class TdpConfig(bw.ScaleConfig):
    @classmethod
    def copy_add_tdps(cls, config, disable_am=True):
        config = copy.deepcopy(config)
        for arch in [SimpleTdpsArch, TdpNorm, BWTdpsOutput]:
            arch().set_config(config)
        if disable_am is NonTrainable:
            arch = Arch.recognize(config)
            disable_am.make(arch, config)
        print(type(config))
        if disable_am:
            config.update(learning_rate_update)
            config.pop("extra_python", None)
            from recipe.crnn.helpers.mann import pretrain
            pretrain.remove_pretrain(config)
        return cls.from_config(config)
