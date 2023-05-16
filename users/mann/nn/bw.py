import copy

import numpy as np

from sisyphus import tk, delayed_ops

from i6_core import returnn
from i6_core.rasr import RasrCommand, RasrConfig, WriteRasrConfigJob
# from .inspect import InspectTFCheckpointJob
from i6_experiments.users.mann.setups.tdps import SimpleTransitionModel

from .util import maybe_add_dependencies, DelayedCodeWrapper

def add_bw_layer(csp, crnn_config, am_scale=1.0, ce_smoothing=0.0,
                 exp_average=0.001, 
                 prior_scale=1.0, tdp_scale=1.0,
                 chunking=False, keep_ce=False, num_classes=None,
                 prior=True,
                 ):
    crnn_config = crnn_config.config
    # maybe remove chunking
    if chunking:
        raise NotImplementedError("Chunking not implemented yet.")
    crnn_config.pop("chunking", None)

    # Prepare output layer to compute sequence loss
    assert crnn_config['use_tensorflow']

    am_output_layer = "output"
    if prior:
        crnn_config['network']['accumulate_prior'] = {}
        crnn_config['network']['accumulate_prior']['class'] = "accumulate_mean"
        crnn_config['network']['accumulate_prior']['from'] = ['output']
        crnn_config['network']['accumulate_prior']["exp_average"] =  exp_average
        crnn_config['network']['accumulate_prior']["is_prob_distribution"] = True

        crnn_config['network']['combine_prior'] = {}
        crnn_config['network']['combine_prior']['class'] = 'combine'
        crnn_config['network']['combine_prior']['from'] = ['output', 'accumulate_prior']
        crnn_config['network']['combine_prior']['kind']  = 'eval'
        crnn_config['network']['combine_prior']['eval']  = "safe_log(source(0)) * am_scale - safe_log(source(1)) * prior_scale"
        crnn_config['network']['combine_prior']['eval_locals'] = {'am_scale': am_scale, 'prior_scale': prior_scale}
        am_output_layer = "combine_prior"

    crnn_config['network']['fast_bw'] = {}
    crnn_config['network']['fast_bw']['class'] = 'fast_bw'
    crnn_config['network']['fast_bw']['from']  = [am_output_layer]
    crnn_config['network']['fast_bw']['align_target'] = 'sprint'
    crnn_config['network']['fast_bw']['tdp_scale'] = tdp_scale
    crnn_config['network']['fast_bw']['sprint_opts'] = {
        "sprintExecPath":       RasrCommand.select_exe(csp.nn_trainer_exe, 'nn-trainer'),
        "sprintConfigStr":      "--config=fastbw.config",
        "sprintControlConfig":  {"verbose": True},
        "usePythonSegmentOrder": False,
        "numInstances": 1}
    if not prior:
        crnn_config['network']['fast_bw']['input_type'] = "prob"
        crnn_config['network']['fast_bw']['am_scale'] = am_scale

    crnn_config['network']['output_bw'] = {'class': 'copy', 'from': 'output'}
    crnn_config['network']['output_bw']['loss']       = 'via_layer'
    crnn_config['network']['output_bw']['loss_opts']  = {"loss_wrt_to_act_in": "softmax", "align_layer": "fast_bw"}

    if not keep_ce:
        assert isinstance(num_classes, (int, tk.Variable, delayed_ops.DelayedBase))
        del crnn_config['network']['output']['loss']
        del crnn_config['network']['output']['loss_opts']
        crnn_config['network']['output']['n_out'] = num_classes
    else:
        crnn_config["network"]["output"]["loss_scale"] = 0.0


class ScaleConfig(returnn.ReturnnConfig):
    @classmethod
    def from_config(cls, config):
        if isinstance(config, returnn.ReturnnConfig):
            res = copy.deepcopy(config)
            res.__class__ = cls
            return res
        return cls.from_config(returnn.ReturnnConfig(config))
    
    @classmethod
    def copy_add_bw(cls, config, csp, **bw_args):
        config = copy.deepcopy(config)
        add_bw_layer(csp, config, **bw_args)
        return cls.from_config(config)
    
    def maybe_add_dependencies(self, *dependencies):
        maybe_add_dependencies(self, *dependencies)
  
    def set_prior_scale(self, prior_scale):
        assert isinstance(prior_scale, (int, float, returnn.CodeWrapper))
        # if self.is_prior_relative:
        #     prior_scale = self.am_scale * prior_scale
        self.config["network"]["combine_prior"]["eval_locals"]["prior_scale"] = prior_scale

    def get_prior_scale(self):
        return self.config["network"]["combine_prior"]["eval_locals"]["prior_scale"]
    
    prior_scale = property(get_prior_scale, set_prior_scale)

    def set_am_scale(self, am_scale):
        assert isinstance(am_scale, (int, float, returnn.CodeWrapper))
        self.config["network"]["combine_prior"]["eval_locals"]["am_scale"] = am_scale

    def get_am_scale(self):
        return self.config["network"]["combine_prior"]["eval_locals"]["am_scale"]
    
    am_scale = property(get_am_scale, set_am_scale)

    def get_tdp_scale(self):
        return self.config["network"]["fast_bw"]["tdp_scale"]
    
    def set_tdp_scale(self, tdp_scale):
        assert isinstance(tdp_scale, (int, float))
        self.config["network"]["fast_bw"]["tdp_scale"] = tdp_scale
        return self
    
    tdp_scale = property(get_tdp_scale, set_tdp_scale)

    def set_tdps(self, tdps: SimpleTransitionModel):
        tdp_log_prob = tdps.to_log_probs()
        tdp_unit = self["network"]["tdps"]["subnetwork"]
        # self.tdps = tdps_log_prob
        for phon in ["speech", "silence"]:
            values = getattr(tdp_log_prob, phon).values
            tdp_unit[phon]["init"] = returnn.CodeWrapper(
                "np.array([{}, {}])".format(*values)
            )
        self.maybe_add_dependency("import numpy as np")
    
    def set_rasr_config(self, config, post_config, inplace=False, save_under=None):
        assert isinstance(config, RasrConfig) and isinstance(post_config, RasrConfig)
        if inplace:
            new_config = self
        else:
            new_config = copy.deepcopy(self)
        config_file = WriteRasrConfigJob(config, post_config).out_config
        if save_under:
            tk.register_output(f"nn_configs/{save_under}/fastbw.config", config_file)
        config_str = delayed_ops.DelayedFormat("--config={}", config_file)
        if "network" in new_config.config:
            new_config.config["network"]["fast_bw"]["sprint_opts"]["sprintConfigStr"] = config_str
        else:
            assert new_config.staged_network_dict
            for net in new_config.staged_network_dict.values():
                net["fast_bw"]["sprint_opts"]["sprintConfigStr"] = config_str
        return new_config

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

def read_tdps_from_checkpoint(chkpt):
    tensors = InspectTFCheckpointJob(
        chkpt, 
        all_tensors=True,
    ).tensors
    tdps = {}
    from itertools import product
    for s, (i, d) in product(["speech", "silence"], enumerate(["f", "l"])):
        tdps[s[:2] + d] = 0 - tensors[f"tdps/{s}/{s}"][i]
    return tdps

def read_tdps_from_checkpoint_raw(chkpt):
    tensors = InspectTFCheckpointJob(
        chkpt, 
        all_tensors=True,
    ).tensors
    tdps = {}
    # from itertools import product
    # for s, (i, d) in product(["speech", "silence"], enumerate(["f", "l"])):
    #     tdps[s[:2] + d] = 0 - tensors[f"tdps/{s}/{s}"][i]
    for s in ["speech", "silence"]:
        tdps[s] = tensors[f"tdps/{s}/{s}"]
    return tdps

def weights_to_prob(tdps):
    speech = np.array([tdps["spf"], tdps["spl"]])
    silence = np.array([tdps["sif"], tdps["sil"]])
    speech = np.exp(-speech)
    silence = np.exp(-silence)
    speech = speech / np.sum(speech, keepdims=True)
    silence = silence / np.sum(silence, keepdims=True)
    return silence, speech


