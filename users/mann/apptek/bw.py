from sisyphus import tk, delayed_ops

import numpy as np
from typing import List, Optional

from i6_core.returnn import ReturnnConfig, CodeWrapper
from i6_core.rasr import (
    CommonRasrParameters,
    crp_add_default_output,
    crp_set_corpus,
    build_config_from_mapping,
    WriteRasrConfigJob,
    RasrConfig,
)
from i6_core.meta import CorpusObject
from i6_core.am import acoustic_model_config

_DEBUG = False

class BwConfigBuilder:
    """
    Helper class for building a ReturnnConfig for fast_bw training given an existing ReturnnConfig.
    Multiple outputs can be handled.
    """

    def __init__(
        self,
        returnn_config,
        nn_rasr_trainer_exe,
        fastbw_config,
        num_classes,
        tdp_scale=0.3,
        am_scale=0.3,
    ):
        self.returnn_config = returnn_config
        self.nn_rasr_trainer_exe = nn_rasr_trainer_exe
        self.fastbw_config = fastbw_config
        self.num_classes = num_classes
        self.tdp_scale = tdp_scale
        self.am_scale = am_scale
    
    @property
    def net(self):
        return self.returnn_config.config["network"]

    def add_fastbw_layer(
        self,
        name="fast_bw",
        input_layer="output",
    ):
        self.net[name] = {
            "class": "fast_bw",
            "from": [input_layer],
            "align_target": "sprint",
            "tdp_scale": self.tdp_scale,
            "sprint_opts": {
                "sprintExecPath": self.nn_rasr_trainer_exe,
                "sprintConfigStr": delayed_ops.DelayedFormat(
                    "--config={}", self.fastbw_config
                ),
                "sprintControlConfig": {"verbose": True},
                "usePythonSegmentOrder": False,
                "numInstances": 1,
            },
            "input_type": "prob",
            "am_scale": self.am_scale,
        }
    
    def add_loss_layer(
        self,
        name="output_bw",
        bw_target_layer="fast_bw",
        input_layer="output",
    ):
        self.net[name] = {
            "class": "copy",
            "from": input_layer,
            "loss": "via_layer",
            "loss_opts": {"loss_wrt_to_act_in": "softmax", "align_layer": bw_target_layer}
        }
    
    def remove_layer(self, name):
        pass
    
    def remove_ce_loss(self, layer="output"):
        layer_dict = self.net[layer]
        for key in list(layer_dict.keys()):
            if key.startswith("loss"):
                del layer_dict[key]
        layer_dict.pop("target", None)
        layer_dict["n_out"] = self.num_classes
    
    def find_output_layers(self) -> List[str]:
        output_layers = []
        for name, layer in self.net.items():
            if layer.get("loss", None) == "ce":
                output_layers.append(name)
        
        if len(output_layers) == 0:
            raise ValueError("No output layer found")
        
        return output_layers

    def find_layer_with_input(self, input_layer):
        for layer_name, layer in self.net.items():
            if layer.get("from") == input_layer:
                return layer_name
        raise KeyError(f"Could not find layer with input {input_layer}")

    def skip_layer(self, layer_name):
        input_layer = self.net[layer_name]["from"]
        assert isinstance(input_layer, str)
        next_layer = self.find_layer_with_input(layer_name)
        self.net[next_layer]["from"] = input_layer
        del self.net[layer_name]

    def safe_remove_mask_layers(self):
        found = False
        for layer_name in list(self.net.keys()):
            if layer_name.startswith("masked_tconv"):
                self.skip_layer(layer_name)
                found = True
        if not found:
            raise ValueError("Could not find masked_tconv layer")


def make_crp(
    bliss_corpus: tk.Path,
    bliss_lexicon: tk.Path,
    state_tying: dict,
    speech_fwd_probability: float = 1/3,
    silence_fwd_probability: float = 1/40,
):
    crp = CommonRasrParameters()
    crp_add_default_output(crp)

    # set corpus
    corpus_object = CorpusObject()
    corpus_object.corpus_file = bliss_corpus
    crp_set_corpus(crp, corpus_object)

    # set lexicon
    crp.lexicon_config = RasrConfig()
    crp.lexicon_config.file = bliss_lexicon
    crp.lexicon_config.normalize_pronunciation = False

    # set acoustic model config
    crp.acoustic_model_config = acoustic_model_config(
        state_tying=state_tying["type"],
        tdp_transition=(
            -np.log(1 - speech_fwd_probability),
            -np.log(speech_fwd_probability),
            "infinity", 0.0
        ),
        tdp_silence=(
            -np.log(1 - silence_fwd_probability),
            -np.log(silence_fwd_probability),
            "infinity", 0.0
        )
    )
    crp.acoustic_model_config.state_tying.file = state_tying.get("file", None)
    if "extra_config" in state_tying:
        crp.acoustic_model_config.state_tying._update(state_tying["extra_config"])
    return crp


def make_fastbw_rasr_config(
	bliss_corpus: tk.Path,
	bliss_lexicon: tk.Path,
	state_tying: dict,
	speech_fwd_probability: float = 1/3,
	silence_fwd_probability: float = 1/40,
):
    """Generate a RASR config file for fastbw training."""
    crp = make_crp(
        bliss_corpus=bliss_corpus,
        bliss_lexicon=bliss_lexicon,
        state_tying=state_tying,
        speech_fwd_probability=speech_fwd_probability,
        silence_fwd_probability=silence_fwd_probability,
    )
    crp.acoustic_model_config.fix_allophone_context_at_word_boundaries = True
    crp.acoustic_model_config.transducer_builder_filter_out_invalid_allophones = True
    crp.acoustic_model_config.applicator_type = "corrected"

    if _DEBUG:
        crp.acoustic_model_config.fix_tdp_leaving_epsilon_arc = True

    mapping = {
        'corpus': 'neural-network-trainer.corpus',
        'lexicon': 'neural-network-trainer.alignment-fsa-exporter.model-combination.lexicon',
        'acoustic_model': 'neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model',
    }

    config, post_config = build_config_from_mapping(crp, mapping)
    post_config['*'].output_channel.file = 'fastbw.log'

    # Define action
    config.neural_network_trainer.action = 'python-control'

    # neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder
    orth_parser_config = config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser
    orth_parser_config.allow_for_silence_repetitions = False
    orth_parser_config.normalize_lemma_sequence_scores = False

    write_rasr_config = WriteRasrConfigJob(config, post_config)
    return write_rasr_config.out_config


def add_bw_loss(
	returnn_config: ReturnnConfig,
	bliss_corpus: tk.Path,
	bliss_lexicon: tk.Path,
    nn_rasr_trainer_exe: tk.Path,
	state_tying: dict,
    num_classes: int,
	speech_fwd_probability: float = 1/3,
	silence_fwd_probability: float = 1/40,
	reuse_bw_alignment: bool = True,
    remove_mask_layers: bool = True, 
):
    """
    Adds a bw loss to an existing RETURNN config given a corpus and lexicon file.
    A state-tying has to be passed to the function as a dict of format:
        {
            "type": [...],
            "path": [...],
        }
    If the neural network has multiple output layers, e.g. for applying
    a loss to multiple layers inside the network, one can choose to
    compute only one bw alignment and apply it to all output layers instead
    of computing bw alignments for each layer.

    :param ReturnnConfig returnn_config:
    :param Path bliss_lexicon: path to training bliss lexicon.
    :param Path bliss_corpus: path to bliss corpus containing all relevant
    segments for training
    :param Path nn_rasr_trainer_exe: path to RASR nn trainer binary
    :param dict state_tying: a dictionary containing the state-tying type
    and a path to the CART file if the type is CART
    :param float speech_fwd_probability: speech forward probability
    :param float silence_fwd_probability: silence forward probability
    :param bool reuse_bw_alignment: whether only compute one bw alignment and
    apply it to all output layers
    :param bool remove_mask_layers: necessary for conformers

    :return: RETURNN config with bw loss added
    """
    # build rasr config
    fastbw_config = make_fastbw_rasr_config(
        bliss_corpus=bliss_corpus,
        bliss_lexicon=bliss_lexicon,
        state_tying=state_tying,
        speech_fwd_probability=speech_fwd_probability,
        silence_fwd_probability=silence_fwd_probability,
    )

    # transform config
    config_builder = BwConfigBuilder(
        returnn_config=returnn_config,
        fastbw_config=fastbw_config,
        nn_rasr_trainer_exe=nn_rasr_trainer_exe,
        num_classes=num_classes,
    )

    output_layers = config_builder.find_output_layers()
    assert "output" in returnn_config.config["network"], "No output layer found in RETURNN config."

    if remove_mask_layers:
        config_builder.safe_remove_mask_layers()

    if reuse_bw_alignment or len(output_layers) == 1:
        config_builder.add_fastbw_layer()
        for layer in output_layers:
            config_builder.remove_ce_loss(layer)
            config_builder.add_loss_layer(
                name="%s_bw" % layer,
                input_layer=layer,
            )
        return config_builder.returnn_config
    
    for layer in output_layers:
        config_builder.remove_ce_loss(layer)
        fastbw_layer_name="%s_fast_bw" % layer
        config_builder.add_fastbw_layer(
            name=fastbw_layer_name,
            input_layer=layer,
        )
        config_builder.add_loss_layer(
            name="%s_bw" % layer,
            input_layer=layer,
            bw_target_layer=fastbw_layer_name,
        )
    return config_builder.returnn_config


def make_uniform_label_posterior_net(network):
    network["uniform_label_posterior"] = {
        "class": "eval",
        "from": "output",
        "eval": "tf.zeros_like(source(0), dtype=tf.float32)",
    }
    for layer in network.values():
        if layer["class"] != "fast_bw": continue
        layer["from"] = "uniform_label_posterior"
        del layer["input_type"] # defaults to "log_prob"
    return network


def uniform_label_pretrain_construction_algo(idx, net_dict):
    if idx != 0: return None
    return make_uniform_label_posterior_net(net_dict)


def add_uniform_pretrain(
    returnn_config: ReturnnConfig,
    num_pretrain_epochs: int,
    learning_rate_copy: bool = False,
    pretrain_learning_rate: Optional[float] = None,
):
    new_config = ReturnnConfig(
        config={
            "pretrain": {
                "repetitions": {
                    "default": num_pretrain_epochs,
                    "final": 1,
                },
                "construction_algo": CodeWrapper(uniform_label_pretrain_construction_algo.__name__)
            }
        },
        python_prolog=(make_uniform_label_posterior_net, uniform_label_pretrain_construction_algo)

    )
    if pretrain_learning_rate:
        new_config.config["pretrain_learning_rate"] = pretrain_learning_rate
    
    if learning_rate_copy:
        new_config.config["use_learning_rate_control_always"] = True
        lrs = returnn_config.config["learning_rates"]
        new_config.config["learning_rates"] = lrs[:num_pretrain_epochs] + lrs

    returnn_config.update(new_config)

    return returnn_config
