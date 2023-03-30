import copy
from pathlib import Path
from i6_core.am.config import acoustic_model_config
from sisyphus.delayed_ops import DelayedFunction
import i6_core.rasr as rasr
from sisyphus import tk
from typing import Dict, List, Optional, Tuple, Union
from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack
from i6_experiments.users.berger.network.helpers.feature_extraction import (
    add_gt_feature_extraction,
)
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack


def make_ctc_rasr_loss_config_v1(
    loss_corpus_path: tk.Path,
    loss_lexicon_path: tk.Path,
    am_args: Dict,
    add_blank_transition: bool = True,
    allow_label_loop: bool = True,
    blank_label_index: int = -1,  # -1 => replace silence
    extra_config: Optional[rasr.RasrConfig] = None,
    extra_post_config: Optional[rasr.RasrConfig] = None,
):
    # Make crp and set loss_corpus and lexicon
    loss_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(loss_crp)

    loss_crp.corpus_config = rasr.RasrConfig()  # type: ignore
    loss_crp.corpus_config.file = loss_corpus_path  # type: ignore
    loss_crp.corpus_config.remove_corpus_name_prefix = "loss-corpus/"  # type: ignore

    loss_crp.lexicon_config = rasr.RasrConfig()  # type: ignore
    loss_crp.lexicon_config.file = loss_lexicon_path  # type: ignore

    loss_crp.acoustic_model_config = acoustic_model_config(**am_args)  # type: ignore

    # Make config from crp
    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }
    config, post_config = rasr.build_config_from_mapping(
        loss_crp,
        mapping,
        parallelize=False,
    )
    config.action = "python-control"
    config.python_control_loop_type = "python-control-loop"
    config.extract_features = False

    # Allophone state transducer
    config["*"].transducer_builder_filter_out_invalid_allophones = True  # type: ignore
    config["*"].fix_allophone_context_at_word_boundaries = True  # type: ignore

    # Automaton manipulation
    config.alignment_fsa_exporter.add_blank_transition = add_blank_transition  # type: ignore
    config.alignment_fsa_exporter.blank_label_index = blank_label_index  # type: ignore
    config.alignment_fsa_exporter.allow_label_loop = allow_label_loop  # type: ignore

    # maybe not needed
    config["*"].allow_for_silence_repetitions = False  # type: ignore

    config._update(extra_config)
    post_config._update(extra_post_config)

    return config, post_config


def make_ctc_rasr_loss_config_v2(
    loss_corpus_path: str,
    loss_lexicon_path: str,
    am_args: Dict,
    allow_label_loop: bool = True,
    min_duration: int = 1,
    extra_config: Optional[rasr.RasrConfig] = None,
    extra_post_config: Optional[rasr.RasrConfig] = None,
):

    # Make crp and set loss_corpus and lexicon
    loss_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(loss_crp)

    loss_crp.corpus_config = rasr.RasrConfig()  # type: ignore
    loss_crp.corpus_config.file = loss_corpus_path  # type: ignore
    loss_crp.corpus_config.remove_corpus_name_prefix = "loss-corpus/"  # type: ignore

    loss_crp.lexicon_config = rasr.RasrConfig()  # type: ignore
    loss_crp.lexicon_config.file = loss_lexicon_path  # type: ignore

    loss_crp.acoustic_model_config = acoustic_model_config(**am_args)  # type: ignore

    # Make config from crp
    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }
    config, post_config = rasr.build_config_from_mapping(
        loss_crp,
        mapping,
        parallelize=False,
    )
    config.action = "python-control"
    config.python_control_loop_type = "python-control-loop"
    config.extract_features = False

    # Allophone state transducer
    config["*"].transducer_builder_filter_out_invalid_allophones = True  # type: ignore
    config["*"].fix_allophone_context_at_word_boundaries = True  # type: ignore

    # Automaton manipulation
    if allow_label_loop:
        topology = "ctc"
    else:
        topology = "rna"
    config["*"].allophone_state_graph_builder.topology = topology  # type: ignore

    if min_duration > 1:
        config["*"].allophone_state_graph_builder.label_min_duration = min_duration  # type: ignore

    # maybe not needed
    config["*"].allow_for_silence_repetitions = False  # type: ignore

    config._update(extra_config)
    post_config._update(extra_post_config)

    return config, post_config


def make_rasr_ctc_loss_opts(
    rasr_binary_path: str, rasr_arch: str = "linux-x86_64-standard", v2: bool = True, num_instances: int = 2, **kwargs
):
    trainer_exe = Path(rasr_binary_path) / f"nn-trainer.{rasr_arch}"

    if v2:
        config, post_config = make_ctc_rasr_loss_config_v2(**kwargs)
    else:
        config, post_config = make_ctc_rasr_loss_config_v1(**kwargs)

    loss_opts = {
        "sprint_opts": {
            "sprintExecPath": trainer_exe.as_posix(),
            "sprintConfigStr": f"{config} {post_config} --*.LOGFILE=nn-trainer.loss.log --*.TASK=1",
            "minPythonControlVersion": 4,
            "numInstances": num_instances,
            "usePythonSegmentOrder": False,
        },
        "tdp_scale": 0.0,
    }
    return loss_opts


def add_rasr_fastbw_output_layer(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    l2: Optional[float] = None,
    **kwargs,
):
    network[name] = {
        "class": "softmax",
        "from": from_list,
        "loss": "fast_bw",
        "loss_opts": {
            "sprint_opts": make_rasr_ctc_loss_opts(**kwargs),
            "tdp_scale": 0.0,
        },
        "target": None,
        "n_out": num_outputs,
    }
    if l2:
        network[name]["L2"] = l2

    return name


def make_blstm_fullsum_ctc_model(
    num_outputs: int,
    gt_args: Dict = {},
    blstm_args: Dict = {},
    mlp_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, Union[str, List[str]]]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list, python_code = add_gt_feature_extraction(network, from_list=from_list, name="gt", **gt_args)
    from_list, _ = add_blstm_stack(network, from_list, **blstm_args)
    network["encoder"] = {"class": "copy", "from": from_list}
    from_list = add_feed_forward_stack(network, "encoder", **mlp_args)
    add_rasr_fastbw_output_layer(network, from_list=from_list, num_outputs=num_outputs, **output_args)

    return network, python_code


def make_blstm_ctc_recog_model(
    num_outputs: int,
    gt_args: Dict = {},
    blstm_args: Dict = {},
    mlp_args: Dict = {},
) -> Tuple[Dict, Union[str, List[str]]]:
    network = {}

    from_list = ["data"]

    gt_args_mod = copy.deepcopy(gt_args)
    gt_args_mod.setdefault("specaug_before_dct", False)
    gt_args_mod.setdefault("specaug_after_dct", False)

    from_list, python_code = add_gt_feature_extraction(network, from_list=from_list, name="gt", **gt_args_mod)
    from_list, _ = add_blstm_stack(network, from_list, **blstm_args)
    network["encoder"] = {"class": "copy", "from": from_list}
    from_list = add_feed_forward_stack(network, "encoder", **mlp_args)

    network["output"] = {
        "class": "linear",
        "from": from_list,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, python_code
