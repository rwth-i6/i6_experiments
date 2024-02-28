from enum import Enum, auto
from typing import Dict, List, Union, Optional

from sisyphus import tk


from i6_core.am.config import acoustic_model_config
import i6_core.rasr as rasr
from sisyphus import tk
from sisyphus.delayed_ops import DelayedFunction
from typing import Dict, List, Optional, Union


class CtcLossType(Enum):
    RasrFastBW = auto()
    ReturnnFastBW = auto()
    ReturnnTF = auto()


def add_ctc_output_layer(type: CtcLossType = CtcLossType.RasrFastBW, **kwargs):
    if type == CtcLossType.RasrFastBW:
        return add_rasr_fastbw_output_layer(**kwargs)
    elif type == CtcLossType.ReturnnFastBW:
        return add_returnn_fastbw_output_layer(**kwargs)
    elif type == CtcLossType.ReturnnTF:
        return add_returnn_tf_ctc_output_layer(**kwargs)
    raise NotImplementedError


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
    loss_crp.acoustic_model_config.allophones.add_all = True  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_from_lexicon = False  # type: ignore

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
    loss_corpus_path: tk.Path,
    loss_lexicon_path: tk.Path,
    am_args: Dict,
    allow_label_loop: bool = True,
    min_duration: int = 1,
    extra_config: Optional[rasr.RasrConfig] = None,
    extra_post_config: Optional[rasr.RasrConfig] = None,
    remove_prefix: str = "loss-corpus/",
):
    # Make crp and set loss_corpus and lexicon
    loss_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(loss_crp)

    loss_crp.corpus_config = rasr.RasrConfig()  # type: ignore
    loss_crp.corpus_config.file = loss_corpus_path  # type: ignore
    loss_crp.corpus_config.remove_corpus_name_prefix = remove_prefix  # type: ignore

    loss_crp.lexicon_config = rasr.RasrConfig()  # type: ignore
    loss_crp.lexicon_config.file = loss_lexicon_path  # type: ignore

    loss_crp.acoustic_model_config = acoustic_model_config(**am_args)  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_all = True  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_from_lexicon = False  # type: ignore

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


def format_func(s, *args, **kwargs):
    return s % args


def make_rasr_ctc_loss_opts(
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    v2: bool = True,
    num_instances: int = 2,
    **kwargs,
):
    trainer_exe = rasr_binary_path.join_right(f"nn-trainer.{rasr_arch}")

    if v2:
        config, post_config = make_ctc_rasr_loss_config_v2(**kwargs)
    else:
        config, post_config = make_ctc_rasr_loss_config_v1(**kwargs)

    loss_opts = {
        "sprint_opts": {
            "sprintExecPath": trainer_exe,
            "sprintConfigStr": DelayedFunction(
                "%s %s --*.LOGFILE=nn-trainer.loss.log --*.TASK=1", format_func, config, post_config
            ),
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
        "loss_opts": make_rasr_ctc_loss_opts(**kwargs),
        "target": None,
        "n_out": num_outputs,
    }
    if l2:
        network[name]["L2"] = l2

    return name


def add_returnn_fastbw_output_layer(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    reuse_from_name: Optional[str] = None,
    target_key: str = "classes",
    blank_index: Optional[int] = None,
    l2: Optional[float] = None,
):
    if blank_index is None:
        blank_index = 0
    else:
        assert 0 <= blank_index < num_outputs

    network[name] = {
        "class": "softmax",
        "from": from_list,
        "n_out": num_outputs,
    }
    if l2:
        network[name]["L2"] = l2
    if reuse_from_name is not None:
        network[name]["reuse_params"] = reuse_from_name

    network[f"{name}_ctc_loss"] = {
        "class": "fast_bw",
        "from": name,
        "align_target_key": target_key,
        "align_target": "ctc",
        "input_type": "prob",
        "tdp_scale": 0.0,
        "ctc_opts": {"blank_idx": blank_index},
    }
    network[f"{name}_apply_loss"] = {
        "class": "copy",
        "from": name,
        "loss": "via_layer",
        "loss_opts": {
            "loss_wrt_to_act_in": "softmax",
            "align_layer": f"{name}_ctc_loss",
        },
    }


def add_returnn_tf_ctc_output_layer(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    reuse_from_name: Optional[str] = None,
    target_key: str = "classes",
    blank_index: int = 0,
    l2: Optional[float] = None,
):
    assert 0 <= blank_index < num_outputs

    network[name] = {
        "class": "softmax",
        "from": from_list,
        "n_out": num_outputs,
        "loss": "ctc",
        "target": target_key,
        "loss_opts": {
            "use_native": True,
            "beam_width": 1,
            "ctc_opts": {
                "blank_index": blank_index,
            },
        },
    }
    if l2:
        network[name]["L2"] = l2
    if reuse_from_name is not None:
        network[name]["reuse_params"] = reuse_from_name
