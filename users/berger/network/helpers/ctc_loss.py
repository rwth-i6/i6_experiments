import copy
from enum import Enum, auto
from typing import Dict, List, Union, Optional

from sisyphus import gs, tk

from i6_core.lib.corpus import Corpus
from i6_core.corpus import MergeStrategy, MergeCorporaJob, SegmentCorpusJob
from i6_core.rasr import (
    RasrCommand,
    CommonRasrParameters,
    RasrConfig,
    build_config_from_mapping,
    crp_add_default_output,
)


class CtcLossType(Enum):
    RasrFastBW = (auto(),)
    ReturnnFastBW = auto()


def add_ctc_output_layer(type: CtcLossType = CtcLossType.RasrFastBW, **kwargs):
    if type == CtcLossType.RasrFastBW:
        return add_rasr_fastbw_output_layer(**kwargs)
    else:
        return add_returnn_fastbw_output_layer(**kwargs)


def make_rasr_fullsum_loss_opts(rasr_exe=None):
    trainer_exe = RasrCommand.select_exe(rasr_exe, "nn-trainer")
    loss_opts = {
        "sprintExecPath": trainer_exe,
        "sprintConfigStr": "--config=rasr.loss.config --*.LOGFILE=nn-trainer.loss.log --*.TASK=1",
        "minPythonControlVersion": 4,
        "numInstances": 2,
        "usePythonSegmentOrder": False,
    }
    return loss_opts


def add_rasr_fastbw_output_layer(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    l2: Optional[float] = None,
    rasr_exe=None,
):
    network[name] = {
        "class": "softmax",
        "from": from_list,
        "loss": "fast_bw",
        "loss_opts": {
            "sprint_opts": make_rasr_fullsum_loss_opts(rasr_exe),
            "tdp_scale": 0.0,
        },
        "target": None,
        "n_out": num_outputs,
    }
    if l2:
        network[name]["L2"] = l2

    return name


def make_ctc_rasr_loss_config(
    train_corpus_path: str,
    dev_corpus_path: str,
    base_crp: Optional[CommonRasrParameters] = None,
    add_blank_transition: bool = True,
    allow_label_loop: bool = True,
    blank_label_index: int = -1,  # -1 => replace silence
    extra_config: Optional[RasrConfig] = None,
    extra_post_config: Optional[RasrConfig] = None,
):
    # Check if train and dev corpus names are equal
    # train_corpus = Corpus()
    # train_corpus.load(train_corpus_path)
    # dev_corpus = Corpus()
    # dev_corpus.load(dev_corpus_path)
    # assert train_corpus.name == dev_corpus.name

    # Create loss corpus by merging train and dev corpus
    loss_corpus = MergeCorporaJob(
        [train_corpus_path, dev_corpus_path],
        name="loss-corpus",
        merge_strategy=MergeStrategy.SUBCORPORA,
    ).out_merged_corpus

    # Make crp from base_crp and set loss_corpus and segments
    if base_crp:
        loss_crp = copy.deepcopy(base_crp)
    else:
        loss_crp = CommonRasrParameters()
    crp_add_default_output(loss_crp, unbuffered=True)

    # loss_crp.python_exe = gs.RASR_PYTHON_EXE
    loss_crp.corpus_config.file = loss_corpus
    loss_crp.corpus_config.remove_corpus_name_prefix = "loss-corpus/"
    loss_crp.segment_path = SegmentCorpusJob(
        loss_corpus, 1, remove_prefix="loss-corpus/"
    ).out_segment_path

    # Make config from crp
    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }
    config, post_config = build_config_from_mapping(
        loss_crp, mapping, parallelize=(base_crp.concurrent == 1)
    )
    config.neural_network_trainer.action = "python-control"
    config.neural_network_trainer.python_control_loop_type = "python-control-loop"
    config.neural_network_trainer.extract_features = False

    # Allophone state transducer
    config["*"].transducer_builder_filter_out_invalid_allophones = True
    config["*"].fix_allophone_context_at_word_boundaries = True

    # Automaton manipulation
    config.neural_network_trainer.alignment_fsa_exporter.add_blank_transition = (
        add_blank_transition
    )
    config.neural_network_trainer.alignment_fsa_exporter.blank_label_index = (
        blank_label_index
    )
    config.neural_network_trainer.alignment_fsa_exporter.allow_label_loop = (
        allow_label_loop
    )

    # maybe not needed
    config["*"].allow_for_silence_repetitions = False

    config._update(extra_config)
    post_config._update(extra_post_config)

    return config, post_config


def make_ctc_rasr_loss_config_v2(
    train_corpus_path: tk.Path,
    dev_corpus_path: tk.Path,
    base_crp: Optional[CommonRasrParameters] = None,
    allow_label_loop: bool = True,
    min_duration: int = 1,
    extra_config: Optional[RasrConfig] = None,
    extra_post_config: Optional[RasrConfig] = None,
):
    # Create loss corpus by merging train and dev corpus
    loss_corpus = MergeCorporaJob(
        [train_corpus_path, dev_corpus_path],
        name="loss-corpus",
        merge_strategy=MergeStrategy.SUBCORPORA,
    ).out_merged_corpus

    # Make crp from base_crp and set loss_corpus and segments
    if base_crp:
        loss_crp = copy.deepcopy(base_crp)
    else:
        loss_crp = CommonRasrParameters()
    crp_add_default_output(loss_crp, unbuffered=True)
    loss_crp.set_executables(
        tk.Path("/u/berger/rasr_github/arch/linux-x86_64-standard")
    )

    loss_crp.corpus_config.file = loss_corpus
    loss_crp.corpus_config.remove_corpus_name_prefix = "loss-corpus/"

    # Make config from crp
    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }
    config, post_config = build_config_from_mapping(
        loss_crp, mapping, parallelize=(base_crp.concurrent == 1)
    )
    config.neural_network_trainer.action = "python-control"
    config.neural_network_trainer.python_control_loop_type = "python-control-loop"
    config.neural_network_trainer.extract_features = False

    # Allophone state transducer
    config["*"].transducer_builder_filter_out_invalid_allophones = True
    config["*"].fix_allophone_context_at_word_boundaries = True

    # Automaton manipulation
    if allow_label_loop:
        topology = "ctc"
    else:
        topology = "rna"
    config["*"].allophone_state_graph_builder.topology = topology

    if min_duration > 1:
        config["*"].allophone_state_graph_builder.label_min_duration = min_duration

    # maybe not needed
    config["*"].allow_for_silence_repetitions = False

    config._update(extra_config)
    post_config._update(extra_post_config)

    return config, post_config


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
        blank_index = num_outputs - 1
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
        "from": "output_0",
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
