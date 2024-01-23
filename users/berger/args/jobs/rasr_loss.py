from sisyphus import Path
from typing import Optional, List
from i6_core import rasr
from i6_core.corpus import SegmentCorpusJob


def create_full_sum_loss_config(
    base_crp,
    num_classes,
    corpus_file: Optional[Path] = None,
    loss_am_config: Optional[rasr.RasrConfig] = None,
    loss_lexicon: Optional[Path] = None,
    add_blank_transition: bool = True,
    allow_label_loop: bool = True,
    blank_index: Optional[int] = None,
    skip_segments: Optional[List[str]] = None,
    extra_config: Optional[rasr.RasrConfig] = None,
    post_config: Optional[rasr.RasrConfig] = None,
):

    crp = rasr.CommonRasrParameters(base=base_crp)
    rasr.crp_add_default_output(crp, unbuffered=True)
    if corpus_file:
        crp.corpus_config.file = corpus_file
        all_segments = SegmentCorpusJob(corpus_file, 1)
        crp.segment_path = all_segments.out_segment_path
    if loss_am_config:
        crp.acoustic_model_config = loss_am_config
    if loss_lexicon:
        loss_crp.lexicon_config.file = loss_lexicon

    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }

    config, post_config = rasr.build_config_from_mapping(crp, mapping, parallelize=(crp.concurrent == 1))
    # concrete action in PythonControl called from RETURNN SprintErrorSignals.py derived from Loss/Layers
    config.neural_network_trainer.action = "python-control"
    config.neural_network_trainer.python_control_loop_type = "python-control-loop"
    config.neural_network_trainer.extract_features = False
    # allophone-state transducer
    config["*"].transducer_builder_filter_out_invalid_allophones = True
    config["*"].fix_allophone_context_at_word_boundaries = True
    # Automaton manipulation (RASR): default CTC topology
    config.neural_network_trainer.alignment_fsa_exporter.add_blank_transition = add_blank_transition
    config.neural_network_trainer.alignment_fsa_exporter.allow_label_loop = allow_label_loop
    # default blank replace silence
    if blank_index:
        config.neural_network_trainer.alignment_fsa_exporter.blank_label_index = blank_index
    config["*"].allow_for_silence_repetitions = False
    config["*"].number_of_classes = num_classes
    if skip_segments:
        config.neural_network_trainer["*"].segments_to_skip = skip_segments

    config._update(extra_config)
    post_config._update(post_config)
    return config, post_config


def create_rasr_loss_opts(cls, sprint_exe=None, custom_config=None, **kwargs):
    """

    :param sprint_exe:
    :param custom_config:
    :param kwargs:
    :return:
    """
    trainer_exe = rasr.RasrCommand.select_exe(sprint_exe, "nn-trainer")
    python_seg_order = False  # get automaton by segment name
    sprint_opts = {
        "sprintExecPath": trainer_exe,
        "minPythonControlVersion": 4,
        "numInstances": kwargs.get("num_sprint_instance", 2),
        "usePythonSegmentOrder": python_seg_order,
    }
    if custom_config:
        sprint_opts["sprintConfigStr"] = DelayedFormat(
            '"--config={} --*.LOGFILE=nn-trainer.loss.log --*.TASK=1"', custom_config
        )
    else:
        sprint_opts["sprintConfigStr"] = "--config=rasr.loss.config --*.LOGFILE=nn-trainer.loss.log --*.TASK=1"
    return sprint_opts
