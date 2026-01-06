from typing import Optional, Tuple
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob, build_config_from_mapping
from i6_core.rasr.crp import CommonRasrParameters

from i6_core.am.config import TdpValues, acoustic_model_config

from sisyphus import *


from .default_tools import rasr_binary_path


def get_no_op_label_scorer_config() -> RasrConfig:
    config = RasrConfig()
    config.type = "no-op"

    return config

def get_tree_timesync_recog_config(
    lexicon_file: tk.Path,
    collapse_repeated_labels: bool,
    label_scorer_config: Optional[RasrConfig] = None,
    am_config: Optional[RasrConfig] = None,
    lm_config: Optional[RasrConfig] = None,
    blank_index: Optional[int] = None,
    max_beam_size: int = 1024,
    max_word_end_beam_size: Optional[int] = None,
    intermediate_max_beam_size: Optional[int] = 1024,
    score_threshold: Optional[float] = 18.0,
    word_end_score_threshold: Optional[float] = None,
    intermediate_score_threshold: Optional[float] = 18.0,
    sentence_end_fallback: bool = True,
    maximum_stable_delay: Optional[int] = None,
    log_stepwise_statistics: bool = True,
    logfile_suffix: str = "recog",
) -> Tuple[RasrConfig, RasrConfig]:
    crp = CommonRasrParameters()

    # LibRASR does not have a channel manager so the settings from `crp_add_default_output` don't work
    logfile_name = f"rasr.{logfile_suffix}.log"

    log_config = RasrConfig()
    log_config["*.log.channel"] = logfile_name
    log_config["*.warning.channel"] = logfile_name
    log_config["*.error.channel"] = logfile_name
    log_config["*.statistics.channel"] = logfile_name
    log_config["*.unbuffered"] = False

    log_post_config = RasrConfig()
    log_post_config["*.encoding"] = "UTF-8"
    crp.log_config = log_config  # type: ignore
    crp.log_post_config = log_post_config  # type: ignore
    crp.default_log_channel = logfile_name

    crp.set_executables(rasr_binary_path=rasr_binary_path)

    rasr_config, rasr_post_config = build_config_from_mapping(crp=crp, mapping={}, include_log_config=True)

    rasr_config.lib_rasr = RasrConfig()

    rasr_config.lib_rasr.lexicon = RasrConfig()

    rasr_config.lib_rasr.search_algorithm = RasrConfig()
    rasr_config.lib_rasr.search_algorithm.type = "tree-timesync-beam-search"

    rasr_config.lib_rasr.lexicon.file = lexicon_file

    if lm_config is not None:
        rasr_config.lib_rasr.lm = lm_config
    else:
        rasr_config.lib_rasr.lm = RasrConfig()
        rasr_config.lib_rasr.lm.scale = 0.0

    if am_config is None:
        am_config = acoustic_model_config(
            states_per_phone=1,
            tdp_transition=TdpValues(loop=0.0, forward=0.0, skip="infinity", exit=0.0),
            tdp_silence=TdpValues(loop=0.0, forward=0.0, skip="infinity", exit=0.0),
            phon_history_length=0,
            phon_future_length=0,
        )
    rasr_config.lib_rasr.acoustic_model = am_config

    if collapse_repeated_labels:
        rasr_config.lib_rasr.search_algorithm.tree_builder_type = "ctc"
    else:
        rasr_config.lib_rasr.search_algorithm.tree_builder_type = "rna"

    rasr_config.lib_rasr.search_algorithm.max_beam_size = max_beam_size
    if max_word_end_beam_size is not None:
        rasr_config.lib_rasr.search_algorithm.max_word_end_beam_size = max_word_end_beam_size
    if intermediate_max_beam_size is not None:
        rasr_config.lib_rasr.search_algorithm.intermediate_max_beam_size = intermediate_max_beam_size
    if score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.score_threshold = score_threshold
    if word_end_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.word_end_score_threshold = word_end_score_threshold
    if intermediate_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.intermediate_score_threshold = intermediate_score_threshold
    if blank_index is not None:
        rasr_config.lib_rasr.search_algorithm.blank_label_index = blank_index
    rasr_config.lib_rasr.search_algorithm.collapse_repeated_labels = collapse_repeated_labels
    rasr_config.lib_rasr.search_algorithm.force_blank_between_repeated_labels = collapse_repeated_labels
    rasr_config.lib_rasr.search_algorithm.sentence_end_fall_back = sentence_end_fallback
    rasr_config.lib_rasr.search_algorithm.log_stepwise_statistics = log_stepwise_statistics
    if maximum_stable_delay is not None:
        rasr_config.lib_rasr.search_algorithm.maximum_stable_delay = maximum_stable_delay

    if label_scorer_config is not None:
        rasr_config.lib_rasr.label_scorer = label_scorer_config
    else:
        rasr_config.lib_rasr.label_scorer = get_no_op_label_scorer_config()

    return rasr_config, rasr_post_config