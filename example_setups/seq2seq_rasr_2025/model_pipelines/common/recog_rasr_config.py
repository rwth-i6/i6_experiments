__all__ = [
    "LexiconfreeTimesyncRecogParams",
    "get_lexiconfree_timesync_recog_config",
    "LexiconfreeLabelsyncRecogParams",
    "get_lexiconfree_labelsync_recog_config",
    "TreeTimesyncRecogParams",
    "get_tree_timesync_recog_config",
]

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from i6_core.am.config import TdpValues, acoustic_model_config
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob, build_config_from_mapping
from i6_core.rasr.crp import CommonRasrParameters
from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from ...tools import rasr_binary_path


def _add_label_scorers_to_rasr_config(label_scorer_configs: List[RasrConfig], rasr_config: RasrConfig) -> None:
    if len(label_scorer_configs) == 1:
        rasr_config.label_scorer = label_scorer_configs[0]
    else:
        rasr_config.num_label_scorers = len(label_scorer_configs)
        for i, scorer_config in enumerate(label_scorer_configs, start=1):
            rasr_config[f"label-scorer-{i}"] = scorer_config


@dataclass
class LexiconfreeTimesyncRecogParams:
    max_beam_sizes: List[int]
    collapse_repeated_labels: bool
    score_thresholds: Optional[List[float]] = None
    allow_blank_after_sentence_end: bool = False
    log_stepwise_statistics: bool = True
    maximum_stable_delay: Optional[int] = None
    maximum_stable_delay_pruning_interval: Optional[int] = None


def get_lexiconfree_timesync_recog_config(
    vocab_file: tk.Path,
    label_scorer_configs: List[RasrConfig],
    params: LexiconfreeTimesyncRecogParams,
    blank_index: Optional[int] = None,
    sentence_end_index: Optional[int] = None,
    logfile_suffix: str = "recog",
) -> tk.Path:
    crp = CommonRasrParameters()

    # LibRASR does not have a channel manager so the settings from `crp_add_default_output` don't work
    log_config = RasrConfig()
    logfile_name = f"rasr.{logfile_suffix}.log"
    log_config["*.log.channel"] = logfile_name
    log_config["*.log.channel"] = logfile_name
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
    rasr_config.lib_rasr.search_algorithm.type = "lexiconfree-timesync-beam-search"

    rasr_config.lib_rasr.lexicon.file = DelayedFormat("vocab-text:{}", vocab_file)

    rasr_config.lib_rasr.search_algorithm.max_beam_size = params.max_beam_sizes
    if params.score_thresholds is not None:
        rasr_config.lib_rasr.search_algorithm.score_threshold = params.score_thresholds
    if blank_index is not None:
        rasr_config.lib_rasr.search_algorithm.blank_label_index = blank_index
    if sentence_end_index is not None:
        rasr_config.lib_rasr.search_algorithm.sentence_end_label_index = sentence_end_index
    if params.maximum_stable_delay is not None:
        rasr_config.lib_rasr.search_algorithm.maximum_stable_delay = params.maximum_stable_delay
        if params.maximum_stable_delay_pruning_interval is not None:
            rasr_config.lib_rasr.search_algorithm.maximum_stable_delay_pruning_interval = (
                params.maximum_stable_delay_pruning_interval
            )
    rasr_config.lib_rasr.search_algorithm.collapse_repeated_labels = params.collapse_repeated_labels
    rasr_config.lib_rasr.search_algorithm.allow_blank_after_sentence_end = params.allow_blank_after_sentence_end
    rasr_config.lib_rasr.search_algorithm.log_stepwise_statistics = params.log_stepwise_statistics

    _add_label_scorers_to_rasr_config(label_scorer_configs, rasr_config.lib_rasr)

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config

    return recog_rasr_config_path


@dataclass
class LexiconfreeLabelsyncRecogParams:
    max_beam_sizes: List[int]
    score_thresholds: Optional[List[float]] = None
    max_labels_per_time_step: int = 1
    length_norm_scale: Optional[float] = None
    log_stepwise_statistics: bool = True
    maximum_stable_delay: Optional[int] = None
    maximum_stable_delay_pruning_interval: Optional[int] = None


def get_lexiconfree_labelsync_recog_config(
    vocab_file: tk.Path,
    label_scorer_configs: List[RasrConfig],
    params: LexiconfreeLabelsyncRecogParams,
    sentence_end_index: Optional[int] = None,
    logfile_suffix: str = "recog",
) -> tk.Path:
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
    rasr_config.lib_rasr.search_algorithm.type = "lexiconfree-labelsync-beam-search"

    rasr_config.lib_rasr.lexicon.file = DelayedFormat("vocab-text:{}", vocab_file)

    rasr_config.lib_rasr.search_algorithm.max_beam_size = params.max_beam_sizes
    if params.score_thresholds is not None:
        rasr_config.lib_rasr.search_algorithm.score_threshold = params.score_thresholds
    if sentence_end_index is not None:
        rasr_config.lib_rasr.search_algorithm.sentence_end_label_index = sentence_end_index
    if params.length_norm_scale is not None:
        rasr_config.lib_rasr.search_algorithm.length_norm_scale = params.length_norm_scale
    rasr_config.lib_rasr.search_algorithm.max_labels_per_time_step = params.max_labels_per_time_step
    rasr_config.lib_rasr.search_algorithm.log_stepwise_statistics = params.log_stepwise_statistics
    if params.maximum_stable_delay is not None:
        rasr_config.lib_rasr.search_algorithm.maximum_stable_delay = params.maximum_stable_delay
        if params.maximum_stable_delay_pruning_interval is not None:
            rasr_config.lib_rasr.search_algorithm.maximum_stable_delay_pruning_interval = (
                params.maximum_stable_delay_pruning_interval
            )

    _add_label_scorers_to_rasr_config(label_scorer_configs, rasr_config.lib_rasr)

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config

    return recog_rasr_config_path


@dataclass
class TreeTimesyncRecogParams:
    collapse_repeated_labels: bool
    max_beam_sizes: List[int]
    max_word_end_beam_size: Optional[int] = None
    score_thresholds: Optional[List[float]] = None
    word_end_score_threshold: Optional[float] = None
    sentence_end_fallback: bool = True
    allow_blank_after_sentence_end: bool = False
    maximum_stable_delay: Optional[int] = None
    maximum_stable_delay_pruning_interval: Optional[int] = None
    log_stepwise_statistics: bool = True


def get_tree_timesync_recog_config(
    lexicon_file: tk.Path,
    label_scorer_configs: List[RasrConfig],
    params: TreeTimesyncRecogParams,
    am_config: Optional[RasrConfig] = None,
    lm_config: Optional[RasrConfig] = None,
    logfile_suffix: str = "recog",
) -> tk.Path:
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

    if params.collapse_repeated_labels:
        rasr_config.lib_rasr.search_algorithm.tree_builder_type = "ctc"
    else:
        rasr_config.lib_rasr.search_algorithm.tree_builder_type = "rna"

    rasr_config.lib_rasr.search_algorithm.max_beam_size = params.max_beam_sizes
    if params.max_word_end_beam_size is not None:
        rasr_config.lib_rasr.search_algorithm.max_word_end_beam_size = params.max_word_end_beam_size
    if params.score_thresholds is not None:
        rasr_config.lib_rasr.search_algorithm.score_threshold = params.score_thresholds
    if params.score_thresholds is not None and params.word_end_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.word_end_score_threshold = params.word_end_score_threshold
    rasr_config.lib_rasr.search_algorithm.collapse_repeated_labels = params.collapse_repeated_labels
    rasr_config.lib_rasr.search_algorithm.force_blank_between_repeated_labels = params.collapse_repeated_labels
    rasr_config.lib_rasr.search_algorithm.sentence_end_fall_back = params.sentence_end_fallback
    rasr_config.lib_rasr.search_algorithm.log_stepwise_statistics = params.log_stepwise_statistics
    rasr_config.lib_rasr.search_algorithm.allow_blank_after_sentence_end = params.allow_blank_after_sentence_end
    if params.maximum_stable_delay is not None:
        rasr_config.lib_rasr.search_algorithm.maximum_stable_delay = params.maximum_stable_delay
        if params.maximum_stable_delay_pruning_interval is not None:
            rasr_config.lib_rasr.search_algorithm.maximum_stable_delay_pruning_interval = (
                params.maximum_stable_delay_pruning_interval
            )

    _add_label_scorers_to_rasr_config(label_scorer_configs, rasr_config.lib_rasr)

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config
    return recog_rasr_config_path


class LabelsyncGlobalPruningStrategy(Enum):
    NONE = "none"
    ACTIVE_AGAINST_TERMINATED = "active-against-terminated"
    ALL = "all"


def get_tree_labelsync_recog_config(
    lexicon_file: tk.Path,
    label_scorer_configs: List[RasrConfig],
    am_config: Optional[RasrConfig] = None,
    lm_config: Optional[RasrConfig] = None,
    max_beam_size: int = 1024,
    max_word_end_beam_size: Optional[int] = None,
    global_max_beam_size: Optional[int] = None,
    score_threshold: Optional[float] = 18.0,
    word_end_score_threshold: Optional[float] = None,
    global_score_threshold: Optional[float] = None,
    global_pruning_strategy: LabelsyncGlobalPruningStrategy = LabelsyncGlobalPruningStrategy.ACTIVE_AGAINST_TERMINATED,
    domination_score_threshold: Optional[float] = None,
    length_norm_scale: Optional[float] = None,
    max_labels_per_time_step: int = 1,
    sentence_end_fallback: bool = True,
    log_stepwise_statistics: bool = True,
    logfile_suffix: str = "recog",
) -> tk.Path:
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
    rasr_config.lib_rasr.search_algorithm.type = "tree-labelsync-beam-search"

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

    rasr_config.lib_rasr.search_algorithm.tree_builder_type = "aed"

    rasr_config.lib_rasr.search_algorithm.max_beam_size = max_beam_size
    if max_word_end_beam_size is not None:
        rasr_config.lib_rasr.search_algorithm.max_word_end_beam_size = max_word_end_beam_size
    if global_max_beam_size is not None:
        rasr_config.lib_rasr.search_algorithm.global_max_beam_size = global_max_beam_size
    if score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.score_threshold = score_threshold
    if word_end_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.word_end_score_threshold = word_end_score_threshold
    if global_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.global_score_threshold = global_score_threshold
    if domination_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.domination_score_threshold = domination_score_threshold
    rasr_config.lib_rasr.search_algorithm.global_pruning_strategy = global_pruning_strategy.value
    if length_norm_scale is not None:
        rasr_config.lib_rasr.search_algorithm.length_norm_scale = length_norm_scale
    if max_labels_per_time_step is not None:
        rasr_config.lib_rasr.search_algorithm.max_labels_per_time_step = max_labels_per_time_step
    rasr_config.lib_rasr.search_algorithm.sentence_end_fall_back = sentence_end_fallback
    rasr_config.lib_rasr.search_algorithm.log_stepwise_statistics = log_stepwise_statistics

    _add_label_scorers_to_rasr_config(label_scorer_configs, rasr_config.lib_rasr)

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config
    return recog_rasr_config_path
