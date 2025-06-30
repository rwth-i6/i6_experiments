__all__ = [
    "get_no_op_label_scorer_config",
    "get_combine_label_scorer_config",
    "get_lexiconfree_timesync_recog_config",
    "get_lexiconfree_labelsync_recog_config",
    "get_tree_timesync_recog_config",
]

from typing import List, Optional, Tuple

from i6_core.am.config import TdpValues, acoustic_model_config
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob, build_config_from_mapping
from i6_core.rasr.crp import CommonRasrParameters
from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from ...tools import rasr_binary_path


def get_no_op_label_scorer_config() -> RasrConfig:
    config = RasrConfig()
    config.type = "no-op"

    return config


def get_combine_label_scorer_config(sub_scorers: List[Tuple[RasrConfig, float]]) -> RasrConfig:
    rasr_config = RasrConfig()
    rasr_config.type = "combine"
    rasr_config.num_scorers = len(sub_scorers)

    for i, (sub_scorer, scale) in enumerate(sub_scorers, start=1):
        rasr_config[f"scorer-{i}"] = sub_scorer
        rasr_config[f"scorer-{i}.scale"] = scale

    return rasr_config


def get_lexiconfree_timesync_recog_config(
    vocab_file: tk.Path,
    collapse_repeated_labels: bool,
    label_scorer_config: Optional[RasrConfig] = None,
    blank_index: Optional[int] = None,
    max_beam_size: int = 1024,
    intermediate_max_beam_size: Optional[int] = 1024,
    score_threshold: Optional[float] = 18.0,
    intermediate_score_threshold: Optional[float] = 18.0,
    log_stepwise_statistics: bool = False,
) -> tk.Path:
    crp = CommonRasrParameters()

    # LibRASR does not have a channel manager so the settings from `crp_add_default_output` don't work
    log_config = RasrConfig()
    log_config["*.log.channel"] = "rasr.log"
    log_config["*.warning.channel"] = "rasr.log"
    log_config["*.error.channel"] = "rasr.log"
    log_config["*.statistics.channel"] = "rasr.log"
    log_config["*.unbuffered"] = True

    log_post_config = RasrConfig()
    log_post_config["*.encoding"] = "UTF-8"
    crp.log_config = log_config  # type: ignore
    crp.log_post_config = log_post_config  # type: ignore
    crp.default_log_channel = "rasr.log"

    crp.set_executables(rasr_binary_path=rasr_binary_path)

    rasr_config, rasr_post_config = build_config_from_mapping(crp=crp, mapping={}, include_log_config=True)

    rasr_config.lib_rasr = RasrConfig()

    rasr_config.lib_rasr.lexicon = RasrConfig()

    rasr_config.lib_rasr.search_algorithm = RasrConfig()
    rasr_config.lib_rasr.search_algorithm.type = "lexiconfree-timesync-beam-search"

    rasr_config.lib_rasr.lexicon.file = DelayedFormat("vocab-text:{}", vocab_file)

    rasr_config.lib_rasr.search_algorithm.max_beam_size = max_beam_size
    if intermediate_max_beam_size is not None:
        rasr_config.lib_rasr.search_algorithm.intermediate_max_beam_size = intermediate_max_beam_size
    if score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.score_threshold = score_threshold
    if intermediate_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.intermediate_score_threshold = intermediate_score_threshold
    if blank_index is not None:
        rasr_config.lib_rasr.search_algorithm.blank_label_index = blank_index
    rasr_config.lib_rasr.search_algorithm.collapse_repeated_labels = collapse_repeated_labels
    rasr_config.lib_rasr.search_algorithm.log_stepwise_statistics = log_stepwise_statistics

    if label_scorer_config is not None:
        rasr_config.lib_rasr.label_scorer = label_scorer_config
    else:
        rasr_config.lib_rasr.label_scorer = get_no_op_label_scorer_config()

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config

    return recog_rasr_config_path


def get_lexiconfree_labelsync_recog_config(
    vocab_file: tk.Path,
    label_scorer_config: Optional[RasrConfig] = None,
    sentence_end_index: Optional[int] = None,
    max_beam_size: int = 1024,
    intermediate_max_beam_size: Optional[int] = 1024,
    score_threshold: Optional[float] = 18.0,
    intermediate_score_threshold: Optional[float] = 18.0,
    max_labels_per_time_step: int = 1,
    length_norm_scale: Optional[float] = 1.0,
    log_stepwise_statistics: bool = False,
) -> tk.Path:
    crp = CommonRasrParameters()

    # LibRASR does not have a channel manager so the settings from `crp_add_default_output` don't work
    log_config = RasrConfig()
    log_config["*.log.channel"] = "rasr.log"
    log_config["*.warning.channel"] = "rasr.log"
    log_config["*.error.channel"] = "rasr.log"
    log_config["*.statistics.channel"] = "rasr.log"
    log_config["*.unbuffered"] = True

    log_post_config = RasrConfig()
    log_post_config["*.encoding"] = "UTF-8"
    crp.log_config = log_config  # type: ignore
    crp.log_post_config = log_post_config  # type: ignore
    crp.default_log_channel = "rasr.log"

    crp.set_executables(rasr_binary_path=rasr_binary_path)

    rasr_config, rasr_post_config = build_config_from_mapping(crp=crp, mapping={}, include_log_config=True)

    rasr_config.lib_rasr = RasrConfig()

    rasr_config.lib_rasr.lexicon = RasrConfig()

    rasr_config.lib_rasr.search_algorithm = RasrConfig()
    rasr_config.lib_rasr.search_algorithm.type = "lexiconfree-labelsync-beam-search"

    rasr_config.lib_rasr.lexicon.file = DelayedFormat("vocab-text:%s", vocab_file)

    rasr_config.lib_rasr.search_algorithm.max_beam_size = max_beam_size
    if intermediate_max_beam_size is not None:
        rasr_config.lib_rasr.search_algorithm.intermediate_max_beam_size = intermediate_max_beam_size
    if score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.score_threshold = score_threshold
    if intermediate_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.intermediate_score_threshold = intermediate_score_threshold
    if sentence_end_index is not None:
        rasr_config.lib_rasr.search_algorithm.sentence_end_index = sentence_end_index
    if length_norm_scale is not None:
        rasr_config.lib_rasr.search_algorithm.length_norm_scale = length_norm_scale
    rasr_config.lib_rasr.search_algorithm.max_labels_per_time_step = max_labels_per_time_step
    rasr_config.lib_rasr.search_algorithm.log_stepwise_statistics = log_stepwise_statistics

    if label_scorer_config is not None:
        rasr_config.lib_rasr.label_scorer = label_scorer_config
    else:
        rasr_config.lib_rasr.label_scorer = get_no_op_label_scorer_config()

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config

    return recog_rasr_config_path


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
    log_stepwise_statistics: bool = False,
) -> tk.Path:
    crp = CommonRasrParameters()

    # LibRASR does not have a channel manager so the settings from `crp_add_default_output` don't work
    log_config = RasrConfig()
    log_config["*.log.channel"] = "rasr.log"
    log_config["*.warning.channel"] = "rasr.log"
    log_config["*.error.channel"] = "rasr.log"
    log_config["*.statistics.channel"] = "rasr.log"
    log_config["*.unbuffered"] = True

    log_post_config = RasrConfig()
    log_post_config["*.encoding"] = "UTF-8"
    crp.log_config = log_config  # type: ignore
    crp.log_post_config = log_post_config  # type: ignore
    crp.default_log_channel = "rasr.log"

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
            phon_history_length=1,
            phon_future_length=1,
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

    if label_scorer_config is not None:
        rasr_config.lib_rasr.label_scorer = label_scorer_config
    else:
        rasr_config.lib_rasr.label_scorer = get_no_op_label_scorer_config()

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config

    return recog_rasr_config_path
