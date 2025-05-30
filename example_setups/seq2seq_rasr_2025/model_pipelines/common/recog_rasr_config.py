__all__ = [
    "get_no_op_label_scorer_config",
    "get_combine_label_scorer_config",
    "RasrRecogOptions",
    "get_rasr_config_file",
]

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob, build_config_from_mapping
from i6_core.rasr.crp import CommonRasrParameters
from sisyphus import tk

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


@dataclass
class RasrRecogOptions:
    vocab_file: tk.Path
    lexicon_file: Optional[tk.Path] = None
    max_beam_size: int = 1
    intermediate_max_beam_size: Optional[int] = None
    score_threshold: Optional[float] = None
    intermediate_score_threshold: Optional[float] = None
    blank_index: Optional[int] = None
    sentence_end_index: Optional[int] = None
    collapse_repeated_labels: bool = False
    length_norm_scale: Optional[float] = None
    lm_config: Optional[RasrConfig] = None
    am_config: Optional[RasrConfig] = None


def get_rasr_config_file(
    recog_options: RasrRecogOptions,
    label_scorer_config: Optional[Union[List[RasrConfig], RasrConfig]] = None,
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

    if recog_options.lm_config is not None:
        rasr_config.lib_rasr.lm = recog_options.lm_config

    if recog_options.am_config is not None:
        rasr_config.lib_rasr.acoustic_model = recog_options.am_config

    rasr_config.lib_rasr.search_algorithm = RasrConfig()
    if recog_options.lexicon_file is not None:
        rasr_config.lib_rasr.search_algorithm.type = "tree-timesync-beam-search"
        rasr_config.lib_rasr.lexicon.file = recog_options.lexicon_file
        if recog_options.blank_index is not None and recog_options.collapse_repeated_labels:
            rasr_config.lib_rasr.search_algorithm.tree_builder_type = "ctc"
        elif recog_options.blank_index is not None and not recog_options.collapse_repeated_labels:
            rasr_config.lib_rasr.search_algorithm.tree_builder_type = "rna"
        else:
            raise NotImplementedError
    else:
        rasr_config.lib_rasr.lexicon.file = f"vocab-text:{recog_options.vocab_file}"
        if recog_options.sentence_end_index is None:
            rasr_config.lib_rasr.search_algorithm.type = "lexiconfree-timesync-beam-search"
        else:
            rasr_config.lib_rasr.search_algorithm.type = "lexiconfree-labelsync-beam-search"

    rasr_config.lib_rasr.search_algorithm.max_beam_size = recog_options.max_beam_size
    if recog_options.intermediate_max_beam_size is not None:
        rasr_config.lib_rasr.search_algorithm.intermediate_max_beam_size = recog_options.intermediate_max_beam_size
    if recog_options.score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.score_threshold = recog_options.score_threshold
    if recog_options.intermediate_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.intermediate_score_threshold = recog_options.intermediate_score_threshold

    if recog_options.blank_index is not None:
        rasr_config.lib_rasr.search_algorithm.use_blank = True
        rasr_config.lib_rasr.search_algorithm.blank_label_index = recog_options.blank_index
        if recog_options.lexicon_file is not None and recog_options.collapse_repeated_labels:
            rasr_config.lib_rasr.search_algorithm.force_blank_between_repeated_labels_across_words = True

    if recog_options.sentence_end_index is not None:
        rasr_config.lib_rasr.search_algorithm.sentence_end_index = recog_options.sentence_end_index

    if recog_options.length_norm_scale is not None:
        rasr_config.lib_rasr.search_algorithm.length_norm_scale = recog_options.length_norm_scale

    rasr_config.lib_rasr.search_algorithm.collapse_repeated_labels = recog_options.collapse_repeated_labels

    rasr_config.lib_rasr.search_algorithm.log_stepwise_statistics = True

    if label_scorer_config is not None:
        rasr_config.lib_rasr.label_scorer = label_scorer_config
    else:
        rasr_config.lib_rasr.label_scorer = get_no_op_label_scorer_config()

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config

    return recog_rasr_config_path
