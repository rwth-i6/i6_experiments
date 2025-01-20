__all__ = [
    "get_no_op_label_scorer_config",
    "get_combine_label_scorer_config",
    "RasrRecogOptions",
    "get_rasr_config_file",
]

from dataclasses import dataclass
from typing import List, Optional, Tuple

from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob, build_config_from_mapping
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
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
    max_beam_size: int = 1
    top_k_tokens: Optional[int] = None
    score_threshold: Optional[float] = None
    blank_index: Optional[int] = None
    sentence_end_index: Optional[int] = None
    allow_label_loop: bool = False


def get_rasr_config_file(
    recog_options: RasrRecogOptions,
    label_scorer_config: Optional[RasrConfig] = None,
) -> tk.Path:
    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.set_executables(rasr_binary_path=rasr_binary_path)

    rasr_config, rasr_post_config = build_config_from_mapping(crp=crp, mapping={}, include_log_config=True)

    rasr_config.lib_rasr = RasrConfig()

    rasr_config.lib_rasr.lexicon = RasrConfig()
    rasr_config.lib_rasr.lexicon.type = "vocab-text"
    rasr_config.lib_rasr.lexicon.file = recog_options.vocab_file

    rasr_config.lib_rasr.search_algorithm = RasrConfig()
    if recog_options.max_beam_size == 1 or recog_options.score_threshold == 0:
        rasr_config.lib_rasr.search_algorithm.type = "lexiconfree-greedy-search"
    else:
        rasr_config.lib_rasr.search_algorithm.type = "lexiconfree-beam-search"
        rasr_config.lib_rasr.search_algorithm.max_beam_size = recog_options.max_beam_size
        if recog_options.top_k_tokens is not None:
            rasr_config.lib_rasr.search_algorithm.top_k_tokens = recog_options.top_k_tokens
        if recog_options.score_threshold is not None:
            rasr_config.lib_rasr.search_algorithm.score_threshold = recog_options.score_threshold

    if recog_options.blank_index is not None:
        rasr_config.lib_rasr.search_algorithm.use_blank = True
        rasr_config.lib_rasr.search_algorithm.blank_label_index = recog_options.blank_index

    if recog_options.sentence_end_index is not None:
        rasr_config.lib_rasr.search_algorithm.use_sentence_end = True
        rasr_config.lib_rasr.search_algorithm.sentence_end_index = recog_options.sentence_end_index

    rasr_config.lib_rasr.search_algorithm.allow_label_loop = recog_options.allow_label_loop

    if label_scorer_config is not None:
        rasr_config.lib_rasr.label_scorer = label_scorer_config
    else:
        rasr_config.lib_rasr.label_scorer = RasrConfig()
        rasr_config.lib_rasr.label_scorer.type = "no-op"

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config

    return recog_rasr_config_path
