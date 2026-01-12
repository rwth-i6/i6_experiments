import dataclasses
from dataclasses import dataclass

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.beam_search_config import (
    BeamSearchConfig,
    beam_search_baseline,
    greedy,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.prior_config import (
    prior_v1,
    PriorConfig,
)


@dataclass(frozen=True)
class SearchConfig:
    """
    Search (inference) configuration base dataclass.

    Can contain default values.
    """

    batch_size: int
    batch_size_factor: int
    use_gpu: bool
    gpu_memory: int  # Avoid using bigger that 11Gb
    avg_best_loss_name: str
    max_seqs: int

    # WIP
    prior: PriorConfig

    beam_search: BeamSearchConfig
    # ctc scores?
    lm_scales: list[float]
    prior_scales: list[float]

    forward_method: str = None
    run_ctc_greedy_decoding: bool = False

    debug_returnn_param: bool = True

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        if self.use_gpu:
            assert self.gpu_memory is not None, "if use_gpu is set please set gpu_memory variable."


"""
parameter sets
"""

_LM_PRIOR_SCALES = dict(
    lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8],
    prior_scales=[0.7, 0.9],
)

"""
Specific configurations set below.
"""


def search_baseline() -> SearchConfig:
    return SearchConfig(
        batch_size=15_000,
        batch_size_factor=160,
        use_gpu=True,
        gpu_memory=11,
        beam_search=beam_search_baseline(),
        prior=prior_v1(),
        lm_scales=[0.0],
        prior_scales=[0.0],
        avg_best_loss_name="dev_loss_ce",
        max_seqs=200,
    )


def search_baseline_with_ctc_gd() -> SearchConfig:
    return dataclasses.replace(search_baseline(), run_ctc_greedy_decoding=True)


def greedy_search() -> SearchConfig:
    return dataclasses.replace(search_baseline(), beam_search=greedy())


def greedy_search_v2() -> SearchConfig:
    return dataclasses.replace(search_baseline(), batch_size=13_000, beam_search=greedy())


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
