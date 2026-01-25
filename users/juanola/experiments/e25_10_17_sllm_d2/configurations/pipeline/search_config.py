import dataclasses
import warnings
from dataclasses import dataclass

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.beam_search_config import (
    BeamSearchConfig,
    beam_search_baseline,
    greedy,
    beam_search_multiple_beams,
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

    prior: PriorConfig

    # Tunable Parameters # TODO: this could be grouped...
    beam_search: BeamSearchConfig
    lm_scales: list[float]
    prior_scales: list[float]
    ctc_scales: list[float]

    # Other
    forward_method: str = None
    run_ctc_greedy_decoding_last_epoch: bool = False

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

"""
V1
"""


def search_baseline() -> SearchConfig:
    """
    V2 should be used!
    :return:
    """
    warnings.warn(
        "[BUG] Doesn't use beam search correctly -> essentially greedy",
        DeprecationWarning,
        stacklevel=2,
    )
    return SearchConfig(
        batch_size=15_000,
        batch_size_factor=160,
        use_gpu=True,
        gpu_memory=11,
        beam_search=beam_search_baseline(),
        prior=prior_v1(),
        lm_scales=[0.0],
        prior_scales=[0.0],
        ctc_scales=[0.0],
        avg_best_loss_name="dev_loss_ce",
        max_seqs=200,
    )


def search_baseline_with_ctc_gd() -> SearchConfig:
    warnings.warn(
        "[BUG] Doesn't use beam search correctly -> essentially greedy",
        DeprecationWarning,
        stacklevel=2,
    )
    return dataclasses.replace(search_baseline(), run_ctc_greedy_decoding_last_epoch=True)


def greedy_search() -> SearchConfig:
    warnings.warn(
        "[BUG] Doesn't use beam search correctly -> essentially greedy",
        DeprecationWarning,
        stacklevel=2,
    )
    return dataclasses.replace(search_baseline(), beam_search=greedy())


def greedy_search_v2() -> SearchConfig:
    warnings.warn(
        "[BUG] Doesn't use beam search correctly -> essentially greedy",
        DeprecationWarning,
        stacklevel=2,
    )
    return dataclasses.replace(search_baseline(), batch_size=13_000, beam_search=greedy())


"""
V2
"""


def search_baseline_v2() -> SearchConfig:
    return SearchConfig(
        forward_method="forward_step_v2",
        batch_size=15_000,
        batch_size_factor=160,
        use_gpu=True,
        gpu_memory=11,
        beam_search=beam_search_baseline(),
        prior=prior_v1(),
        lm_scales=[None],  # Not used!
        prior_scales=[None],  # Not used!
        ctc_scales=[None],  # Not used!
        avg_best_loss_name="dev_loss_ce",
        max_seqs=200,
    )


def search_baseline_v2_multiple_beams() -> SearchConfig:
    return dataclasses.replace(search_baseline_v2(), beam_search=beam_search_multiple_beams())


"""
ctc decoding
"""


def search_baseline_ctc_decoding_11gb() -> SearchConfig:
    return SearchConfig(
        forward_method="forward_step_ctc_decoding",
        batch_size=5_000,
        batch_size_factor=160,
        use_gpu=True,
        gpu_memory=11,  # TODO: perhaps increase this
        beam_search=beam_search_baseline(),
        prior=prior_v1(),
        avg_best_loss_name="dev_loss_ce",
        max_seqs=200,
        lm_scales=[1.0],
        ctc_scales=[1.0],
        prior_scales=[0.0],
    )


def search_baseline_ctc_decoding_24gb() -> SearchConfig:
    return dataclasses.replace(search_baseline_ctc_decoding_11gb(), batch_size=10_000, gpu_memory=24)


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
