from dataclasses import dataclass

from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.pipeline.beam_search_config import (
    BeamSearchConfig,
    beam_search_baseline,
)
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.pipeline.prior_config import (
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

    beam_search: BeamSearchConfig
    lm_scales: list[float]
    prior_scales: list[float]
    ctc_scales: list[float]

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
Specific configurations set below.
"""

"""
V2
"""


def search_ctc_greedy() -> SearchConfig:
    return SearchConfig(
        forward_method="forward_step_greedy_ctc",
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


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
