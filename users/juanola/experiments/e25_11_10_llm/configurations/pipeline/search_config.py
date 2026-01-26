from dataclasses import dataclass

from .beam_search_config import BeamSearchConfig, beam_search_baseline
from .prior_config import prior_v1, PriorConfig


@dataclass(frozen=True)
class SearchConfig:
    """
    Search (inference) configuration base dataclass.

    Can contain default values.
    """

    batch_size: int
    use_gpu: bool
    gpu_memory: int  # Avoid using bigger that 11Gb
    avg_best_loss_name: str
    max_seqs: int

    forward_method: str = None

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


def search_baseline_v2() -> SearchConfig:
    return SearchConfig(
        forward_method="perplexity_forward_step",
        batch_size=15_000,
        use_gpu=True,
        gpu_memory=11,
        max_seqs=200,
        avg_best_loss_name="dev_loss_ce",
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
