from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class PriorConfig:
    """
    Prior (inference) configuration base dataclass.

    Can contain default values.
    """


    batch_size_factor: int
    batch_size: int

    cpu_memory: int

    forward_method: str

    debug_returnn_param: bool = True

    static_prior_file: Optional[str] = None

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass


"""
Specific configurations set below.
"""


def prior_v1() -> PriorConfig:
    return PriorConfig(batch_size_factor=500, batch_size=16_000, cpu_memory=12, forward_method="prior_step_v1")

def static_prior() -> PriorConfig:
    return replace(prior_v1(), static_prior_file="/u/marti.juanola/experiments/25_10_17_sllm_d2/work/i6_experiments/users/juanola/sisyphus_jobs/prior/ComputePriorWithoutBlank/ComputePriorWithoutBlank.EgWQfj8EJzJR/output/prior_without_blank.pt")

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
