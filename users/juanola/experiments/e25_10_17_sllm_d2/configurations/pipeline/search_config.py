from dataclasses import dataclass

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.beam_search_config import \
    BeamSearchConfig, beam_search_baseline


@dataclass(frozen=True)
class SearchConfig:
    """
    Search (inference) configuration base dataclass.

    Can contain default values.
    """
    batch_size: int

    gpu_memory: int  # Avoid using bigger that 11Gb

    beam_search: BeamSearchConfig


"""
Specific configurations set below.
"""


def search_baseline() -> SearchConfig:
    return SearchConfig(
        batch_size=15_000,
        gpu_memory=11,

        beam_search=beam_search_baseline(),
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
