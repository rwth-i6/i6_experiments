from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    """
    Search (inference) configuration base dataclass.

    Can contain default values.
    """
    batch_size: int

    gpu_memory: int  # Avoid using bigger that 11Gb


"""
Specific configurations set below.
"""


def search_baseline() -> SearchConfig:
    return SearchConfig(
        batch_size=15_000,
        gpu_memory=11,
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
