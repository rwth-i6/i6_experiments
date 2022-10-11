"""
Score results, used by the task interface.
"""

from __future__ import annotations

import dataclasses
from typing import Optional
from sisyphus import tk


@dataclasses.dataclass
class RecogOutput:
    """
    Corresponds to the target values of datasets defined by :class:`Task`
    """
    output: tk.Path


@dataclasses.dataclass
class ScoreResult:
    """
    Corresponds to one dataset. E.g. via sclite, or sth else...
    """
    dataset_name: str
    main_measure_value: tk.Path
    report: Optional[tk.Path] = None


@dataclasses.dataclass
class ScoreResultCollection:
    """
    Intended to cover all relevant results over all eval datasets.
    """
    main_measure_value: tk.Path  # e.g. the final best WER% on test-other
    output: tk.Path  # JSON dict with all score outputs


@dataclasses.dataclass(frozen=True)
class MeasureType:
    """measure type, e.g. WER%"""
    short_name: str  # e.g. "WER%"
    lower_is_better: bool = True
