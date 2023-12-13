"""
Score results, used by the task interface.
"""

from __future__ import annotations
from typing import Optional, Dict, Iterator
import dataclasses
import sisyphus
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
    main_measure_value: tk.Path  # single float value, as text
    report: Optional[tk.Path] = None  # arbitrary format, might be a dir


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


class JoinScoreResultsJob(sisyphus.Job):
    """
    Joins the score results of multiple jobs into one ScoreResultCollection.
    """
    def __init__(self, score_results: Dict[str, ScoreResult]):
        self.score_results = score_results
        self.out_score_results = self.output_path("score_results.json")

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task('run', mini_task=True)

    def run(self):
        """run"""
        import ast
        import json
        res = {}
        for key, score_result in self.score_results.items():
            value_str = open(score_result.main_measure_value.get_path(), "r").read()
            value = ast.literal_eval(value_str)
            res[key] = value
        with open(self.out_score_results.get_path(), "w") as f:
            f.write(json.dumps(res))
            f.write("\n")


def join_score_results(score_results: Dict[str, ScoreResult], main_measure_key: str) -> ScoreResultCollection:
    """join score results"""
    return ScoreResultCollection(
        main_measure_value=score_results[main_measure_key].main_measure_value,
        output=JoinScoreResultsJob(score_results).out_score_results)
