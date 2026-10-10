import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy
from sisyphus import Job, Task, tk

from .throughput import _rank_zero, _rank_zero_lines, step_losses

STEP_TOLERANCE = 5e-3
MEAN_TOLERANCE = 1e-3
_RANDOM_MEAN_TOLERANCE = 1e-2
_PARITY_STEPS = 200
_GRAD_NORM = "grad_norm"
_TOP = 40
_TABLES = ("categories", "passes", "ops", "modules", "aten", "kernels")
_WARM_RUN_TABLES = ("categories", "passes", "ops", "modules")
_TRACE_NAME = "torch_profile.json"
_SUMMARY_NAME = "torch_profile_summary.json"


def torch_profile_config(*, capture_step: int, replay_steps: int = 3) -> dict:
    """
    Builds the torch_profile option recording the capture step and the first replays of a captured training.

    :param capture_step: the train step that captures the graph, the number of eager warmup steps
    :param replay_steps: the number of recorded replay steps after the capture step
    :return: the option dict, which writes the trace and its summary next to the learning rates and trains on
    """
    return {
        "exit_after": False,
        "profile_memory": False,
        "with_flops": False,
        "record_shapes": True,
        "with_stack": True,
        "schedule": {"skip_first": capture_step - 1, "wait": 0, "warmup": 1, "active": 1 + replay_steps, "repeat": 1},
        "trace_path": f"../output/{_TRACE_NAME}",
        "summary_path": f"../output/{_SUMMARY_NAME}",
    }


def read_learning_rates(path: str) -> Dict[int, Dict[str, Any]]:
    """
    Reads the per-epoch scores of a RETURNN learning rates file.

    :param path: the learning rates file
    :return: per epoch the scores by name
    """
    with open(path, "rt") as f:
        text = f.read()
    data = eval(
        text,
        {
            "EpochData": lambda learningRate=None, error=None, **_kwargs: dict(error or {}),
            "nan": float("nan"),
            "inf": float("inf"),
            "np": numpy,
        },
    )
    return {int(epoch): scores for epoch, scores in data.items()}


def compare_losses(
    lines: List[str],
    reference_lines: List[str],
    scores: Dict[int, Dict[str, Any]],
    reference_scores: Dict[int, Dict[str, Any]],
    *,
    num_steps: int,
    step_tolerance: Optional[float],
    mean_tolerance: float,
) -> Dict[str, Any]:
    """
    Compares the per-step train losses over the first steps and the epoch mean train losses of two trainings.

    :param lines: the rank zero log lines of the training
    :param reference_lines: the rank zero log lines of the reference training
    :param scores: the per-epoch scores of the training
    :param reference_scores: the per-epoch scores of the reference training
    :param num_steps: the number of first steps compared one by one
    :param step_tolerance: the largest per-step difference relative to the reference value or to one,
        None to judge only the epoch means
    :param mean_tolerance: the largest relative difference of an epoch mean train loss
    :return: the differences and the verdict
    """
    steps = step_losses(lines)
    reference_steps = step_losses(reference_lines)
    compared = sorted(set(steps) & set(reference_steps))[:num_steps]
    worst = {}
    first_beyond = None
    for key in compared:
        for name, reference_value in reference_steps[key].items():
            if name not in steps[key]:
                continue
            value = steps[key][name]
            difference = abs(value - reference_value) / max(1.0, abs(reference_value))
            if name not in worst or not difference <= worst[name]:
                worst[name] = difference
            judged = not name.startswith(_GRAD_NORM)
            if judged and step_tolerance is not None and first_beyond is None and not difference <= step_tolerance:
                first_beyond = {
                    "epoch": key[0],
                    "step": key[1],
                    "name": name,
                    "value": value,
                    "reference": reference_value,
                }
    means = []
    for epoch in sorted(set(scores) & set(reference_scores)):
        for name, reference_value in sorted(reference_scores[epoch].items()):
            if not name.startswith("train_loss_") or name not in scores[epoch]:
                continue
            value = scores[epoch][name]
            relative = abs(value - reference_value) / max(abs(reference_value), 1e-12)
            means.append(
                {
                    "epoch": epoch,
                    "name": name,
                    "value": value,
                    "reference": reference_value,
                    "relative_difference": relative,
                    "passed": relative <= mean_tolerance,
                }
            )
    steps_passed = step_tolerance is None or (bool(compared) and first_beyond is None)
    judged_means = [mean for mean in means if not mean["name"].startswith(f"train_loss_{_GRAD_NORM}")]
    return {
        "compared_steps": len(compared),
        "step_tolerance": step_tolerance,
        "worst_step_difference": worst,
        "first_step_beyond_tolerance": first_beyond,
        "mean_tolerance": mean_tolerance,
        "epoch_means": means,
        "passed": steps_passed and bool(judged_means) and all(mean["passed"] for mean in judged_means),
    }


def loss_parity(
    *,
    learning_rates: str,
    reference_learning_rates: str,
    reference_log: Optional[str],
    step_tolerance: Optional[float],
    mean_tolerance: float,
    num_steps: int,
) -> Dict[str, Any]:
    """
    Compares the train losses of a training with a reference training from their logs and learning rates files.

    :param learning_rates: the learning rates file of the training, two levels below its job directory
    :param reference_learning_rates: the learning rates file of the reference training
    :param reference_log: the run log holding the compared steps of the reference, None for the run logs of its job
    :param step_tolerance: the largest per-step difference relative to the reference value or to one,
        None to judge only the epoch means
    :param mean_tolerance: the largest relative difference of an epoch mean train loss
    :param num_steps: the number of first steps compared one by one
    :return: the differences and the verdict
    """
    if reference_log is None:
        reference_lines = _rank_zero_lines(os.path.dirname(os.path.dirname(reference_learning_rates)))
    else:
        with open(reference_log, "rt", errors="replace") as f:
            reference_lines = _rank_zero(f)
    return compare_losses(
        _rank_zero_lines(os.path.dirname(os.path.dirname(learning_rates))),
        reference_lines,
        read_learning_rates(learning_rates),
        read_learning_rates(reference_learning_rates),
        num_steps=num_steps,
        step_tolerance=step_tolerance,
        mean_tolerance=mean_tolerance,
    )


def _row(name: str, ms: float, reference_ms: Optional[float]) -> str:
    """
    Formats one table row, with the reference value and the change when a reference is given.

    :param name: the row name
    :param ms: the milliseconds per step
    :param reference_ms: the milliseconds per step of the reference, None without a reference
    :return: the row text
    """
    if reference_ms is None:
        return f"{ms:10.3f}  {name}"
    return f"{ms:10.3f}{reference_ms:10.3f}{ms - reference_ms:+10.3f}  {name}"


def _rows(rows: Sequence[Dict[str, Any]], reference_rows: Optional[Sequence[Dict[str, Any]]]) -> List[str]:
    """
    Formats a table of milliseconds per step merged with the reference table, largest first.

    :param rows: the rows with name and ms
    :param reference_rows: the rows of the reference, None without a reference
    :return: the row texts
    """
    values = {row["name"]: row["ms"] for row in rows}
    references = {row["name"]: row["ms"] for row in reference_rows} if reference_rows is not None else {}
    names = sorted(set(values) | set(references), key=lambda n: -max(values.get(n, 0.0), references.get(n, 0.0)))
    return [
        _row(name, values.get(name, 0.0), references.get(name, 0.0) if reference_rows is not None else None)
        for name in names[:_TOP]
    ]


def _replay_step_means(summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Averages the step times and the kernel time per phase over the replay steps of a summary.

    :param summary: the profile summary
    :return: the rows of wall, gpu busy and gpu idle time, and the rows of the phases
    """
    steps = [step for step in summary["steps"] if step["step"] in summary["table_steps"]]
    count = max(len(steps), 1)
    totals = [
        {"name": name, "ms": sum(step[key] for step in steps) / count}
        for name, key in (("wall", "wall_ms"), ("gpu busy", "gpu_busy_ms"), ("gpu idle", "gpu_idle_ms"))
    ]
    phases = defaultdict(float)
    for step in steps:
        for phase in step["phases"]:
            phases[phase["name"]] += phase["kernel_ms"] / count
    return totals, [{"name": name, "ms": ms} for name, ms in phases.items()]


def format_profile(summary: Dict[str, Any], reference: Optional[Dict[str, Any]] = None) -> str:
    """
    Formats a profile summary as tables of milliseconds per replay step, next to a reference summary when given.

    :param summary: the profile summary of a training
    :param reference: the profile summary of the reference training
    :return: the report
    """
    replay = summary["replay"]
    lines = [
        f"{summary['trace']}, replay steps {summary['table_steps']}, {replay['num_replays']} replays with"
        f" {replay['kernels_per_replay']} kernels each ({replay['without_launch']} without a launch,"
        f" {replay['shared_launch']} with a shared launch), matched fraction {replay['matched_fraction']},"
        f" {summary['unattributed_kernels']} unattributed kernels",
        "milliseconds per replay step" + ("" if reference is None else ", then the reference and the change"),
    ]
    totals, phases = _replay_step_means(summary)
    reference_totals, reference_phases = _replay_step_means(reference) if reference is not None else (None, None)
    sections = [("step", totals, reference_totals), ("phases by kernel time", phases, reference_phases)]
    for key in _TABLES:
        sections.append((key, summary.get(key, []), reference.get(key, []) if reference is not None else None))
    for key in _WARM_RUN_TABLES:
        reference_rows = reference["warm_run"].get(key, []) if reference is not None else None
        rows = summary["warm_run"].get(key, [])
        sections.append((f"{key} in one eager warm run at the bound shapes", rows, reference_rows))
    for title, rows, reference_rows in sections:
        lines.append(title)
        lines.extend(_rows(rows, reference_rows))
    return "\n".join(lines) + "\n"


class TrainingProfileJob(Job):
    """
    Summarizes the torch profiler trace a profiled training wrote next to its learning rates, against a reference.
    """

    def __init__(self, *, learning_rates: tk.Path, returnn_root: tk.Path, reference: Optional[tk.Path] = None):
        """
        :param learning_rates: the learning rates output of the profiled training
        :param returnn_root: the RETURNN checkout whose trace summarizer runs
        :param reference: the summary output of the profile job of the reference training
        """
        self.learning_rates = learning_rates
        self.returnn_root = returnn_root
        self.reference = reference
        self.out_summary = self.output_path("profile_summary.json")
        self.out_report = self.output_path("profile.txt")

    def tasks(self):
        """
        :return: the run task
        """
        yield Task("run", mini_task=True)

    def run(self):
        """
        Summarizes the trace and writes the report.
        """
        trace = os.path.join(os.path.dirname(self.learning_rates.get_path()), _TRACE_NAME)
        script = os.path.join(self.returnn_root.get_path(), "returnn", "torch", "util", "profile_summary.py")
        subprocess.check_call([sys.executable, script, trace, self.out_summary.get_path()])
        with open(self.out_summary.get_path(), "rt") as f:
            summary = json.load(f)
        reference = None
        if self.reference is not None:
            with open(self.reference.get_path(), "rt") as f:
                reference = json.load(f)
        with open(self.out_report.get_path(), "wt") as f:
            f.write(format_profile(summary, reference))


class TrainingLossParityJob(Job):
    """
    Compares the train losses of a training with a reference training, per step over the first steps and as epoch means.
    """

    __sis_hash_exclude__ = {"reference_log": None}

    def __init__(
        self,
        *,
        learning_rates: tk.Path,
        reference_learning_rates: tk.Path,
        step_tolerance: Optional[float],
        mean_tolerance: float,
        num_steps: int = _PARITY_STEPS,
        reference_log: Optional[tk.Path] = None,
    ):
        """
        :param learning_rates: the learning rates output of the training
        :param reference_learning_rates: the learning rates output of the reference training
        :param step_tolerance: the largest per-step difference relative to the reference value or to one,
            None to judge only the epoch means
        :param mean_tolerance: the largest relative difference of an epoch mean train loss
        :param num_steps: the number of first steps compared one by one
        :param reference_log: the run log holding the compared steps of the reference, None for the run logs of its job
        """
        self.learning_rates = learning_rates
        self.reference_learning_rates = reference_learning_rates
        self.step_tolerance = step_tolerance
        self.mean_tolerance = mean_tolerance
        self.num_steps = num_steps
        self.reference_log = reference_log
        self.out_parity = self.output_path("parity.json")

    def tasks(self):
        """
        :return: the run task
        """
        yield Task("run", mini_task=True)

    def run(self):
        """
        Compares the losses and writes the differences with the verdict.
        """
        result = loss_parity(
            learning_rates=self.learning_rates.get_path(),
            reference_learning_rates=self.reference_learning_rates.get_path(),
            reference_log=self.reference_log.get_path() if self.reference_log is not None else None,
            step_tolerance=self.step_tolerance,
            mean_tolerance=self.mean_tolerance,
            num_steps=self.num_steps,
        )
        with open(self.out_parity.get_path(), "wt") as f:
            json.dump(result, f, indent=1, sort_keys=True)
            f.write("\n")


def register_profiled_rows(
    *, base_prefix: str, rows: Sequence[Tuple[str, Any, Optional[float]]], returnn_root: tk.Path
) -> None:
    """
    Registers the profile report of every profiled training, for later trainings against the first with the loss parity.

    :param base_prefix: the output prefix of the trainings
    :param rows: per training its name, its training job and the per-step loss tolerance against the first training,
        None to judge only the epoch means
    :param returnn_root: the RETURNN checkout whose trace summarizer runs
    """
    (reference_name, reference_job, _), *variants = rows
    reference = TrainingProfileJob(learning_rates=reference_job.out_learning_rates, returnn_root=returnn_root)
    tk.register_output(f"{base_prefix}/{reference_name}/profile.txt", reference.out_report)
    for name, train_job, step_tolerance in variants:
        profile = TrainingProfileJob(
            learning_rates=train_job.out_learning_rates, returnn_root=returnn_root, reference=reference.out_summary
        )
        tk.register_output(f"{base_prefix}/{name}/profile.txt", profile.out_report)
        parity = TrainingLossParityJob(
            learning_rates=train_job.out_learning_rates,
            reference_learning_rates=reference_job.out_learning_rates,
            step_tolerance=step_tolerance,
            mean_tolerance=MEAN_TOLERANCE if step_tolerance is not None else _RANDOM_MEAN_TOLERANCE,
        )
        tk.register_output(f"{base_prefix}/{name}/loss_parity.json", parity.out_parity)
