import glob
import json
import os
import re
import statistics
import tarfile
from typing import Dict, Iterable, List, Tuple

from sisyphus import Job, Task, tk


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_EPOCH = re.compile(
    r"Epoch (\d+): Trained (\d+) steps, (\d+):(\d\d):(\d\d) elapsed \(([\d.]+)% computing time"
    r"(?:, (.*?) (bound slack|padding)\))?"
)
_SLACK = re.compile(r"([^\s,:]+): ([\d.]+)%")
_STEP = re.compile(
    r"train, step (\d+), .*num_seqs (\d+), .*mem_usage:cuda:0 ([\d.]+)GB"
    r"(?:, mem_graph_pool:cuda:0 ([\d.]+)GB)?, ([\d.]+) sec/step"
)
_STEP_LOSSES = re.compile(r"ep (\d+) train, step (\d+), (.*)$")
_BATCH_INFO = ("num_seqs", "max_size:", "sum_size:", "mem_usage:", "mem_graph_pool:")
_LINE_EPOCH = re.compile(r"\bep (\d+) train, step \d+,|Epoch (\d+): Trained ")
_ENGINE_RUN_LOG = re.compile(r"(engine/)?[^/]*\.run\.[^/.]+\.\d+")
_RUN_LOG = re.compile(r"(.*/)?log\.run\.\d+")


def _rank_zero(lines: Iterable[str]) -> List[str]:
    """
    Keeps the lines of the first rank, colour codes stripped.

    :param lines: raw log lines
    :return: the kept lines
    """
    kept = []
    for line in lines:
        line = _ANSI.sub("", line)
        if line.startswith("[default") and not line.startswith("[default0]:"):
            continue
        kept.append(line)
    return kept


def _epoch_lines(lines: Iterable[str]) -> Dict[int, List[str]]:
    """
    Groups the train step and epoch summary lines of the first rank of one run by epoch.

    :param lines: raw log lines of the run
    :return: per epoch its lines without colour codes
    """
    epochs = {}
    for line in _rank_zero(lines):
        m = _LINE_EPOCH.search(line)
        if m:
            epochs.setdefault(int(m.group(1) or m.group(2)), []).append(line)
    return epochs


def _read_runs(job_dir: str, parse) -> list:
    """
    Reads every run of a training job in the order they ran, from the engine logs that keep each slice of a resubmit.

    A finished training has its logs packed into ``finished.tar.gz`` by the cleanup, with ``log.run.1``
    left empty, so a reader that only globs the live logs finds nothing for every finished row.

    :param job_dir: the sisyphus job directory of the training, live or archived by the cleanup
    :param parse: turns the lines of one run into whatever the caller wants
    :return: per run the result of ``parse``
    """
    runs = []
    engine = sorted(
        (p for p in glob.glob(os.path.join(job_dir, "engine", "*")) if _ENGINE_RUN_LOG.fullmatch(os.path.basename(p))),
        key=lambda p: (os.path.getmtime(p), p),
    )
    logs = engine or sorted(p for p in glob.glob(os.path.join(job_dir, "log.run.*")) if _RUN_LOG.fullmatch(p))
    for path in logs:
        with open(path, "rt", errors="replace") as f:
            runs.append(parse(f))

    archive = os.path.join(job_dir, "finished.tar.gz")
    if not any(runs) and os.path.exists(archive):
        runs = []
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if (m.isfile() or m.islnk()) and m.name.startswith("engine/")]
            members = [m for m in members if _ENGINE_RUN_LOG.fullmatch(m.name)]
            members = members or [m for m in tar.getmembers() if _RUN_LOG.fullmatch(m.name)]
            for member in sorted(members, key=lambda m: (m.mtime, m.name)):
                text = tar.extractfile(member).read().decode("utf-8", errors="replace")
                runs.append(parse(text.splitlines(True)))

    return runs


def _runs(job_dir: str) -> List[Dict[int, List[str]]]:
    """
    Reads every run of a training job, its lines grouped by epoch.

    :param job_dir: the sisyphus job directory of the training, live or archived by the cleanup
    :return: per run its lines grouped by epoch
    """
    return _read_runs(job_dir, _epoch_lines)


def rank_zero_log_text(job_dir: str) -> str:
    """
    Reads the whole first-rank log of a training, every run in the order they ran.

    Unlike :func:`_rank_zero_lines` this keeps every line and not only the train step and epoch
    summaries, so a caller can also read what the engine prints once at start-up.

    :param job_dir: the sisyphus job directory of the training, live or archived by the cleanup
    :return: the log with the colour codes stripped and the other ranks dropped
    """
    return "".join("".join(run) for run in _read_runs(job_dir, _rank_zero))


def _rank_zero_lines(job_dir: str) -> List[str]:
    """
    Reads the train step and epoch summary lines of the first rank of a training, each epoch from the last run with it.

    :param job_dir: the sisyphus job directory of the training
    :return: the lines without colour codes, in epoch order
    """
    by_epoch = {}
    for run in _runs(job_dir):
        by_epoch.update(run)
    return [line for epoch in sorted(by_epoch) for line in by_epoch[epoch]]


def summarize_log(lines: List[str]) -> Dict:
    """
    Collects the epoch summaries and the step statistics of a training log.

    :param lines: the log lines of one rank
    :return: the epochs, the median step time, the peak memory and the mean batch size
    """
    epochs = []
    step_seconds = []
    mem_gb = []
    pool_gb = []
    num_seqs = []
    for line in lines:
        m = _EPOCH.search(line)
        if m:
            epochs.append(
                {
                    "epoch": int(m.group(1)),
                    "steps": int(m.group(2)),
                    "seconds": int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5)),
                    "computing_time_percent": float(m.group(6)),
                    "slack_percent": {key: float(value) for key, value in _SLACK.findall(m.group(7) or "")},
                    "slack_kind": m.group(8),
                }
            )
            continue
        m = _STEP.search(line)
        if m:
            num_seqs.append(int(m.group(2)))
            mem_gb.append(float(m.group(3)))
            pool_gb.append(float(m.group(4) or 0.0))
            step_seconds.append(float(m.group(5)))
    return {
        "epochs": epochs,
        "logged_steps": len(step_seconds),
        "median_sec_per_step": round(statistics.median(step_seconds), 4) if step_seconds else None,
        "mean_sec_per_step": round(statistics.fmean(step_seconds), 4) if step_seconds else None,
        "max_mem_gb": max(mem_gb) if mem_gb else None,
        "max_graph_pool_gb": max(pool_gb) if pool_gb else None,
        "mean_num_seqs": round(statistics.fmean(num_seqs), 2) if num_seqs else None,
        "max_num_seqs": max(num_seqs) if num_seqs else None,
    }


def step_losses(lines: List[str]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Collects the per-step train losses of a training log.

    :param lines: the log lines of one rank
    :return: per epoch and step the logged losses by name
    """
    res = {}
    for line in lines:
        m = _STEP_LOSSES.search(line.rstrip("\n"))
        if not m:
            continue
        losses = {}
        for item in m.group(3).split(", "):
            name, _, value = item.partition(" ")
            if not value or name.startswith(_BATCH_INFO) or value.endswith("sec/step"):
                break
            try:
                losses[name] = float(value)
            except ValueError:
                break
        res[(int(m.group(1)), int(m.group(2)))] = losses
    return res


class TrainingThroughputJob(Job):
    """Reads the epoch times and the step statistics of a finished training from its RETURNN log."""

    def __init__(self, *, learning_rates: tk.Path):
        self.learning_rates = learning_rates
        self.out_summary = self.output_path("throughput.json")
        self.out_report = self.output_path("throughput.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        job_dir = os.path.dirname(os.path.dirname(self.learning_rates.get_path()))
        summary = summarize_log(_rank_zero_lines(job_dir))
        with open(self.out_summary.get_path(), "wt") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
            f.write("\n")
        with open(self.out_report.get_path(), "wt") as f:
            for epoch in summary["epochs"]:
                slack = ", ".join(f"{key} {value} percent" for key, value in epoch["slack_percent"].items())
                f.write(
                    f"epoch {epoch['epoch']}: {epoch['steps']} steps, {epoch['seconds']} s,"
                    f" {epoch['computing_time_percent']} percent computing time"
                    + (f", {epoch['slack_kind']} {slack}" if slack else "")
                    + "\n"
                )
            f.write(
                f"median {summary['median_sec_per_step']} s per step over {summary['logged_steps']} steps,"
                f" peak {summary['max_mem_gb']} GB of which {summary['max_graph_pool_gb']} GB graph pool,"
                f" mean {summary['mean_num_seqs']} seqs per step\n"
            )
