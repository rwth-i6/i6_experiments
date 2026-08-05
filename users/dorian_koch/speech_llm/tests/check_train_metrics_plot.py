"""Guard: TrainMetricsPlot must survive a PREEMPTED-AND-RESUMED run.

``metrics.train.jsonl`` is append-ordered, not step-ordered. When a run is preempted it restarts
from its last complete checkpoint and replays every step in between, so the step counter runs
backwards mid-file. Observed live on ``a8_fast`` (2026-08-05): step 1930 -> 1510, 43 steps replayed.

Two things break if that is ignored, and this file guards both:

1.  The figure draws a polyline travelling right-to-left across the plot -- the visible symptom,
    and the one a reader spots immediately.
2.  Worse and silent: the replayed steps are double-counted, and ``final_loss`` /
    ``final_probe_accuracy`` are read off whichever point happens to be last in FILE order. After a
    resume that can be a point from the attempt that DIED -- i.e. from weights that were rolled
    back and never trained forward.

The third case here is unrelated to resumes but shares the plumbing: a run whose probe stopped
reporting while training continued (``a8_long``, whose probe disabled itself at step 900 on an I/O
error and left 5,200 steps unmeasured). ``last_probe_step`` vs ``last_loss_step`` is what makes that
detectable instead of looking like a finished measurement.

Run: ``CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_train_metrics_plot.py``
"""

import json
import os
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm.train_metrics_plot import TrainMetricsPlot  # noqa: E402


def _write(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _loss(step, loss):
    return {"step": step, "loss": loss}


def _probe(step, acc):
    return {"kind": "knowledge", "step": step, "knowledge_accuracy": acc, "knowledge_n": 64}


def check_resumed_run():
    """A run preempted at 40 and resumed from 20, exactly the a8_fast shape."""
    rows = []
    for s in range(10, 50, 10):  # 10, 20, 30, 40
        rows.append(_loss(s, 1.0 - s / 100.0))
    rows.append(_probe(20, 0.20))
    rows.append(_probe(40, 0.40))
    # --- preemption: restart from the step-20 checkpoint, replay 30 and 40 with DIFFERENT values
    for s in (30, 40, 50):
        rows.append(_loss(s, 0.5 - s / 100.0))
    rows.append(_probe(40, 0.04))
    path = _write(rows)
    try:
        s = TrainMetricsPlot._parse(path)
    finally:
        os.unlink(path)

    # The bug that started this: the raw append order is genuinely non-monotonic, so a guard that
    # only checked the parsed output could pass against an input that never exercised the case.
    assert s["resumes"] == [{"died_at_step": 40, "resumed_from_step": 30}], s["resumes"]

    # Every segment handed to the plotter is monotonic -- no backwards line can be drawn.
    for seg_steps, _ in s["loss_segments"] + s["probe_segments"]:
        assert all(b > a for a, b in zip(seg_steps, seg_steps[1:])), seg_steps
    assert len(s["loss_segments"]) == 2, s["loss_segments"]

    # Canonical series: one point per step, ascending, LAST occurrence winning.
    assert s["loss_steps"] == [10, 20, 30, 40, 50], s["loss_steps"]
    assert abs(s["loss"][s["loss_steps"].index(40)] - 0.1) < 1e-9, s["loss"]  # replayed, not 0.6
    assert s["superseded_loss_points"] == 2, s["superseded_loss_points"]

    # The one that would silently corrupt a reported number: the probe at step 40 exists twice and
    # the surviving run scored 4%, not 40%. Reading file order would report the rolled-back 40%.
    assert s["probe_steps"] == [20, 40], s["probe_steps"]
    assert abs(s["probe_accuracy"][-1] - 4.0) < 1e-9, s["probe_accuracy"]
    assert s["superseded_probe_points"] == 1, s["superseded_probe_points"]
    print("PASS  resumed run: segments monotonic, replay supersedes, no double-count")


def check_clean_run():
    """No resume => exactly one segment and nothing marked superseded."""
    rows = [_loss(s, 1.0 / s) for s in range(10, 60, 10)] + [_probe(50, 0.5)]
    path = _write(rows)
    try:
        s = TrainMetricsPlot._parse(path)
    finally:
        os.unlink(path)
    assert s["resumes"] == [], s["resumes"]
    assert len(s["loss_segments"]) == 1, s["loss_segments"]
    assert s["superseded_loss_points"] == 0 and s["superseded_probe_points"] == 0
    assert s["loss_steps"] == [10, 20, 30, 40, 50], s["loss_steps"]
    print("PASS  clean run: single segment, nothing superseded")


def check_probe_died_mid_run():
    """Training outlives its probe -- last_probe_step must expose the gap (the a8_long failure)."""
    rows = []
    for s in range(10, 110, 10):
        rows.append(_loss(s, 1.0 / s))
        if s <= 30:
            rows.append(_probe(s, 0.1))
    path = _write(rows)
    try:
        s = TrainMetricsPlot._parse(path)
    finally:
        os.unlink(path)
    assert s["loss_steps"][-1] == 100, s["loss_steps"]
    assert s["probe_steps"][-1] == 30, s["probe_steps"]
    assert s["resumes"] == [], "a stalled probe is not a resume"
    print("PASS  probe died mid-run: gap between last probe (30) and last loss (100) is visible")


def check_malformed_lines_counted():
    """A truncated final line must be counted, not raised -- partial curves are still useful."""
    path = _write([_loss(10, 1.0), _loss(20, 0.5)])
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"step": 30, "loss": 0.2')  # truncated mid-write
    try:
        s = TrainMetricsPlot._parse(path)
    finally:
        os.unlink(path)
    assert s["malformed_lines"] == 1, s["malformed_lines"]
    assert s["loss_steps"] == [10, 20], s["loss_steps"]
    print("PASS  malformed line counted, not fatal")


if __name__ == "__main__":
    check_clean_run()
    check_resumed_run()
    check_probe_died_mid_run()
    check_malformed_lines_counted()
    print("ALL PASS")
