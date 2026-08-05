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
import math
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


#: A real bucket size ratio from the a8-4gpu LoRA checkpoint: the depformer holds 23x the text
#: head's weights, which is the whole reason raw norms cannot be compared across stacks.
NUMEL = {"text_head": 4_620_288, "audio_heads": 3_145_728, "depformer": 107_479_040,
         "temporal": 272_629_760}


def _rich(step, loss, *, rms=None, numel=None, raw=None, rel=None, clipped=False):
    """A modern training row. Fields are opt-in so a check can build any historical shape."""
    row = {"step": step, "loss": loss, "lr": 1e-6, "lr_scale": 0.5,
           "grad_norm": 7.0 if clipped else 0.5, "grad_clip": 1.0, "grad_clipped": clipped}
    if raw is not None:
        row["grad_norm_by_module"] = raw
    if rms is not None:
        row["grad_rms_by_module"] = rms
    if numel is not None:
        row["module_numel"] = numel
    if rel is not None:
        row["weight_delta_rel_by_module"] = rel
    return row


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


def check_legacy_file_still_renders():
    """A run with only loss + probe (a8_fast, a8_long, a6_mix) must lose no existing behaviour."""
    path = _write([_loss(10, 1.0), _probe(10, 0.2), _loss(20, 0.9)])
    try:
        s = TrainMetricsPlot._parse(path)
    finally:
        os.unlink(path)
    assert s["loss_steps"] == [10, 20], s["loss_steps"]
    assert s["probe_steps"] == [10], s["probe_steps"]
    # Every new series must be EMPTY, not absent -- run() indexes them unconditionally.
    for k in ("lr_steps", "grad_norm_steps", "clipped_steps"):
        assert s[k] == [], (k, s[k])
    assert s["grad_modules"] == {} and s["delta_modules"] == {}, s
    assert s["grad_modules_note"] == "run predates per-module metrics", s["grad_modules_note"]
    print("PASS  a legacy loss-only file still parses, with the new series present but empty")


def check_raw_norms_are_never_plotted_unnormalised():
    """THE load-bearing check: raw per-module norms with no numel must yield NO series.

    This is the a8-4gpu shape -- logged after the A2 work but before module_numel existed. Falling
    back to the raw values would put four curves on one axis whose heights differ by sqrt(parameter
    count), which is exactly the comparison that produced a wrong finding on 2026-08-05.
    """
    raw = {"text_head": 0.132, "depformer": 0.074}
    path = _write([_rich(10, 1.0, raw=raw), _rich(20, 0.9, raw=raw)])
    try:
        bare = TrainMetricsPlot._parse(path)
        fixed = TrainMetricsPlot._parse(path, NUMEL)  # same file, numel supplied by the caller
    finally:
        os.unlink(path)
    assert bare["grad_modules"] == {}, (
        f"raw norms were plotted without normalisation: {bare['grad_modules']}"
    )
    assert "module_numel" in bare["grad_modules_note"], bare["grad_modules_note"]

    # With numel supplied the panel appears, and the values are raw / sqrt(numel).
    assert set(fixed["grad_modules"]) == {"text_head", "depformer"}, fixed["grad_modules"]
    for b in ("text_head", "depformer"):
        want = raw[b] / math.sqrt(NUMEL[b])
        assert abs(fixed["grad_modules"][b][1][0] - want) < 1e-15, (b, fixed["grad_modules"][b])
    # And the normalisation REVERSES the ordering here, which is why it matters at all.
    assert raw["text_head"] > raw["depformer"]
    assert fixed["grad_modules"]["text_head"][1][0] > fixed["grad_modules"]["depformer"][1][0]
    print("PASS  raw per-module norms are never plotted unnormalised; numel= makes them plottable")


def check_logged_rms_is_preferred_over_raw():
    """A run that logs both must use the values it computed, not re-derive them."""
    rms = {"text_head": 1.5e-4, "depformer": 2.5e-5}
    raw = {"text_head": 999.0, "depformer": 999.0}  # would give obviously different numbers
    path = _write([_rich(10, 1.0, rms=rms, raw=raw, numel=NUMEL)])
    try:
        s = TrainMetricsPlot._parse(path)
    finally:
        os.unlink(path)
    assert abs(s["grad_modules"]["text_head"][1][0] - 1.5e-4) < 1e-15, s["grad_modules"]
    assert s["module_numel"] == NUMEL, s["module_numel"]
    print("PASS  a logged grad_rms_by_module wins over re-deriving it from the raw norms")


def check_relative_delta_drops_unmeasurable_buckets():
    """LoRA zero-inits B, so ||theta_0||==0 is real: those buckets arrive as None and must not plot."""
    path = _write([
        _rich(10, 1.0, rel={"text_head": 0.01, "depformer": None}),
        _rich(20, 0.9, rel={"text_head": 0.02, "depformer": None}),
    ])
    try:
        s = TrainMetricsPlot._parse(path)
    finally:
        os.unlink(path)
    assert set(s["delta_modules"]) == {"text_head"}, s["delta_modules"]
    assert s["delta_modules"]["text_head"][1] == [0.01, 0.02], s["delta_modules"]
    print("PASS  a None relative delta (zero theta_0) is dropped, not plotted as a break")


def check_new_series_survive_a_resume():
    """The replay collapse must apply to the new series too, not only loss and probe."""
    rows = [
        _rich(10, 1.0, rms={"text_head": 1e-4}),
        _rich(20, 0.9, rms={"text_head": 2e-4}),
        _rich(30, 0.8, rms={"text_head": 3e-4}),
        # preempted; resume from 20 and replay 30 with a different value
        _rich(20, 0.7, rms={"text_head": 9e-4}),
        _rich(30, 0.6, rms={"text_head": 8e-4}),
    ]
    path = _write(rows)
    try:
        s = TrainMetricsPlot._parse(path)
    finally:
        os.unlink(path)
    assert s["lr_steps"] == [10, 20, 30], s["lr_steps"]
    assert s["grad_norm_steps"] == [10, 20, 30], s["grad_norm_steps"]
    steps, vals = s["grad_modules"]["text_head"]
    assert steps == [10, 20, 30], steps
    # Last write wins: the replayed step-30 value, not the rolled-back one.
    assert abs(vals[-1] - 8e-4) < 1e-15, vals
    print("PASS  lr / grad-norm / per-module series all collapse a replay last-write-wins")


def check_clipped_and_probe_seconds_reach_stats():
    path = _write([
        _rich(10, 1.0, clipped=True),
        _rich(20, 0.9, clipped=False),
        {"kind": "knowledge", "step": 20, "knowledge_accuracy": 0.2, "probe_seconds": 301.5},
    ])
    try:
        s = TrainMetricsPlot._parse(path)
    finally:
        os.unlink(path)
    assert s["clipped_steps"] == [10], s["clipped_steps"]
    assert s["grad_clip"] == 1.0, s["grad_clip"]
    assert s["probe_seconds"] == [301.5], s["probe_seconds"]
    print("PASS  clipping and probe_seconds are captured (the number D3 needs)")


if __name__ == "__main__":
    check_clean_run()
    check_resumed_run()
    check_probe_died_mid_run()
    check_malformed_lines_counted()
    check_legacy_file_still_renders()
    check_raw_norms_are_never_plotted_unnormalised()
    check_logged_rms_is_preferred_over_raw()
    check_relative_delta_drops_unmeasurable_buckets()
    check_new_series_survive_a_resume()
    check_clipped_and_probe_seconds_reach_stats()
    print("ALL PASS")
