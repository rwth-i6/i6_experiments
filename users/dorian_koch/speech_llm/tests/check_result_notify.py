"""Guard: a training arm's ResultNotify digest must carry NUMBERS, not a directory path.

`notify_result` is the "never miss a result" sink: it fires exactly when an experiment finishes and
appends one line to RESULTS.jsonl. Until 2026-09-07 it was attached to two things -- the
per-checkpoint quick evals and the VoiceBench sweep -- so no *training* arm ever appended a line, and
attaching one would not have helped: a run's only declared output is its `run_dir`, and the digest's
directory branch recorded `{"is_dir": true}`. A 600-step arm would have "fired" and told you nothing.

Two things are asserted here, and the second is the one that rots quietly:

  1. Given a run_dir, the digest summarises every `metrics*.jsonl` inside it.
  2. The summary is taken in STEP order, not file order. `metrics.train.jsonl` is append-ordered, so
     a preempted run replays steps on resume and the last LINE is not the last STEP (the same
     property check_train_metrics_plot exists for). The fixture below replays, and the check asserts
     the step-ordered answer differs from the naive last-line one -- so a regression to reading the
     tail cannot pass.

Since 2026-10-28 also: the line carries ``produced_at`` (the result's own mtime, so a sink attached after the fact
sorts by when the result was made); ``read_results.py`` renders every record of the live ``RESULTS.jsonl`` (an
unknown shape is shown flagged, never as an empty cell); and with ``--graph`` (one config load, ~2 min) every result
output of a ``SINK_KINDS`` class in the real graph has a sink -- ``attach_result_sinks`` exists because, by
2026-10, only MFDB and MRQA still fed the log.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_result_notify.py [--graph]
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))

from i6_experiments.users.dorian_koch.speech_llm import read_results  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.result_notify import (  # noqa: E402
    ResultNotify,
    summarize_metrics_jsonl,
)

#: A resumed run: steps 1,2,3 then a preemption rolls back and replays 2,3,4. The last LINE is
#: step 4 here, but line 3 (step 3, loss 9.99) is the trap -- a naive tail read of a file whose
#: replay ended early would report it.
TRAIN_ROWS = [
    {"step": 1, "loss": 2.0, "lr": 1e-6},
    {"step": 2, "loss": 1.8, "lr": 2e-6},
    {"step": 3, "loss": 9.99, "lr": 3e-6},  # pre-preemption value, superseded by the replay
    {"step": 2, "loss": 1.7, "lr": 2e-6},
    {"step": 3, "loss": 1.5, "lr": 3e-6},
    {"step": 1, "loss": 5.0, "lr": 1e-6},  # out-of-order straggler: must NOT become "first"... it is
]
EVAL_ROWS = [
    {"step": 25, "eval_loss": 1.42},
    {"step": 600, "eval_loss": 1.10},
]


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def check_summary_is_step_ordered():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "metrics.train.jsonl")
        _write(p, TRAIN_ROWS)
        s = summarize_metrics_jsonl(p)
        assert s["rows"] == len(TRAIN_ROWS), s
        assert s["last_step"] == 3, s
        # step-ordered: highest step is 3, and the LATER row for step 3 (the replay) must win.
        assert s["loss"]["last"] == 1.5, (
            f"loss.last={s['loss']['last']} -- expected the replayed step-3 value 1.5. Reading the "
            f"file's tail would give {TRAIN_ROWS[-1]['loss']}, and the pre-preemption step 3 was "
            f"{TRAIN_ROWS[2]['loss']}; both are wrong."
        )
        assert s["loss"]["min"] == 1.5 and s["loss"]["max"] == 9.99, s
        # Non-vacuity: the naive tail read must actually disagree, or this proves nothing.
        assert TRAIN_ROWS[-1]["loss"] != s["loss"]["last"], "fixture no longer distinguishes the bug"
    print("[ok] metrics summary is taken in step order, not file order")


def check_run_dir_is_digested():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "work", "SomeJob", "output", "run_dir")
        os.makedirs(run_dir)
        _write(os.path.join(run_dir, "metrics.train.jsonl"), TRAIN_ROWS)
        _write(os.path.join(run_dir, "metrics.eval.jsonl"), EVAL_ROWS)
        os.utime(run_dir, (1700000000, 1700000000))  # the run finished long before its sink fired
        digest_path = os.path.join(tmp, "work", "SomeJob", "output", "result.json")

        # Job.__new__ intercepts construction (sisyphus), so drive the real run() unbound.
        job = SimpleNamespace(
            tag="train_a17_diag",
            results={"run_dir": SimpleNamespace(get=lambda: run_dir)},
            note="A17 a17_diag",
            out_digest=SimpleNamespace(get=lambda: digest_path),
        )
        ResultNotify.run(job)

        digest = json.load(open(digest_path))
        entry = digest["results"]["run_dir"]
        assert "metrics" in entry, (
            f"the digest for a run_dir carries {sorted(entry)} -- no 'metrics'. A training arm's only "
            f"declared output IS its run_dir, so without this the sink fires and reports a path."
        )
        assert set(entry["metrics"]) == {"metrics.train.jsonl", "metrics.eval.jsonl"}, entry["metrics"]
        ev = entry["metrics"]["metrics.eval.jsonl"]["eval_loss"]
        assert (ev["first"], ev["last"]) == (1.42, 1.10), ev
        # And the central log really got a line with those numbers in it.
        line = json.loads(open(os.path.join(tmp, "RESULTS.jsonl")).read().strip())
        assert line["tag"] == "train_a17_diag", line
        assert "1.1" in json.dumps(line["summary"]), (
            "RESULTS.jsonl line does not carry the eval loss -- scanning it at session start would "
            "show that the arm finished and not how it went."
        )
        want = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(1700000000))
        assert line["produced_at"] == want and line["at"] != want, (line["at"], line["produced_at"], want)
        text, ok = read_results.render(line)
        assert ok and "step 3" in text and "loss 1.5" in text, text
    print("[ok] a run_dir digest carries both metrics files; RESULTS.jsonl gets the numbers and the result's own time")


def check_reader_renders_the_live_log():
    log = SETUP / "RESULTS.jsonl"
    recs = read_results.load(str(log))
    bad = [t for t, r in recs.items() if not read_results.render(r)[1]]
    assert not bad, f"{len(bad)} of {len(recs)} records have no reader: {bad[:5]}"
    text, ok = read_results.render({"tag": "x", "summary": {"summary": {"some_new_metric": 1}}})
    assert not ok and text.startswith("?"), f"an unknown shape must be flagged, got {text!r}"
    print(f"[ok] read_results renders all {len(recs)} records of RESULTS.jsonl; an unknown shape is flagged")


def check_graph_coverage():
    import logging

    os.chdir(SETUP)
    os.environ.setdefault("CUDA_HOME", "/usr")
    logging.disable(logging.WARNING)
    from sisyphus.loader import config_manager

    from i6_experiments.users.dorian_koch.speech_llm.result_notify import sinked_paths, unsinked_results

    config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
    left = unsinked_results()
    assert not left, f"{len(left)} result outputs without a sink: {list(left)[:5]}"
    sinked = sinked_paths()
    one = next(p for p in sorted(sinked) if p.endswith("summary.json"))
    gap = unsinked_results(sinked=sinked - {one})
    assert [p.get_path() for _, p in gap.values()] == [one], f"a dropped sink was not caught: {one} -> {list(gap)}"
    print(f"[ok] every result output in the graph has a sink ({len(sinked)} sinked paths); a dropped one is caught")


def check_empty_and_malformed_are_survivable():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "metrics.train.jsonl")
        open(p, "w").write('{"step": 1, "loss": 1.0}\n\n{"step": 2, "loss":\n')  # torn final line
        s = summarize_metrics_jsonl(p)
        assert s["rows"] == 1 and s["loss"]["last"] == 1.0, s
        empty = os.path.join(tmp, "empty.jsonl")
        open(empty, "w").close()
        assert summarize_metrics_jsonl(empty) == {"rows": 0}
    print("[ok] a live (half-written) or empty metrics file does not break the digest")


if __name__ == "__main__":
    check_summary_is_step_ordered()
    check_run_dir_is_digested()
    check_empty_and_malformed_are_survivable()
    check_reader_renders_the_live_log()
    if "--graph" in sys.argv:
        check_graph_coverage()
    print("ALL PASS")
