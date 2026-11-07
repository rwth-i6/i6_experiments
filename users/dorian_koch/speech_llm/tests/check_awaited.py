"""Guard: ``awaited.py`` turns an item READY exactly when its expected results have landed in RESULTS.jsonl.

An awaited item that never turns READY is a result nobody reads; one that turns READY early is a read-out of numbers that
are not all there. Both are checked on a temporary log, plus closing an item and the pattern count (``--expect``).
``add`` against a sink manifest: a pattern no sink produces is refused (a Mimi probe item waited on a kind with no sink),
and the same arm's sinked results no pattern covers are listed (a staleness item left out the arm's MFDB); a refused
bare arm name (patterns match whole tags) is answered with the sinked tags containing it (MT v4's item, 2026-11-07).

    python3 recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_awaited.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import awaited  # noqa: E402


def run(*argv, rc=0) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        assert awaited.main(list(argv)) == rc, argv
    return buf.getvalue()


def check_sinks(d):
    aw, log, man = (os.path.join(d, n) for n in ("A2.jsonl", "R2.jsonl", "S2.json"))
    arm = "ppx_ft_some_long_arm_name"
    with open(man, "w") as f:
        json.dump({"written": "2026-10-30 10:00:00", "tags": [f"benchmark/llm_judge/n1000/{arm}", f"fdb_{arm}"]}, f)
    base = ["--file", aw, "--log", log, "--sinked", man]
    out = run(*base, "add", "--tag", "mimi_aug_probe/*", "--why", "w", rc=2)
    assert "refused" in out, out
    run(*base, "add", "--tag", "mimi_aug_probe/*", "--why", "w", "--unsinked")
    out = run(*base, "add", "--tag", arm, "--why", "w", rc=2)  # the bare arm name: no tag equals it
    assert "refused" in out and f"benchmark/llm_judge/n1000/{arm}" in out and f"--tag '*{arm}*'" in out, out
    assert "matches 2 tag(s)" in out and "restart" not in out, out  # the typo case, not the no-sink causes
    out = run(*base, "add", "--tag", "mimi_aug_probe/*", "--why", "w", rc=2)
    assert "no known tag contains it" in out and "--unsinked" in out and "SINK_KINDS" in out, out
    out = run(*base, "add", "--tag", f"benchmark/llm_judge/n1000/{arm}*", "--why", "w")
    assert f"fdb_{arm}" in out and "not awaited" in out, f"the arm's MFDB was not listed:\n{out}"
    out = run(*base, "add", "--tag", f"benchmark/llm_judge/n1000/{arm}*", "--tag", f"fdb_{arm}", "--why", "w")
    assert "not awaited" not in out, out


def land(log, tag, acc, produced_at=None):
    rec = {"tag": tag, "summary": {"overall": {"accuracy": acc, "n": 1000}}}
    if produced_at:
        rec["produced_at"] = produced_at
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    with tempfile.TemporaryDirectory() as d:
        aw, log = os.path.join(d, "AWAITED.jsonl"), os.path.join(d, "RESULTS.jsonl")
        # no manifest here (the default RESULTS_SINKED.json of a setup root would refuse these made-up tags)
        base = ["--file", aw, "--log", log, "--sinked", os.path.join(d, "no_manifest.json")]
        run(*base, "add", "--tag", "benchmark/x/n1000/*", "--expect", "2", "--why", "decides X", "--next", "read it")
        run(*base, "add", "--tag", "benchmark/y/n1000/arm", "--why", "decides Y")
        out = run(*base)
        assert "waiting 0/2" in out and "waiting 0/1" in out and "0 of 2 ready" in out, out
        land(log, "benchmark/x/n1000/a", 0.2)
        land(log, "benchmark/other/n1000/a", 0.3)  # matches no pattern
        out = run(*base)
        assert "waiting 1/2" in out and "READY" not in out, f"one of two landed must not be READY:\n{out}"
        land(log, "benchmark/x/n1000/b", 0.25)
        out = run(*base)
        assert out.count("READY") == 1 and "acc 25.0%" in out and "decides X" in out, out
        run(*base, "done", "1", "--note", "read out")
        out = run(*base)
        assert "[1]" not in out and "[2] waiting 0/1" in out, out
        assert "closed" in run(*base, "list", "--all")
        # a result produced before the item was marked (a re-run under the same tag) does not make it READY
        land(log, "benchmark/z/n1000/arm", 0.4, produced_at="2026-10-01 12:00:00")
        run(*base, "add", "--tag", "benchmark/z/n1000/arm", "--why", "decides Z")
        out = run(*base)
        assert "[3] waiting 0/1" in out, f"an old result made a fresh item READY:\n{out}"
        land(log, "benchmark/z/n1000/arm", 0.5, produced_at="2099-01-01 00:00:00")
        assert "[3] READY" in run(*base), "the new result does not make it READY"
        # a torn line does not break every call; two adds with one id are both listed
        with open(aw, "a") as f:
            f.write(
                '{"event": "add", "id": 3, "tags": ["t/dup"], "expect": 1, "why": "dup", "at": "2026-10-29", "at_unix": 0}\n'
            )
            f.write('{"event": "add", "id": 9, "ta\n')
        out = run(*base)
        assert "decides Z" in out and "dup" in out, f"a duplicate id replaced an item:\n{out}"
        check_sinks(d)
    print("check_awaited: ALL PASS")


if __name__ == "__main__":
    main()
