"""Guard: ``awaited.py`` turns an item READY exactly when its expected results have landed in RESULTS.jsonl.

An awaited item that never turns READY is a result nobody reads; one that turns READY early is a read-out of numbers that
are not all there. Both are checked on a temporary log, plus closing an item and the pattern count (``--expect``).

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


def run(*argv) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert awaited.main(list(argv)) == 0, argv
    return buf.getvalue()


def land(log, tag, acc):
    with open(log, "a") as f:
        f.write(json.dumps({"tag": tag, "summary": {"overall": {"accuracy": acc, "n": 1000}}}) + "\n")


def main():
    with tempfile.TemporaryDirectory() as d:
        aw, log = os.path.join(d, "AWAITED.jsonl"), os.path.join(d, "RESULTS.jsonl")
        base = ["--file", aw, "--log", log]
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
    print("check_awaited: ALL PASS")


if __name__ == "__main__":
    main()
