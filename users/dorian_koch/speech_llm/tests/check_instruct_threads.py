"""Guard: oasst2 -> multi-turn instruct threads (``instruct_prompts.oasst_threads`` / ``flatten_threads``; 2026-10-11).

Can fail: following a lower-ranked reply or follow-up instead of the rank-0 path; keeping deleted / review-failed /
non-English messages; a thread longer than ``max_turns`` or shorter than ``min_turns`` (or ending on a user turn with
no reply counted as a turn); flattened rows of a thread out of order or interleaved with another thread.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_instruct_threads.py
"""

import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm.instruct_prompts import flatten_threads, oasst_threads  # noqa: E402


def m(mid, parent, role, text, rank=None, lang="en", deleted=False, review=True):
    return {"message_id": mid, "parent_id": parent, "role": role, "text": text, "rank": rank, "lang": lang,
            "deleted": deleted, "review_result": review}  # fmt: skip


ROWS = [
    # tree A: 4 user turns on the rank-0 path; a rank-1 reply and a rank-1 follow-up must be ignored
    m("a0", None, "prompter", "A u0"),
    m("a1", "a0", "assistant", "A r0 best", rank=0),
    m("a1x", "a0", "assistant", "A r0 worse", rank=1),
    m("a2x", "a1", "prompter", "A u1 worse", rank=1),
    m("a2", "a1", "prompter", "A u1", rank=0),
    m("a3", "a2", "assistant", "A r1"),
    m("a4", "a3", "prompter", "A u2"),
    m("a5", "a4", "assistant", "A r2"),
    m("a6", "a5", "prompter", "A u3"),
    m("a7", "a6", "assistant", "A r3"),
    # tree B: the follow-up has no reply -> 1 complete turn -> dropped (min 2)
    m("b0", None, "prompter", "B u0"),
    m("b1", "b0", "assistant", "B r0"),
    m("b2", "b1", "prompter", "B u1 unanswered"),
    # tree C: German -> dropped
    m("c0", None, "prompter", "C u0", lang="de"),
    m("c1", "c0", "assistant", "C r0"),
    m("c2", "c1", "prompter", "C u1"),
    m("c3", "c2", "assistant", "C r1"),
    # tree D: the best reply is deleted -> the next one is used; 2 turns kept
    m("d0", None, "prompter", "D u0"),
    m("d1", "d0", "assistant", "D r0 deleted", rank=0, deleted=True),
    m("d1b", "d0", "assistant", "D r0", rank=1),
    m("d2", "d1b", "prompter", "D u1"),
    m("d3", "d2", "assistant", "D r1", review=False),  # review failed -> skipped, the other reply is used
    m("d3b", "d2", "assistant", "D r1 ok", rank=1),
]


def main():
    th = {t["id"]: t["turns"] for t in oasst_threads(ROWS, min_turns=2, max_turns=3)}
    assert set(th) == {"a0", "d0"}, sorted(th)
    assert [x["question"] for x in th["a0"]] == ["A u0", "A u1", "A u2"], th["a0"]  # rank-0 path, cut at 3
    assert [x["answer"] for x in th["a0"]] == ["A r0 best", "A r1", "A r2"], th["a0"]
    assert [x["answer"] for x in th["d0"]] == ["D r0", "D r1 ok"], th["d0"]
    assert set(th) == {t["id"] for t in oasst_threads(ROWS, min_turns=2, max_turns=4)}
    assert len({t["id"]: t for t in oasst_threads(ROWS, min_turns=2, max_turns=4)}["a0"]["turns"]) == 4
    flat = flatten_threads(oasst_threads(ROWS), "oasst2_mt")
    assert [(r["thread_id"], r["turn"], r["n_turns"]) for r in flat] == [
        ("a0", 0, 3), ("a0", 1, 3), ("a0", 2, 3), ("d0", 0, 2), ("d0", 1, 2)
    ], flat  # fmt: skip
    assert all(r["category"] == "oasst_mt" and r["aliases"] == "[]" for r in flat)
    print("check_instruct_threads: ok")


if __name__ == "__main__":
    main()
