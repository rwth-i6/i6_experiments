"""Instruction-prompt sources (``speech_llm.instruct_prompts``): oasst2 tree extraction and the rule filter.

    cd <setup> && CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_instruct_prompts.py

Can fail: a non-English, deleted or review-failed message used as a prompt or reference; a reply that is not the
rank-0 one; a reply attached to a non-first turn; a code request or an over-long request passed to the LLM pass.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, "recipe/sisyphus")

from i6_experiments.users.dorian_koch.speech_llm.instruct_prompts import oasst_first_turns, rule_drop  # noqa: E402


def _m(mid, parent, role, text, lang="en", rank=None, deleted=False, review=True):
    return {"message_id": mid, "parent_id": parent, "role": role, "text": text, "lang": lang, "rank": rank,
            "deleted": deleted, "review_result": review}  # fmt: skip


def main():
    rows = [
        _m("p1", None, "prompter", "How do volcanoes form?"),
        _m("a1", "p1", "assistant", "second best", rank=1),
        _m("a2", "p1", "assistant", "best reply", rank=0),
        _m("u2", "a2", "prompter", "follow-up turn"),
        _m("a3", "u2", "assistant", "reply to a follow-up", rank=0),
        _m("p2", None, "prompter", "Wie spät ist es?", lang="de"),
        _m("a4", "p2", "assistant", "German reply", lang="de", rank=0),
        _m("p3", None, "prompter", "Deleted prompt", deleted=True),
        _m("a5", "p3", "assistant", "orphan", rank=0),
        _m("p4", None, "prompter", "Unranked single reply?"),
        _m("a6", "p4", "assistant", "only reply"),
        _m("p5", None, "prompter", "Review failed?"),
        _m("a7", "p5", "assistant", "bad", rank=0, review=False),
    ]
    got = {t["id"]: t["answer"] for t in oasst_first_turns(rows)}
    assert got == {"p1": "best reply", "p4": "only reply"}, got
    assert rule_drop("Write a python function that sorts a list", "Here is a sorted function for you.") == "code_or_format"
    assert rule_drop("Hi", "hello there friend") == "question_length"
    assert rule_drop(" ".join(["word"] * 61), "a fine reply here") == "question_length"
    assert rule_drop("What causes the seasons on Earth?", "Axial tilt.") == "reference_short"
    assert rule_drop("What causes the seasons on Earth?", "The tilt of Earth's axis.") is None
    print("check_instruct_prompts OK")


if __name__ == "__main__":
    main()
