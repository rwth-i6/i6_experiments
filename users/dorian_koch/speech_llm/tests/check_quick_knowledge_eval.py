"""Guard the fast in-training knowledge probe (quick_knowledge_eval): the alias-match scorer and its
summary schema. Login-node, no GPU/model/manager.

Why this exists: the whole point of the quick eval is to catch factual-recall collapse a wrong scorer
would hide -- if alias_match silently over- or under-credits, the probe lies exactly when you rely on
it. So pin its semantics (containment, aliases, normalization, empty) and pin that its summary is a
drop-in for LLMGrading's schema. A real-data smoke check confirms it tracks the LLM judge in the
right ballpark (proxy, not equality).

Run:  CUDA_HOME=/usr PYTHONPATH=recipe:recipe/sisyphus .venv/bin/python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_quick_knowledge_eval.py
"""

import json
import os
import sys

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")

from i6_experiments.users.dorian_koch.speech_llm.quick_knowledge_eval import (  # noqa: E402
    alias_match,
    summarize,
)
from speech_llm.full_duplex.text_scoring import normalize_answer  # noqa: E402

# --- 1. scorer semantics ------------------------------------------------------------------------
# Verbose reply that CONTAINS the answer -> correct (the common case; exact-match would miss it).
b, q = alias_match("It's New York City. In 2020 it had a population of ...", "New York", ["New York City"])
assert b == 1, "containment of an alias in a verbose reply must count as correct"

# Article/punctuation/case are normalized away.
b, _ = alias_match("the ANSWER is Molybdenum!", "molybdenum", [])
assert b == 1, "normalization (case/punct) must not break a real match"
b, _ = alias_match("that is the black sea, yeah", "The Black Sea", [])
assert b == 1, "leading article on the reference must be normalized"

# A gold answer ending in a letter ("Vitamin A", "Plan A") must not shrink to its prefix.
assert alias_match("I think it's vitamin C.", "Vitamin A", [])[0] == 0, "Vitamin A must not match vitamin C"
assert alias_match("Plan B was used", "Plan A", [])[0] == 0, "Plan A must not match Plan B"
assert alias_match("it is vitamin a, found in carrots", "Vitamin A", [])[0] == 1, "Vitamin A must still match itself"
assert alias_match("A Tale of Two Cities", "Tale of Two Cities", [])[0] == 1

# A wrong (confabulated) answer -> incorrect, even though it's fluent.
b, q = alias_match("That was William Henry Harrison.", "Benjamin Britten", ["Britten"])
assert b == 0 and q == 1, "a fluent-but-wrong reply must score 0 / quality 1"

# Empty / whitespace prediction -> incorrect.
assert alias_match("", "Poland", ["Republic of Poland"]) == (0, 1)
assert alias_match("   ", "Poland", []) == (0, 1)

# aliases passed as a JSON string (as they can arrive from the jsonl) are handled.
b, _ = alias_match("I think it's gazpacho, served cold", "GAZPACHO", json.dumps(["gazpacho soup"]))
assert b == 1, "aliases given as a JSON string must be parsed"

# A short, essentially-just-the-answer reply gets the top quality bucket; a long one gets the middle.
_, q_short = alias_match("Conclave.", "Conclave", [])
_, q_long = alias_match("The name of the assembly of cardinals that elects a pope is the Conclave.", "Conclave", [])
assert q_short == 5 and q_long == 3, (q_short, q_long)

# normalize_answer keeps word boundaries so a substring can't match across words spuriously.
assert normalize_answer("The Cat") == " cat "
assert " york " in normalize_answer("New York City")

print("[ok] scorer semantics")

# --- 2. summary schema is a drop-in for LLMGrading ----------------------------------------------
rows = [
    {"binary_correct": 1, "quality_score": 5, "category": "unknown"},
    {"binary_correct": 0, "quality_score": 1, "category": "unknown"},
    {"binary_correct": 1, "quality_score": 3, "category": "science"},
]
s = summarize(rows)
assert set(s["overall"]) == {"n", "accuracy", "avg_quality"}, s["overall"]
assert s["overall"]["n"] == 3
assert abs(s["overall"]["accuracy"] - 2 / 3) < 1e-9
assert "science" in s and s["science"]["n"] == 1
assert summarize([])["overall"] == {"n": 0, "accuracy": 0.0, "avg_quality": 0.0}
print("[ok] summary schema matches LLMGrading (n/accuracy/avg_quality, per-category + overall)")

# --- 3. real-data smoke: tracks the LLM judge in the right ballpark ------------------------------
# On the finished A7-mix monologue transcriptions the LLM judge scored 9.6%. The alias-match proxy
# should land in a plausible collapsed band and clearly below a base-like ~18% -- i.e. it would have
# flagged the collapse. Skipped gracefully if the output isn't present (fresh checkout).
p = os.path.realpath("output/benchmark/moshi_ft_a7_mix_monologue/transcription")
if os.path.exists(p):
    data = [json.loads(l) for l in open(p) if l.strip()]
    res = [
        {
            "binary_correct": alias_match(r.get("transcription", ""), r["answer"], r.get("aliases", []))[0],
            "quality_score": 1,
            "category": r.get("category", "unknown"),
        }
        for r in data
    ]
    acc = summarize(res)["overall"]["accuracy"]
    assert 0.03 <= acc <= 0.20, f"proxy acc {acc:.1%} outside plausible band vs LLM-judge 9.6%"
    print(f"[ok] real-data smoke: alias-match acc={acc:.1%} (LLM-judge 9.6%) on {len(data)} rows")
else:
    print("[skip] real-data smoke (no a7_mix transcriptions on disk)")

# --- 4. one scorer: the in-training-loop probe grades with the same functions -------------------
# moshi_family.knowledge_probe runs in the job venv (no sisyphus / i6_experiments); both it and this
# module import text_scoring, so the in-loop probe and the quick eval cannot report different rules.
sys.path.insert(0, "recipe/speech_llm/full_duplex")
import text_scoring  # noqa: E402
from moshi_family import knowledge_probe as kp  # noqa: E402

# two import names (job venv: text_scoring; manager: speech_llm.full_duplex.text_scoring) = one file
_src = {os.path.realpath(f.__code__.co_filename) for f in (kp.alias_match, text_scoring.alias_match, alias_match,
                                                           kp.normalize_answer, normalize_answer)}  # fmt: skip
assert _src == {os.path.realpath(text_scoring.__file__)}, _src
print("[ok] knowledge_probe and quick_knowledge_eval grade with text_scoring's alias_match")

print("\nALL CHECKS PASSED")
