"""A benchmark number must carry WHICH GRADER produced it, and the ledger must never rank across them.

THE CONFUSION THIS EXISTS FOR (2026-09-18, user directive after it happened twice in one session):
two graders write the same summary schema on purpose -- ``AliasMatchGrading``'s docstring says it
emits "the EXACT schema LLMGrading produces, so it is a drop-in for the tail of the pipeline". That
interchangeability is useful and it is also why a number, once it left the job, carried no record of
how it was scored. The grader was recoverable only from a ``quick_`` tag prefix, and the two are not
comparable: ~6 pts apart on base, ~0 on terse arms, so not even a correctable offset.

What went wrong concretely: an n=1000 ALIAS arm (14.8%) was compared against an n=64 ALIAS base
(20.3%), the intervals overlapped, and the conclusion drawn was that no gap could be established --
while the correct judged pair, ``moshi_ft_a41_alldata`` 12.1% vs ``moshi_family`` 17.7%, both
n=1000, both judged, sat in the ledger already and shows a clean 5.6 pt gap (z~3.5). A second
contributor: the two graders register into DIFFERENT trees (``benchmark/<tag>`` vs
``benchmark/quick/<tag>``) and ``read_benchmarks.py`` globbed only the first, so one grader was
invisible to the ledger tool and only visible via ``RESULTS.jsonl``.

So: the identity goes in the data, history is labelled from the producing job class (no re-run), and
the reader groups by grader instead of printing one sorted table.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

failures = []


def check(cond, msg):
    print(f"{'[ok]' if cond else '[FAIL]'} {msg}")
    if not cond:
        failures.append(msg)


# --- 1. Both graders stamp themselves, via ONE shared helper so they cannot drift.
from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import (
    GRADER_SCHEMA_VERSION,
    _grader_block,
)
from i6_experiments.users.dorian_koch.speech_llm.quick_knowledge_eval import summarize

rows = [
    {"category": "unknown", "binary_correct": 1, "quality_score": 5},
    {"category": "unknown", "binary_correct": 0, "quality_score": 1},
]
s = summarize(rows)
check(s.get("grader", {}).get("name") == "alias_match", "AliasMatchGrading's summarize() stamps alias_match")
check(s["grader"].get("schema_version") == GRADER_SCHEMA_VERSION, "the stamp carries a schema version")
check(s["overall"]["n"] == 2 and s["overall"]["accuracy"] == 0.5, "stamping did not disturb the numbers")

judged = _grader_block("llm_judge", model="google/gemma-4-31B-it")
check(judged["name"] == "llm_judge" and judged.get("model"), "the judge stamp records its MODEL too")
try:
    _grader_block("vibes")
    check(False, "an unknown grader name must be refused")
except AssertionError:
    check(True, "an unknown grader name is refused rather than silently recorded")

# --- 2. History is identified from the producing job class, so no re-run was needed to label it.
sys.path.insert(0, ".")
import read_benchmarks as rb

with tempfile.TemporaryDirectory() as td:
    for job, expect in (("LLMGrading", "llm_judge"), ("AliasMatchGrading", "alias_match")):
        d = os.path.join(td, f"work/{job}.HASH1234/output")
        os.makedirs(d)
        real = os.path.join(d, "summary.json")
        with open(real, "w") as f:
            json.dump({"overall": {"n": 10, "accuracy": 0.1, "avg_quality": 1.1}}, f)
        link = os.path.join(td, f"link_{job}")
        os.symlink(real, link)
        with open(link) as f:
            got = rb.grader_of(link, json.load(f))
        check(got == expect, f"a legacy (unstamped) {job} summary is identified as {expect}")

    # Non-vacuous: something we cannot attribute must say so, not guess.
    orphan = os.path.join(td, "orphan.json")
    with open(orphan, "w") as f:
        json.dump({"overall": {"n": 10, "accuracy": 0.1}}, f)
    check(rb.grader_of(orphan, {"overall": {}}) == "unknown", "an unattributable summary reports 'unknown', not a guess")

    # A stamped summary beats any path inference -- the data is the authority.
    mis = os.path.join(td, "work/LLMGrading.XX/output")
    os.makedirs(mis)
    p = os.path.join(mis, "summary.json")
    with open(p, "w") as f:
        json.dump({"overall": {}, "grader": {"name": "alias_match"}}, f)
    with open(p) as f:
        check(rb.grader_of(p, json.load(f)) == "alias_match", "a stamped grader overrides the path inference")

# --- 3. The reader cannot print one cross-grader ranking.
src = open("read_benchmarks.py").read()
check("by_grader" in src and "for grader in (" in src, "the ledger groups rows by grader before printing")
check('{ROOT}/*/n*/*/summary' in src, "the ledger globs the <grader>/n<N>/<tag> layout, so neither grader can be omitted")
check("grader mismatch" in src and "n mismatch" in src, "the ledger refuses a row whose PATH and DATA disagree")
check("ci95" in src, "the ledger prints a 95% interval, so an n=64 score cannot read like an n=1000 one")
check("no_data" in src and "SCORED ZERO ROWS" in src,
      "a summary that scored 0 rows is kept OUT of the ranking (a 0.0% row reads as real knowledge loss)")

# The single-table form is the actual defect; assert it is gone rather than merely that grouping exists.
combined = "scored = sorted([r for r in rows if r[0] is not None], key=lambda r: -r[0])"
check(combined not in src, "the old single sorted table across all graders is gone")

# --- 4. The live tree: every registered summary sits under a path that matches its own data.
#     This is what makes the n in the path trustworthy. Measured 2026-09-18 across all 78 tags,
#     transcription == eval_results == summary.overall.n, i.e. rows never drop between sampling and
#     scoring -- so a mismatch here means something genuinely changed, not ordinary attrition.
import glob as _glob

checked = mismatched = 0
for sp in _glob.glob("output/benchmark/*/n*/*/summary"):
    parts = sp.split(os.sep)
    grader_dir, n_dir = parts[-4], parts[-3]
    try:
        with open(sp) as fh:
            data = json.load(fh)
    except Exception:
        continue
    n_actual = (data.get("overall") or {}).get("n")
    checked += 1
    if not n_actual:
        continue  # scored zero rows: no number to mislabel; the reader must omit it (asserted below)
    if n_dir != "nfull" and n_dir != f"n{n_actual}":
        mismatched += 1
        print(f"      {sp}: path says {n_dir}, summary says n={n_actual}")
    g = data.get("grader", {}).get("name")
    if g and g != grader_dir:
        mismatched += 1
        print(f"      {sp}: path says {grader_dir}, data says {g}")
if checked:
    check(mismatched == 0, f"all {checked} registered summaries sit under a path matching their own grader and n")
else:
    print("[skip] no registered summaries found (run from the setup root after a manager pass)")

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print("all grader-provenance checks passed")
