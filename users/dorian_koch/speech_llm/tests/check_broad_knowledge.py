"""Guard: the broad-spectrum knowledge-RL knobs (2026-10-11, knowledge_broad_rl.md) are hash-neutral at their defaults
and do what they say; the prompt-prep filters catch what they claim.

Can fail: a new ``RLFinetune`` knob re-hashes the existing arms (default not excluded); a broad pool does NOT re-hash
(two different prompt pools sharing one identity); ``knowledge_judge_version`` rendered into configs that never set it
(changes every later render) or missing when set (the launcher would judge with v1); ARC questions that need their
options, or time-dependent NQ questions, passing the rule filter; the decontamination missing a reordered / re-cased
duplicate, or flagging an unrelated question (a filter that drops everything also "decontaminates").

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_broad_knowledge.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import sisyphus.job  # noqa: E402
from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import broad_questions as bq  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.rl_finetune import RLFinetune, _render_knowledge_keys  # noqa: E402

FAILURES: list[str] = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILURES.append(msg)


def P(name):
    return tk.Path(f"/dev/null/{name}", hash_overwrite=f"broad_guard_{name}")


SOURCES = ({"category": "search", "questions": P("nq_q"), "audio": P("nq_a")},)


def rl(**kw):
    sisyphus.job.created_jobs.clear()
    base = dict(
        venv_python_path=P("venv"),
        family="personaplex",
        segment_source="knowledge",
        knowledge_questions=P("tq"),
        knowledge_audio=P("ta"),
        quality_judge="judge",
        max_steps=100,
    )
    return RLFinetune(**{**base, **kw})


NEW_KNOBS = ("knowledge_sources", "knowledge_base_category", "knowledge_balance", "n_probe_per_category",
             "knowledge_judge_version")  # fmt: skip


def check_hashes():
    a = rl()
    # an arm declared BEFORE the knobs existed: its parsed args lack them (comparing two new constructions could not
    # catch an unexcluded knob -- both would carry it)
    legacy = {k: v for k, v in a._sis_kwargs.items() if k not in NEW_KNOBS}
    check(RLFinetune._sis_hash_static(legacy) == RLFinetune._sis_hash_static(a._sis_kwargs),
          "the new knobs at their defaults hash like a pre-knob arm")  # fmt: skip
    b = rl(knowledge_balance=False, n_probe_per_category=0, knowledge_judge_version="v1", knowledge_sources=None)
    check(a._sis_id() == b._sis_id(), "explicit defaults hash like an existing arm")
    broad = rl(
        knowledge_sources=SOURCES, knowledge_base_category="trivia", knowledge_balance=True, n_probe_per_category=16
    )
    check(broad._sis_id() != a._sis_id(), "a broad pool re-hashes")
    v2 = rl(knowledge_judge_version="v2_broad")
    check(v2._sis_id() != a._sis_id(), "the v2_broad judge re-hashes")
    check("knowledge_judge_version" not in _render_knowledge_keys(a), "v1 renders no judge-version key")
    check('knowledge_judge_version: "v2_broad"' in _render_knowledge_keys(v2), "v2_broad is rendered")
    try:
        rl(knowledge_balance=True)
    except AssertionError:
        check(True, "balance without broad sources is refused")
    else:
        check(False, "balance without broad sources was ACCEPTED")


def check_rules():
    drop = bq.rule_drop
    check(drop("arc", "Which of the following is a renewable resource?", "wind") == "needs_options", "ARC 'which of'")
    check(drop("arc", "Which statement best explains why ice floats?", "x") == "needs_options", "ARC 'best'")
    check(
        drop("arc", "What gas do plants take in from the air to make food?", "carbon dioxide") is None, "ARC open kept"
    )
    check(
        drop("arc", "When a switch is used in an electrical circuit, the switch can", "x") == "not_a_question",
        "ARC stem",
    )
    check(drop("nq_open", "who is the current president of france", "x") == "time_dependent", "NQ 'current'")
    check(drop("nq_open", "who wrote the song hotel california", "Don Felder") is None, "NQ stable kept")
    check(drop("nq_open", "who wrote it", "") == "no_answer", "NQ without an answer")


def check_near_duplicates():
    corpus = ["Who wrote the novel Moby Dick?", "What is the boiling point of water at sea level?"]
    sets, index = [], defaultdict(set)
    for q in corpus:
        w = bq.content_words(q)
        sets.append(w)
        for x in w:
            index[x].add(len(sets) - 1)
    hit = bq.is_near_duplicate(bq.content_words("moby dick: who WROTE the novel"), index, sets)
    check(hit == 0, f"reordered / re-cased duplicate found ({hit})")
    miss = bq.is_near_duplicate(bq.content_words("Who painted the Mona Lisa?"), index, sets)
    check(miss is None, f"an unrelated question is not flagged ({miss})")
    check(bq.norm("Who wrote  Moby-Dick?") == bq.norm("who wrote moby dick"), "normalised exact match")


if __name__ == "__main__":
    check_hashes()
    check_rules()
    check_near_duplicates()
    print("FAILED" if FAILURES else "all ok", f"({len(FAILURES)} failures)")
    sys.exit(1 if FAILURES else 0)
