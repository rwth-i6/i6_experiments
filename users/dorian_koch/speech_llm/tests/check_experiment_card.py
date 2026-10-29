"""Guard: an experiment card (``SpeechFinetune`` / ``RLFinetune`` ``card=``) never reaches the Sisyphus hash.

The card (``speech_llm.experiment_card``) says why a run exists, its one change and its baseline, and is written to
``<job>/output/experiment_card.{json,md}`` by ``write_config``. It is documentation: adding or rewording one must not
re-hash the run (a re-hash orphans a finished arm and retrains it). Both jobs pop ``card`` in ``hash()``.

Checked, for both job classes:
  * **hash-free** -- no card, card A and card B hash identically;
  * **non-vacuous** -- a differing real knob still re-hashes, and ``Job.hash`` WITHOUT the pop does differ on the
    card (so the equality is the pop's doing, not a comparison that cannot fail);
  * **validation** -- an unknown key or a missing ``why``/``change`` is refused at construction;
  * **written** -- ``write_card`` produces the JSON (with the job id) and the Markdown.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_experiment_card.py
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

import sisyphus.job  # noqa: E402
from sisyphus import Job, tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.experiment_card import write_card  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.finetune import MOSHI_LIB_ADAPTER, SpeechFinetune  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.rl_finetune import RLFinetune  # noqa: E402

FAILURES: list[str] = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILURES.append(msg)


VENV = tk.Path("/dev/null/venv/python", hash_overwrite="card_guard_venv")
DATA = tk.Path("/dev/null/corpus", hash_overwrite="card_guard_corpus")
CARD_A = {"why": "guard", "change": "lr x2", "baseline": "ab2_base_0929"}
CARD_B = {"why": "reworded", "change": "lr x2 (vs base)", "asked_by": "the lead, 2026-10-11", "notes": ["a", "b"]}


def sft(card=None, max_steps=100):
    sisyphus.job.created_jobs.clear()  # fresh singleton cache: equal hashes must not just be the same object
    return SpeechFinetune(
        adapter=MOSHI_LIB_ADAPTER, venv_python_path=VENV, train_data=DATA, max_steps=max_steps, card=card
    )


def rl(card=None, max_steps=100):
    sisyphus.job.created_jobs.clear()
    return RLFinetune(venv_python_path=VENV, max_steps=max_steps, card=card)


def check_hash_free(name, build):
    ids = [build(card=c)._sis_id() for c in (None, CARD_A, CARD_B)]
    check(len(set(ids)) == 1, f"{name}: no card / card A / card B hash identically ({ids[0].split('.')[-1]})")
    check(build(max_steps=101)._sis_id() != ids[0], f"{name}: a differing max_steps DOES re-hash")


def check_pop_is_what_holds(name, cls, build):
    """Job.hash on the raw args (the class's pop bypassed) must see the card -> the pop is load-bearing."""
    import inspect

    def raw(card):
        job = build(card=card)
        sig = inspect.signature(cls.__init__)
        args = {k: p.default for k, p in sig.parameters.items() if k != "self"}
        args.update(venv_python_path=VENV, max_steps=100, card=card)
        if cls is SpeechFinetune:
            args.update(adapter=job.adapter.name, train_data=DATA)
        return Job.hash(args)

    check(raw(None) != raw(CARD_A), f"{name}: without the pop the card WOULD move the hash (comparison can fail)")


def check_validation(name, build):
    for bad, why in (({"why": "x"}, "missing change"), ({**CARD_A, "reason": "x"}, "unknown key")):
        try:
            build(card=bad)
        except AssertionError as e:
            check(True, f"{name}: {why} refused ({str(e)[:50]})")
        else:
            check(False, f"{name}: {why} was ACCEPTED")


def check_written():
    job = sft(card=CARD_B)
    with tempfile.TemporaryDirectory() as d:
        j, m = os.path.join(d, "c.json"), os.path.join(d, "c.md")
        write_card(job, j, m)
        payload = json.load(open(j))
        md = open(m).read()
    check(payload["job"] == job._sis_id() and payload["notes"] == ["a", "b"], "card JSON carries the job id + fields")
    check("**change**" in md and "- b" in md, "card Markdown renders headings + list items")
    check(hasattr(job, "out_card") and not hasattr(sft(), "out_card"), "out_card only on carded jobs")


if __name__ == "__main__":
    for name, cls, build in (("SpeechFinetune", SpeechFinetune, sft), ("RLFinetune", RLFinetune, rl)):
        check_hash_free(name, build)
        check_pop_is_what_holds(name, cls, build)
        check_validation(name, build)
    check_written()
    print("FAILED" if FAILURES else "all ok", f"({len(FAILURES)} failures)")
    sys.exit(1 if FAILURES else 0)
