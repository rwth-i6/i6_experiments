"""Guard: an ARGUMENT NAME must never reach the hashed ``hparams`` bag.

THE BUG THIS EXISTS FOR (2026-09-18, found 2026-09-19, cost one 600-step arm plus its whole n=1000
benchmark chain, and produced a wrong headline).

``a44_heldout`` was B7 -- "the first GENUINELY held-out number this project will have". It was
declared as::

    _arm("a44_heldout", ..., eval_data=ann_triviaqa_eval,
         extra_hparams={"eval_batches": 32, "eval_freq": 25})

``_arm`` forwards ``**kw`` to ``_anchored``, whose signature names eleven knobs and collects the
rest in ``**extra`` -- and which then passed ``extra_hparams=extra``, i.e. **everything it did not
recognise became an hyper-parameter**. ``eval_data`` is a SpeechFinetune CONSTRUCTOR argument, so
it never reached the job: the run rendered ``do_eval: false`` and ``eval_data: ""``. The nested
``extra_hparams`` dict rode along under its own name and took ``eval_batches``/``eval_freq`` with
it. The arm trained BIT-IDENTICALLY to ``a27_allfixed`` -- the rendered configs differ only in
``out_dir`` -- wrote no ``metrics.eval.jsonl`` at all, was benchmarked at n=1000, and entered the
ledger at 9.7% as a held-out result. It is a duplicate of its own control.

Why nothing caught it. ``check_train_config.py`` guards that every key a typed config object can
EMIT reaches a rendered template; ``extra_hparams`` is free-form by design and bypasses that
entirely. ``check_launcher_config_reads.py`` guards the other direction (rendered -> read by the
launcher). ``report_unread_config`` is fatal for a config key nobody reads -- but these keys never
became config keys, so nothing was unread. And the run still got a distinct hash, so it looked like
a distinct arm everywhere a human would check.

The invariant, asserted in ``runs.train`` (the one construction path): no key of the hashed
``hparams`` bag may be a parameter name of ``train`` / ``make_finetune_run`` /
``SpeechFinetune.__init__``. Those names are derived by REFLECTION, so a constructor argument added
tomorrow is protected the day it is added.

Four halves, because each is a way the fix could rot:

  * **it fires** -- a ctor argument in ``extra_hparams`` raises, and the message names it;
  * **non-vacuous** -- a genuine hparam in the same bag still builds, so the guard is not simply
    rejecting ``extra_hparams`` outright;
  * **the value actually lands** -- passing ``eval_data`` properly sets it ON THE JOB, which is the
    half a pure "it raises" test cannot see. A fix that rejected the bad spelling without making
    the good one work would pass everything above;
  * **the real graph is clean** -- walked, not sampled, because the defect was a single call site
    that read exactly like its correct neighbours.

Run from the setup root (~1 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_hparams_destination.py
"""

import inspect
import logging
import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.finetune import SpeechFinetune  # noqa: E402
from speech_llm.full_duplex.sis_recipe.doriank.runs import (  # noqa: E402
    POLICY_LATEST,
    _construction_parameter_names,
    train,
)
from speech_llm.full_duplex.sis_recipe.doriank.train_config import (  # noqa: E402
    MOSHI_LIB,
    Corpus,
)

FAILURES: list[str] = []


def check(cond, msg, detail=""):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg} {detail}")
        FAILURES.append(msg)


CORPUS = Corpus(
    mix=tk.Path("/dev/null/corpus", hash_overwrite="hparams_dest_guard_corpus"),
    duration_sec=60,
)
EVAL = tk.Path("/dev/null/evalcorpus", hash_overwrite="hparams_dest_guard_eval")


def build(**kw):
    """Build one run through the REAL ``train()`` front door.

    The singleton cache is cleared first: ``JobSingleton`` hands back the SAME object for two
    constructions that hash alike, which would let a later assertion pass because ``a is b`` rather
    than because the property holds.
    """
    import sisyphus.job

    sisyphus.job.created_jobs.clear()
    kw.setdefault("tag", "hparams_dest_guard")
    tag = kw.pop("tag")
    return train(
        tag,
        model=MOSHI_LIB,
        corpus=CORPUS,
        policy=POLICY_LATEST,
        steps=100,
        evals=False,
        register=False,
        **kw,
    )


print("[1] the reflection really covers the construction path")
names = _construction_parameter_names()
# If any of these fell out of the set the assert would go quiet while still passing its own tests.
for n in (
    "eval_data",
    "seed",
    "train_data",
    "max_steps",
    "num_epochs",
    "lora_rank",
    "duration_sec",
    "knowledge_probe_data",
    "extra_hparams",
):
    check(n in names, f"{n!r} is recognised as an argument name")
# `hparams` must NOT be in the set: it is the bag itself, and listing it would make the assert
# complain about a caller that legitimately names the bag.
check("hparams" not in names, "'hparams' itself is excluded")
check("self" not in names and "tag" not in names, "'self'/'tag' excluded")

print("[2] a constructor argument in the hparams bag is REFUSED")
try:
    build(extra_hparams={"eval_data": EVAL})
    check(False, "extra_hparams={'eval_data': ...} raises")
except AssertionError as e:
    check(True, "extra_hparams={'eval_data': ...} raises")
    check("eval_data" in str(e), "and the message names the offending key", str(e)[:160])
# The nested-bag half of the same defect.
try:
    build(extra_hparams={"extra_hparams": {"eval_freq": 25}})
    check(False, "a nested extra_hparams dict raises")
except AssertionError:
    check(True, "a nested extra_hparams dict raises")

print("[3] non-vacuous -- a genuine hyper-parameter in the SAME bag still builds")
try:
    r = build(extra_hparams={"text_pad_weight": 0.01}, tag="hparams_dest_guard_ok")
    check(r.job.hparams.get("text_pad_weight") == 0.01, "text_pad_weight reaches hparams")
except AssertionError as e:
    check(False, "a real hparam still builds", str(e)[:160])

print("[4] passed properly, the value lands ON THE JOB")
r = build(eval_data=EVAL, tag="hparams_dest_guard_eval")
check(getattr(r.job, "eval_data", None) is not None, "train(eval_data=...) sets job.eval_data")
check("eval_data" not in (r.job.hparams or {}), "and does NOT also sit in hparams")

print("[5] `_anchored` routes by destination, not by hope")
src = (SETUP / "recipe/speech_llm/full_duplex/sis_recipe/doriank/training.py").read_text()
# The literal line that caused it. Its return is what made every unknown kwarg an hparam.
check(
    "extra_hparams=extra," not in src,
    "`extra_hparams=extra` (route-everything-to-hparams) is gone",
)
check(
    "_construction_parameter_names()" in src,
    "`_anchored` splits on the same reflected name set the assert uses",
)

print("[6] the REAL graph carries no argument name in any hparams bag")
from sisyphus.loader import config_manager  # noqa: E402

# Sisyphus turns the config path into a dotted MODULE name, and the setup's own directory name
# contains dashes -- so an absolute path is rejected ("Config name is invalid"). It must be given
# relative to the setup root, which is also where the recipe's own relative paths resolve.
os.chdir(SETUP)
config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
ctor = set(inspect.signature(SpeechFinetune.__init__).parameters) - {"self"}
offenders, n_jobs = [], 0
for job in tk.sis_graph.jobs():
    if type(job).__name__ != "SpeechFinetune":
        continue
    n_jobs += 1
    bad = sorted(set(getattr(job, "hparams", None) or {}) & ctor)
    if bad:
        offenders.append((job._sis_id().split(".")[-1], bad))
check(n_jobs > 0, f"walked the graph ({n_jobs} SpeechFinetune jobs)")
check(not offenders, "no arm carries a constructor argument in hparams", str(offenders))

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("all checks passed")
