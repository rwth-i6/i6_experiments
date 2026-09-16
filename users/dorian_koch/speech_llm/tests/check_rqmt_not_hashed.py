"""Guard: what a run asks the SCHEDULER for must never reach its Sisyphus hash.

The bug this exists for (2026-09-16). ``rqmt`` is famously not part of a Sisyphus hash -- that is
what makes a memory or walltime bump free of a re-run cascade, and ``finetune.py`` said so in a
comment two lines below the line that broke it. But ``Compute`` never wrote ``rqmt``: it lowered
``hours``/``gpus`` into the ``rqmt_time_h``/``gpu`` **hparams**, and ``hparams`` IS a hashed
constructor argument of ``SpeechFinetune``. So the safe-sounding rule was true of the dict the job
ends up with and false of the channel the value travelled through.

Raising a running arm's ``hours`` 24 -> 48, to get it onto a longer-walltime partition, re-hashed it
(``ajEFO4gPOlEj`` -> ``6IeG9qqj4s3S``). Sisyphus treats a changed hash as a brand-new job, so that
would have orphaned ~3,000 completed steps and restarted a 35 h run from zero -- to change nothing
about the experiment. ``hash_snapshot.py`` caught it before the manager's next loop.

The fix is a second channel: ``SpeechFinetune(compute=...)``, hash-EXCLUDED, carrying exactly the
scheduling facts. From ``RULE_RQMT_NOT_HASHED`` a run's ``Compute`` lowers there instead of into
``hparams``. Older runs keep the old channel, because removing the keys from THEIR hparams would
re-hash precisely the finished and in-flight runs the split exists to protect -- so both shapes are
pinned below, and the legacy half is asserted to still behave the old way.

Four things are checked, and each one is a way the fix could rot:

  * **hash-free** -- two runs differing only in ``hours`` / ``gpus`` hash identically;
  * **not merely dropped** -- ...and their ``rqmt`` still DIFFERS. A "fix" that routed the value
    nowhere would pass a hash-equality test perfectly while silently pinning every run to 23 h;
  * **non-vacuous** -- a run differing in a real experiment knob still re-hashes, so the equality
    above is a property of the scheduling keys and not of the comparison;
  * **no back door** -- ``extra_hparams={"rqmt_time_h": ...}`` is refused, since the bag is the
    escape hatch through which a scheduling key would re-enter the hash.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_rqmt_not_hashed.py
"""

import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.finetune import SpeechFinetune  # noqa: E402
from speech_llm.full_duplex.sis_recipe.doriank.runs import (  # noqa: E402
    POLICY_2026_09_16,
    POLICY_LATEST,
    RULE_RQMT_NOT_HASHED,
    SCHEDULING_HPARAMS,
    _policy_applies,
    train,
)
from speech_llm.full_duplex.sis_recipe.doriank.train_config import (  # noqa: E402
    MOSHI_LIB,
    Compute,
    Corpus,
    Optim,
)

FAILURES: list[str] = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


#: A stand-in corpus. ``mix`` only has to be a path in VALUE position -- the guard never runs the
#: job, it only builds it and reads the hash Sisyphus would give it.
CORPUS = Corpus(mix=tk.Path("/dev/null/corpus", hash_overwrite="rqmt_guard_corpus"), duration_sec=60)


def build(policy, *, compute=None, optim=None, extra_hparams=None, tag="rqmt_guard"):
    """Build one run through the REAL ``train()`` front door.

    ``register=False`` keeps it out of ``tk.register_output`` -- the guard must not add outputs to
    the graph it is inspecting.

    ⚠ The singleton cache is cleared first, and that is what makes the hash assertions mean
    anything. ``JobSingleton`` returns the SAME object for two constructions that hash alike, so
    once ``compute`` stopped being hashed, ``build(hours=24)`` and ``build(hours=48)`` handed back
    one identical job -- and ``hash(a) == hash(b)`` passed because ``a is b``, while the second
    ``hours`` was silently discarded. A vacuous pass in the exact place the guard exists to watch.
    Building each run in a fresh cache proves the hashes agree AND lets each keep its own ``rqmt``.

    (In the real recipe the sharing is harmless: two arms that hash alike differ in nothing but
    their scheduling request, so they are one experiment and were always one job.)
    """
    import sisyphus.job

    sisyphus.job.created_jobs.clear()
    return train(
        tag,
        model=MOSHI_LIB,
        corpus=CORPUS,
        policy=policy,
        steps=100,
        compute=compute,
        optim=optim,
        extra_hparams=extra_hparams,
        evals=False,
        register=False,
    )


def sis_hash(run):
    return run.job._sis_id()


def check_hours_and_gpus_are_hash_free():
    """The headline property, on a run at the current epoch."""
    a = build(POLICY_LATEST, compute=Compute(hours=24))
    b = build(POLICY_LATEST, compute=Compute(hours=48))
    check(sis_hash(a) == sis_hash(b), f"hours 24 vs 48 hash identically ({sis_hash(a)})")
    # ...and the value was not simply thrown away, which would pass the line above and silently pin
    # every run to the 23 h default.
    check(a.job.rqmt["time"] == 24, f"hours=24 reaches rqmt (got {a.job.rqmt['time']})")
    check(b.job.rqmt["time"] == 48, f"hours=48 reaches rqmt (got {b.job.rqmt['time']})")

    one = build(POLICY_LATEST, compute=Compute(gpus=1, hours=24))
    four = build(POLICY_LATEST, compute=Compute(gpus=4, hours=24))
    check(sis_hash(one) == sis_hash(four), "gpus 1 vs 4 hash identically")
    check(four.job.rqmt["gpu"] == 4, f"gpus=4 reaches rqmt (got {four.job.rqmt['gpu']})")
    check(
        four.job.rqmt["cpu"] == 24 and four.job.rqmt["mem"] == 96,
        f"cpu/mem scale per GPU (got {four.job.rqmt['cpu']}/{four.job.rqmt['mem']})",
    )

    # No compute at all must still be a valid run on the module defaults, and must hash the same as
    # one that spells those defaults out -- otherwise "add an explicit Compute" becomes a re-hash.
    bare = build(POLICY_LATEST)
    check(sis_hash(bare) == sis_hash(a), "omitting Compute hashes like any other compute")
    check(
        bare.job.rqmt["time"] == 23 and bare.job.rqmt["gpu"] == 1,
        f"defaults hold with no Compute (got {bare.job.rqmt})",
    )


def check_non_vacuous():
    """A real experiment knob must STILL move the hash, or the equality above proves nothing."""
    a = build(POLICY_LATEST, compute=Compute(hours=24), optim=Optim(lr=1e-4))
    b = build(POLICY_LATEST, compute=Compute(hours=24), optim=Optim(lr=2e-4))
    check(sis_hash(a) != sis_hash(b), "a differing lr DOES re-hash (comparison can fail)")


def check_scheduling_keys_absent_from_hparams():
    run = build(POLICY_LATEST, compute=Compute(gpus=4, hours=48))
    leaked = SCHEDULING_HPARAMS & set(run.job.hparams)
    check(not leaked, f"no scheduling key in the hashed hparams (leaked: {sorted(leaked)})")
    check(run.job.compute == {"gpu": 4, "rqmt_time_h": 48}, f"compute carries them instead (got {run.job.compute})")


def check_extra_hparams_back_door_is_closed():
    """``extra_hparams`` is the bag through which a scheduling key would re-enter the hash."""
    for key, value in (("rqmt_time_h", 48), ("gpu", 4)):
        try:
            build(POLICY_LATEST, extra_hparams={key: value})
        except AssertionError as e:
            check(key in str(e), f"extra_hparams={{{key!r}: ...}} is refused ({str(e)[:60]}...)")
        else:
            check(False, f"extra_hparams={{{key!r}: ...}} was ACCEPTED into the hashed bag")


def check_legacy_epoch_is_pinned():
    """Pre-epoch runs must keep the OLD channel, exactly.

    This is the half that looks like a bug and is not. Those runs' hparams contain the two keys; a
    later "cleanup" that stripped them everywhere would re-hash every finished arm in the ledger and
    every in-flight one -- the damage the split exists to avoid. So the old behaviour is pinned, and
    the fallback read in ``SpeechFinetune.__init__`` is asserted to still work.
    """
    check(
        not _policy_applies(POLICY_2026_09_16, RULE_RQMT_NOT_HASHED),
        "the rule does not reach back into POLICY_2026_09_16",
    )
    a = build(POLICY_2026_09_16, compute=Compute(hours=24))
    b = build(POLICY_2026_09_16, compute=Compute(hours=48))
    check(sis_hash(a) != sis_hash(b), "a pre-epoch run still re-hashes on `hours` (old shape preserved)")
    check("rqmt_time_h" in a.job.hparams, "a pre-epoch run keeps rqmt_time_h in hparams")
    check(a.job.rqmt["time"] == 24, f"the hparams fallback still feeds rqmt (got {a.job.rqmt['time']})")


def check_compute_beats_hparams():
    """Both channels populated -> the hash-free one wins, so a legacy value cannot pin a new run."""
    job = SpeechFinetune(
        adapter=MOSHI_LIB.adapter,
        venv_python_path=MOSHI_LIB.venv(),
        train_data=CORPUS.mix,
        max_steps=100,
        hparams={"rqmt_time_h": 23, "gpu": 1},
        compute={"rqmt_time_h": 48, "gpu": 4},
    )
    check(job.rqmt["time"] == 48 and job.rqmt["gpu"] == 4, f"compute wins over a stale hparams value (got {job.rqmt})")


def main():
    for fn in (
        check_hours_and_gpus_are_hash_free,
        check_non_vacuous,
        check_scheduling_keys_absent_from_hparams,
        check_extra_hparams_back_door_is_closed,
        check_legacy_epoch_is_pinned,
        check_compute_beats_hparams,
    ):
        print(f"\n== {fn.__name__}")
        fn()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
