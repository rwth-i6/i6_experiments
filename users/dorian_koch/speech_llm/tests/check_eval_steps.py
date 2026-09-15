"""Guard: a checkpoint evals must measure the END of the run it describes.

The bug this exists for (2026-08-05): ``benchmarks.py`` said

    attach_knowledge_evals(run_tag="moshi_ft_a8_4gpu", steps=(1000, 2000, 3000))

while that run had been lengthened to ``max_steps=3750``. Its end-of-epoch checkpoint -- the entire
point of the run -- would have gone unmeasured, and that run has **no in-loop probe by design**
(rank-0 only, it would idle the other 3 ranks), so the track was its only knowledge signal. Nothing
failed; the curve would simply have stopped early and looked finished.

The exact bug is re-introduced below and asserted to raise. A guard that cannot fail is not a guard.

Two properties make the rule non-trivial:

  * **A run's length is not always knowable at graph-build time.** ``resolve_max_steps`` sizes it from
    the dataset when ``num_epochs`` is set -- ``ft_v3_r8`` ends at step 3871. For those, the final
    checkpoint can only be named by reference (``LATEST``), never by number.
  * **A listed step must be a checkpoint that will exist.** ``save_every`` is 500 by default, so a
    120-step run has exactly one checkpoint (its last); asking for 500 would queue a job that fails
    on a missing file after allocating a GPU.
  * **...and a run may override that cadence.** ``a11_full`` saves every 1250 because a full-model
    checkpoint is ~46 GB. Validating its evals against the module default accepted steps that were
    never written -- see ``check_per_run_save_every_override``.

Also guards the dated-policy mechanism in ``arms.py``: a rule must not reach backwards into a run
declared under an older epoch, and an unrecognised date must be rejected rather than silently
treated as older than everything (which would opt the run out of every rule).

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_eval_steps.py
"""

import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm.quick_knowledge_eval import (  # noqa: E402
    LATEST,
    SAVE_EVERY,
    attach_knowledge_evals,
    default_eval_steps,
)


def _rejects(**kwargs) -> str:
    """Call the REAL attach function and require it to reject the arguments.

    Every validation runs before any Sisyphus job is constructed, so an invalid call raises without
    touching the graph -- which is what lets this drive the real caller rather than a copy of its
    rules (the ``check_mixed_loader`` lesson in CLAUDE.md).
    """
    try:
        attach_knowledge_evals(moshi_checkpoint=None, **kwargs)
    except AssertionError as e:
        return str(e)
    raise SystemExit(f"FAIL: these arguments should have been rejected: {kwargs}")


def _validation_passes(**kwargs):
    """Require ``attach_knowledge_evals`` to get PAST its argument validation.

    It constructs real jobs once validation succeeds, which a login-node guard must not do -- so
    anything that is not an ``AssertionError`` means the arguments were accepted and we stopped in
    the job-construction that follows (``moshi_checkpoint=None`` gets no further). Only an
    ``AssertionError`` is a rejection.
    """
    try:
        attach_knowledge_evals(moshi_checkpoint=None, **kwargs)
    except AssertionError as e:
        raise SystemExit(f"FAIL: these arguments should have been accepted: {kwargs}\n  {e}")
    except Exception:
        pass


def check_the_real_bug_is_caught():
    msg = _rejects(run_tag="moshi_ft_a8_4gpu", max_steps=3750, steps=(1000, 2000, 3000))
    assert "FINAL checkpoint" in msg, msg
    assert "3750" in msg, "the message must name the step that is missing"
    # ...and the corrected track is accepted (validation passes; we stop before job construction).
    steps = (1000, 2000, 3000, 3750)
    assert 3750 in steps
    print("PASS  the real a8-4gpu bug (track stops at 3000, run ends at 3750) is rejected")


def check_dynamic_length_needs_LATEST():
    """A num_epochs-sized run has no knowable final step, so only LATEST can satisfy the rule."""
    msg = _rejects(run_tag="v3_r8", max_steps=None, steps=(500, 1000, 1500))
    assert "LATEST" in msg, msg
    # LATEST satisfies it; so does letting it default.
    assert default_eval_steps(None) == (LATEST,), default_eval_steps(None)
    print("PASS  a run sized at run time (num_epochs) is only satisfied by LATEST")


def check_non_checkpoint_steps_rejected():
    # Not a multiple of save_every -> that checkpoint is never written.
    msg = _rejects(run_tag="x", max_steps=3000, steps=(700, 3000))
    assert "not a checkpoint" in msg, msg
    # Past the end of the run -> the job would fail after allocating a GPU.
    msg = _rejects(run_tag="x", max_steps=1500, steps=(500, 4000, 1500))
    assert "past the end" in msg, msg
    # Empty track -> no curve at all.
    msg = _rejects(run_tag="x", max_steps=1500, steps=())
    assert "empty checkpoint evals" in msg, msg
    print("PASS  non-existent, past-the-end and empty steps are all rejected")


def check_derivation_only_names_real_checkpoints():
    # 120-step run, save_every 500: exactly one checkpoint exists, its last.
    assert default_eval_steps(120) == (120,), default_eval_steps(120)
    for max_steps in (1500, 3000, 3750, 6000, 501, 999):
        steps = default_eval_steps(max_steps)
        assert steps[-1] == max_steps, (max_steps, steps)
        assert len(set(steps)) == len(steps), f"duplicate steps for {max_steps}: {steps}"
        assert list(steps) == sorted(steps), steps
        for s in steps[:-1]:
            assert s % SAVE_EVERY == 0, (max_steps, s)
            assert 0 < s < max_steps, (max_steps, s)
    print("PASS  derived tracks always end at max_steps and never name a missing checkpoint")


def check_per_run_save_every_override():
    """A run that overrides ``save_every`` must have its evals validated against ITS cadence.

    The bug this exists for (2026-08-21): ``a11_full`` sets ``save_every=1250`` in ``extra_hparams``
    (a full-model checkpoint is ~15 GB of weights + ~31 GB of optimizer state, so the 500 default
    would have written ~350 GB), but its ``eval_steps`` stayed at a8-4gpu's ``(1000, 2000, 3000)``.
    Every one of those IS a multiple of the module-level ``SAVE_EVERY`` of 500, so the assert in
    ``attach_knowledge_evals`` -- which had no way to know about the override -- passed. Three
    ``ResolveOverlayCheckpoint`` jobs then failed on
    ``checkpoint_001000 not found; have ['checkpoint_001250', 'checkpoint_002500', 'checkpoint_003750']``
    and, because Sisyphus will not run a graph with errored jobs, stalled the manager and with it the
    headline LoRA-vs-full-FT result.

    Both halves are checked, because either alone is vacuous: the validator must reject the mismatch
    when told the true cadence, AND ``attach_evals`` must actually tell it.
    """
    # -- half 1: the validator catches it, given the run's real cadence.
    # a11_full as it really shipped: 3750 IS present, so the final-checkpoint rule is satisfied and
    # the save_every rule is the one under test.
    msg = _rejects(run_tag="moshi_ft_a11_full", max_steps=3750, steps=(1000, 2000, 3000, 3750), save_every=1250)
    assert "not a checkpoint" in msg, msg
    assert "1250" in msg, "the message must name the cadence that makes the step impossible"
    # Non-vacuity: those same steps PASS validation under the default cadence -- which is exactly how
    # this shipped. Were that not so, the rejection above would not be evidence that the *override*
    # is what caught it.
    _validation_passes(run_tag="x", max_steps=3750, steps=(1000, 2000, 3000, 3750))
    # ...and a11_full's corrected track passes under its own cadence.
    _validation_passes(run_tag="x", max_steps=3750, steps=(1250, 2500, 3750), save_every=1250)

    # -- half 2: FinetuneRun.save_every reads the override, not the global default.
    from speech_llm.full_duplex.sis_recipe.doriank.runs import FinetuneRun

    class _Job:
        def __init__(self, hparams):
            self.hparams = hparams

    def _run(hparams):
        return FinetuneRun(tag="t", job=_Job(hparams), policy="2026-08-05", max_steps=3750, lora_rank=128)

    assert _run({"save_every": 1250}).save_every == 1250, "an override must win"
    assert _run({}).save_every == SAVE_EVERY, "no override -> finetune.py's default"

    # -- half 3: attach_evals actually forwards it. The property being right is useless if the call
    # site keeps letting the parameter default; that is precisely the shape of the original bug.
    src = (Path(SETUP) / "recipe/speech_llm/full_duplex/sis_recipe/doriank/runs.py").read_text()
    body = src.split("def attach_evals(")[1].split("\ndef ")[0]
    assert "save_every=run.save_every" in body, (
        "attach_evals must pass the run's own save_every to attach_knowledge_evals, else a run that "
        "overrides the cadence is validated against the wrong one"
    )
    print("PASS  a per-run save_every override is what the evals are validated against")


def check_policy_gate():
    from speech_llm.full_duplex.sis_recipe.doriank.runs import (
        KNOWN_POLICIES,
        POLICY_2026_07_31,
        POLICY_2026_08_05,
        POLICY_LATEST,
        _policy_applies,
    )

    # A rule must NOT reach backwards into a run declared under an older epoch -- that is the whole
    # point: changing a default would otherwise re-hash and re-train every existing run.
    assert not _policy_applies(POLICY_2026_07_31, POLICY_2026_08_05)
    assert _policy_applies(POLICY_2026_08_05, POLICY_2026_08_05), "a rule applies on its own date"
    assert _policy_applies(POLICY_LATEST, POLICY_2026_07_31), "newer runs still get older rules"
    # ISO dates must order as strings, which is what the whole comparison rests on.
    assert POLICY_2026_07_31 < POLICY_2026_08_05
    assert POLICY_LATEST == max(KNOWN_POLICIES), "POLICY_LATEST must be the newest declared epoch"
    print("PASS  dated rules never reach backwards; POLICY_LATEST is the newest epoch")


def check_probe_batch_rule_skips_probe_free_runs():
    """RULE_PROBE_BATCH must fire on a probing arm and NOT on one with the probe switched off.

    The S-series passes the probe SET but `every=0` to disable probing. A rule keyed only on
    `probe.data is not None` therefore emits `knowledge_probe_batch_size` on runs that never probe,
    re-hashing them to configure something they do not run -- which is exactly what happened on the
    first cut of this rule and would have orphaned seven queued arms. Caught by hash_snapshot; this
    keeps it caught.
    """
    from speech_llm.full_duplex.sis_recipe.doriank.runs import (
        POLICY_2026_09_15,
        POLICY_LATEST,
        PROBE_BATCH_SIZE,
        RULE_PROBE_BATCH,
        _policy_applies,
    )
    from speech_llm.full_duplex.sis_recipe.doriank.train_config import Probe

    def fires(policy, probe):
        return bool(
            _policy_applies(policy, RULE_PROBE_BATCH)
            and probe is not None
            and getattr(probe, "data", None) is not None
            and (getattr(probe, "every", None) or 0) > 0
            and getattr(probe, "batch_size", None) is None
        )

    sentinel = object()
    assert fires(POLICY_LATEST, Probe(data=sentinel, every=50)), (
        "a new probing arm must get the batched probe -- otherwise the rule is inert"
    )
    assert not fires(POLICY_LATEST, Probe(data=sentinel, every=0)), (
        "probe-free arm (every=0) must NOT get knowledge_probe_batch_size -- it re-hashes a run "
        "to configure something it never executes"
    )
    assert not fires(POLICY_2026_09_15, Probe(data=sentinel, every=50)), (
        "the rule reached backwards into an older epoch"
    )
    assert not fires(POLICY_LATEST, Probe(data=sentinel, every=50, batch_size=4)), (
        "an arm that states its own probe batch size must win over the default"
    )
    assert PROBE_BATCH_SIZE > 4, "the point of the rule is a LARGER batch than the old default of 4"
    print("PASS  probe-batch rule fires on probing arms only, and never backwards")


def check_unknown_policy_rejected():
    """An invented date would be older than every rule and silently opt the run out of all of them."""
    from speech_llm.full_duplex.sis_recipe.doriank.runs import make_finetune_run

    try:
        make_finetune_run(
            "bogus", policy="2020-01-01", max_steps=100, adapter=None, venv_python_path=None, train_data=None
        )
    except AssertionError as e:
        assert "unknown policy date" in str(e), e
        print("PASS  an unrecognised policy date is rejected instead of silently opting out")
        return
    raise SystemExit("FAIL: an unknown policy date was accepted")


if __name__ == "__main__":
    check_the_real_bug_is_caught()
    check_dynamic_length_needs_LATEST()
    check_non_checkpoint_steps_rejected()
    check_derivation_only_names_real_checkpoints()
    check_per_run_save_every_override()
    check_policy_gate()
    check_probe_batch_rule_skips_probe_free_runs()
    check_unknown_policy_rejected()
    print("ALL PASS")
