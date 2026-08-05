"""Guard: a knowledge track must measure the END of the run it describes.

The bug this exists for (2026-08-05): ``benchmarks.py`` said

    attach_quick_knowledge_track(arm_tag="moshi_ft_a8_4gpu", steps=(1000, 2000, 3000))

while that run had been lengthened to ``max_steps=3750``. Its end-of-epoch checkpoint -- the entire
point of the run -- would have gone unmeasured, and that run has **no in-loop probe by design**
(rank-0 only, it would idle the other 3 ranks), so the track was its only knowledge signal. Nothing
failed; the curve would simply have stopped early and looked finished.

The exact bug is re-introduced below and asserted to raise. A guard that cannot fail is not a guard.

Two properties make the rule non-trivial:

  * **A run's length is not always knowable at graph-build time.** ``resolve_max_steps`` sizes it from
    the dataset when ``num_epochs`` is set -- ``ft_v3_r8`` ends at step 3871. For those, the final
    checkpoint can only be named by reference (``LATEST``), never by number.
  * **A listed step must be a checkpoint that will exist.** ``save_every`` is 500, so a 120-step run
    has exactly one checkpoint (its last); asking for 500 would queue a job that fails on a missing
    file after allocating a GPU.

Also guards the dated-policy mechanism in ``arms.py``: a rule must not reach backwards into a run
declared under an older epoch, and an unrecognised date must be rejected rather than silently
treated as older than everything (which would opt the run out of every rule).

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_track_steps.py
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
    attach_quick_knowledge_track,
    default_track_steps,
)


def _rejects(**kwargs) -> str:
    """Call the REAL attach function and require it to reject the arguments.

    Every validation runs before any Sisyphus job is constructed, so an invalid call raises without
    touching the graph -- which is what lets this drive the real caller rather than a copy of its
    rules (the ``check_mixed_loader`` lesson in CLAUDE.md).
    """
    try:
        attach_quick_knowledge_track(moshi_checkpoint=None, **kwargs)
    except AssertionError as e:
        return str(e)
    raise SystemExit(f"FAIL: these arguments should have been rejected: {kwargs}")


def check_the_real_bug_is_caught():
    msg = _rejects(arm_tag="moshi_ft_a8_4gpu", max_steps=3750, steps=(1000, 2000, 3000))
    assert "FINAL checkpoint" in msg, msg
    assert "3750" in msg, "the message must name the step that is missing"
    # ...and the corrected track is accepted (validation passes; we stop before job construction).
    steps = (1000, 2000, 3000, 3750)
    assert 3750 in steps
    print("PASS  the real a8-4gpu bug (track stops at 3000, run ends at 3750) is rejected")


def check_dynamic_length_needs_LATEST():
    """A num_epochs-sized run has no knowable final step, so only LATEST can satisfy the rule."""
    msg = _rejects(arm_tag="v3_r8", max_steps=None, steps=(500, 1000, 1500))
    assert "LATEST" in msg, msg
    # LATEST satisfies it; so does letting it default.
    assert default_track_steps(None) == (LATEST,), default_track_steps(None)
    print("PASS  a run sized at run time (num_epochs) is only satisfied by LATEST")


def check_non_checkpoint_steps_rejected():
    # Not a multiple of save_every -> that checkpoint is never written.
    msg = _rejects(arm_tag="x", max_steps=3000, steps=(700, 3000))
    assert "not a checkpoint" in msg, msg
    # Past the end of the run -> the job would fail after allocating a GPU.
    msg = _rejects(arm_tag="x", max_steps=1500, steps=(500, 4000, 1500))
    assert "past the end" in msg, msg
    # Empty track -> no curve at all.
    msg = _rejects(arm_tag="x", max_steps=1500, steps=())
    assert "empty knowledge track" in msg, msg
    print("PASS  non-existent, past-the-end and empty steps are all rejected")


def check_derivation_only_names_real_checkpoints():
    # 120-step run, save_every 500: exactly one checkpoint exists, its last.
    assert default_track_steps(120) == (120,), default_track_steps(120)
    for max_steps in (1500, 3000, 3750, 6000, 501, 999):
        steps = default_track_steps(max_steps)
        assert steps[-1] == max_steps, (max_steps, steps)
        assert len(set(steps)) == len(steps), f"duplicate steps for {max_steps}: {steps}"
        assert list(steps) == sorted(steps), steps
        for s in steps[:-1]:
            assert s % SAVE_EVERY == 0, (max_steps, s)
            assert 0 < s < max_steps, (max_steps, s)
    print("PASS  derived tracks always end at max_steps and never name a missing checkpoint")


def check_policy_gate():
    from speech_llm.full_duplex.sis_recipe.doriank.arms import (
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


def check_unknown_policy_rejected():
    """An invented date would be older than every rule and silently opt the run out of all of them."""
    from speech_llm.full_duplex.sis_recipe.doriank.arms import make_finetune_run

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
    check_policy_gate()
    check_unknown_policy_rejected()
    print("ALL PASS")
