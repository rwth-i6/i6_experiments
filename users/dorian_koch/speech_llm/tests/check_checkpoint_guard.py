"""Guard: an eval is never wired to a checkpoint that will never exist -- refused at GRAPH BUILD.

User request (2026-09-09): "When eval jobs depend on checkpoints that will never exist, there should
be an assert that catches that manager side."

Why the manager is the only place that can catch it: Sisyphus waits on input *paths*, not input
jobs. A job whose input lives under a run that has already FINISHED without writing it is not an
error to the manager -- it is merely ``waiting``, and it waits forever while ``error(0)`` says the
graph is healthy. Two earlier incidents were the milder, visible form of the same mistake and each
got a *static* rule in ``attach_knowledge_evals`` (a8-4gpu named steps past the run's end; a11_full
named steps its 1250 cadence never wrote). Those rules only see the one call path that goes through
``attach_knowledge_evals``; ``resolve_lora`` (``knowledge_benchmark_py``, ``fdb_benchmark_py``,
``voicebench_py``), ``FinetuneRun.checkpoint`` and the raw ``FinetuneRun.weights`` /
``optimizer_state`` paths did not go through it at all.

The guard under test, ``assert_checkpoint_will_exist``, sits in ``ResolveOverlayCheckpoint.__init__``
-- the one place every checkpoint reference is minted -- and consults three sources of truth in
order: the disk (a checkpoint that exists, exists), the run's finished marker (finished and absent
means never), and the run's declared shape (``max_steps`` / ``save_every`` for a run still to come).

Every rejection below is paired with an acceptance on the same fixture, so no case can pass by the
guard simply refusing everything.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_checkpoint_guard.py
"""

import inspect
import os
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import quick_knowledge_eval as qke  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.finetune import SpeechFinetune  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.speech_inference import (  # noqa: E402
    CHECKPOINT_DIR_KINDS,
    SAVE_EVERY,
    ResolveOverlayCheckpoint,
    assert_checkpoint_will_exist,
    impossible_checkpoint_reason,
)


class _Run:
    """Stands in for a training job: exactly what the guard reads off a ``tk.Path``'s creator --
    its job dir (for the ``finished`` marker) and its ``hparams``. Not a sisyphus ``Job`` on
    purpose: a real one would look for its marker under the real ``work/``."""

    def __init__(self, job_dir: str, hparams, finished: bool):
        self._dir = job_dir
        if hparams is not None:
            self.hparams = hparams
        os.makedirs(job_dir, exist_ok=True)
        if finished:
            Path(job_dir, "finished").touch()

    tags: set = set()  # sisyphus reads a creator's tags while hashing a job that consumes its path

    def _sis_path(self, path_type=None, *_a, **_k):
        return self._dir if path_type is None else os.path.join(self._dir, path_type)

    def _sis_id(self):
        return "stub/" + os.path.basename(self._dir)

    def finish(self):
        Path(self._dir, "finished").touch()


def _make_run(tmp, name, *, steps_written=(), hparams=None, finished=False, kind="lora"):
    job_dir = os.path.join(tmp, name)
    run_dir = os.path.join(job_dir, "output", "run_dir")
    os.makedirs(run_dir, exist_ok=True)
    for s in steps_written:
        if kind in CHECKPOINT_DIR_KINDS:
            d = os.path.join(run_dir, "checkpoints", f"checkpoint_{s:06d}", "consolidated")
            os.makedirs(d)
            Path(d, "lora.safetensors").touch()
            Path(d, "config.json").touch()
        else:
            d = os.path.join(run_dir, "consolidated")
            os.makedirs(d, exist_ok=True)
            Path(d, "trained_heads.safetensors" if s is None else f"trained_heads.step{s}.safetensors").touch()
    creator = _Run(job_dir, hparams, finished)
    return tk.Path("run_dir", creator=creator), creator


def _rejects(run_dir, kind, step) -> str:
    try:
        assert_checkpoint_will_exist(run_dir, kind, step)
    except AssertionError as e:
        return str(e)
    raise SystemExit(f"FAIL: {kind} step {step} of {run_dir.get_path()} should have been rejected")


def _accepts(run_dir, kind, step):
    try:
        assert_checkpoint_will_exist(run_dir, kind, step)
    except AssertionError as e:
        raise SystemExit(f"FAIL: {kind} step {step} of {run_dir.get_path()} should be accepted:\n  {e}")


def check_finished_run_missing_step(tmp):
    """The case the user asked for: the run is done, the step is not there, so it never will be."""
    run, _ = _make_run(tmp, "done", steps_written=(500, 600), hparams={"max_steps": 600}, finished=True)
    msg = _rejects(run, "lora", 700)
    assert "checkpoint_000700" in msg and "will never exist" in msg, msg
    assert "checkpoint_000500" in msg, "the message must list what the run DID write:\n" + msg
    # The same run, a step it wrote, and LATEST: both fine.
    _accepts(run, "lora", 500)
    _accepts(run, "lora", 600)
    _accepts(run, "lora", None)
    # ...and a finished run with NO checkpoint at all cannot even serve LATEST.
    empty, _ = _make_run(tmp, "done_empty", finished=True)
    msg = _rejects(empty, "lora", None)
    assert "no checkpoints/ at all" in msg, msg
    print("PASS  a finished run that never wrote the step is refused, and it says what the run wrote")


def check_disk_beats_declared_shape(tmp):
    """Evidence wins: a checkpoint on disk exists even if the declared cadence says it should not."""
    run, _ = _make_run(tmp, "stray", steps_written=(250, 600), hparams={"max_steps": 600}, finished=True)
    assert impossible_checkpoint_reason(250, max_steps=600, save_every=SAVE_EVERY) is not None
    _accepts(run, "lora", 250)
    print("PASS  a checkpoint that is on disk is accepted whatever the cadence rule says")


def check_pending_run_declared_shape(tmp):
    """A run still to come is judged by its hparams -- the same rule attach_knowledge_evals uses."""
    run, creator = _make_run(tmp, "pending", hparams={"max_steps": 600})
    _accepts(run, "lora", 500)
    _accepts(run, "lora", 600)  # max_steps itself: run_training's final save
    _accepts(run, "lora", None)
    assert "not a checkpoint" in _rejects(run, "lora", 700)
    assert "past the end" in _rejects(run, "lora", 1000)
    # A per-run save_every override is honoured (the a11_full shape).
    full, _ = _make_run(tmp, "pending_full", hparams={"max_steps": 3750, "save_every": 1250})
    _accepts(full, "full", 1250)
    _accepts(full, "full", 3750)
    msg = _rejects(full, "full", 1000)
    assert "1250" in msg, msg
    # A num_epochs run: no max_steps, so only the cadence rule applies while it is pending...
    epochs, ep_creator = _make_run(tmp, "pending_epochs", hparams={"num_epochs": 1})
    _accepts(epochs, "lora", 1000)
    assert "not a checkpoint" in _rejects(epochs, "lora", 700)
    # ...and once it has finished at step 871, the disk rule takes over.
    for s in (500, 871):
        d = os.path.join(epochs.get_path(), "checkpoints", f"checkpoint_{s:06d}", "consolidated")
        os.makedirs(d)
    ep_creator.finish()
    _accepts(epochs, "lora", 871)
    assert "will never exist" in _rejects(epochs, "lora", 1000)
    print("PASS  a pending run is judged by its declared max_steps/save_every, a finished one by disk")


def check_creator_without_hparams(tmp):
    """MoshiFinetune / RLFinetune / AudexDuplexFinetune declare no hparams dict: nothing can be said
    while they run, everything can once they finish."""
    run, creator = _make_run(tmp, "nohp", hparams=None)
    _accepts(run, "lora", 12345)
    creator.finish()
    assert "will never exist" in _rejects(run, "lora", 12345)
    print("PASS  a creator without hparams is unconstrained while pending and disk-checked once done")


def check_personaplex_layout(tmp):
    run, _ = _make_run(
        tmp, "pplex", steps_written=(300,), hparams={"max_steps": 600}, finished=True, kind="personaplex_heads"
    )
    _accepts(run, "personaplex_heads", 300)
    msg = _rejects(run, "personaplex_heads", 600)
    assert "trained_heads.step600.safetensors" in msg, msg
    msg = _rejects(run, "personaplex_heads", None)
    assert "trained_heads.safetensors" in msg, msg
    Path(run.get_path(), "consolidated", "trained_heads.safetensors").touch()
    _accepts(run, "personaplex_heads", None)
    # The static cadence rule does NOT apply to this layout (its cadence is not declared that way):
    # a pending pplex run accepts any step.
    pend, _ = _make_run(tmp, "pplex_pending", hparams={"max_steps": 600}, kind="personaplex_heads")
    _accepts(pend, "personaplex_heads", 700)
    print("PASS  the PersonaPlex heads layout is disk-checked in its own location")


def check_creator_less_path(tmp):
    """An externally supplied run dir (no Sisyphus creator): its existence is the 'finished' signal."""
    ext = os.path.join(tmp, "external", "run_dir")
    os.makedirs(os.path.join(ext, "checkpoints", "checkpoint_000500", "consolidated"))
    _accepts(tk.Path(ext), "lora", 500)
    assert "will never exist" in _rejects(tk.Path(ext), "lora", 900)
    _accepts(tk.Path(os.path.join(tmp, "not_there_yet", "run_dir")), "lora", 900)
    print("PASS  a creator-less run dir is judged by what is on disk, or left alone if absent")


def check_real_resolver_is_guarded(tmp):
    """The guard is only worth anything if the REAL constructor calls it. Drive it."""
    run, _ = _make_run(tmp, "real", steps_written=(500,), hparams={"max_steps": 500}, finished=True)
    try:
        ResolveOverlayCheckpoint(run_dir=run, overlay_kind="lora", step=1000)
    except AssertionError as e:
        assert "will never exist" in str(e), e
    else:
        raise SystemExit("FAIL: ResolveOverlayCheckpoint accepted a step its finished run never wrote")
    src = inspect.getsource(ResolveOverlayCheckpoint.__init__)
    assert "assert_checkpoint_will_exist(run_dir, overlay_kind, step)" in src, src
    print("PASS  ResolveOverlayCheckpoint itself refuses a never-to-exist step at construction")


def check_wiring():
    """Every other path that names a checkpoint step goes through the same rule."""
    from speech_llm.full_duplex.sis_recipe.doriank.runs import FinetuneRun

    for name in ("weights", "optimizer_state"):
        assert "self._assert_checkpoint(step)" in inspect.getsource(getattr(FinetuneRun, name)), name
    assert "assert_checkpoint_will_exist(self.out_rundir, self.overlay_kind, step)" in inspect.getsource(
        FinetuneRun._assert_checkpoint
    )
    # attach_knowledge_evals applies the identical predicate, not a private copy of it.
    src = inspect.getsource(qke.attach_knowledge_evals)
    assert "impossible_checkpoint_reason(s, max_steps=max_steps, save_every=save_every)" in src, src
    assert qke.SAVE_EVERY == SAVE_EVERY
    # The trainer's template default is the constant the rule assumes.
    ft_src = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/finetune.py").read_text()
    assert f'hp.get("save_every", {SAVE_EVERY})' in ft_src, "finetune.py's save_every default drifted from SAVE_EVERY"
    # And the real training job exposes what the stub above stood in for.
    assert callable(getattr(SpeechFinetune, "_sis_path", None))
    assert "self.hparams = hparams" in inspect.getsource(SpeechFinetune.__init__)
    # The predicate itself, at its edges.
    assert impossible_checkpoint_reason(None, max_steps=100) is None, "LATEST is always possible"
    assert impossible_checkpoint_reason(120, max_steps=120) is None, "max_steps itself is always written"
    assert impossible_checkpoint_reason(500, max_steps=None) is None
    assert impossible_checkpoint_reason(0, max_steps=None) is not None
    print("PASS  FinetuneRun paths, attach_knowledge_evals and finetune.py all share the one rule")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        check_finished_run_missing_step(tmp)
        check_disk_beats_declared_shape(tmp)
        check_pending_run_declared_shape(tmp)
        check_creator_without_hparams(tmp)
        check_personaplex_layout(tmp)
        check_creator_less_path(tmp)
        check_real_resolver_is_guarded(tmp)
    check_wiring()
    print("ALL PASS")
