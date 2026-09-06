"""Guard: an overlay checkpoint must resolve the layout its run ACTUALLY wrote.

The bug this exists for (2026-08-21). ``a11_full`` is a **full** finetune: its checkpoints hold
``consolidated/model.safetensors`` and nothing else. But ``resolve_lora`` hardcoded
``overlay_kind="lora"``, so ``attach_evals`` built a resolver that symlinked a ``lora.safetensors``
and a ``config.json`` the run never wrote.

``os.symlink`` creates a DANGLING link without complaining. So the resolver exited 0, Sisyphus
wrote its ``finished.tar.gz``, and the downstream ``SpeechInference`` was released -- where it died
with a bare::

    assert False, "Job isn't runnable, probably some inputs are not ready"

leaving an EMPTY ``error.run.1``, no ``log.run.1`` at all, and a SLURM state of ``COMPLETED`` in one
second. Nothing named the missing file or the job that wanted it, and because Sisyphus refuses to
run a graph containing errored jobs, the whole graph stalled for two weeks.

Two independent halves, because either alone is vacuous:
  * the resolver must REFUSE a target that does not exist (the silence is the bug, not the mismatch);
  * ``attach_evals`` must pass the run's real ``overlay_kind``, else the resolver is simply told to
    build the wrong thing correctly.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_overlay_resolution.py
"""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm.speech_inference import (  # noqa: E402
    ResolveOverlayCheckpoint,
)


class _Out:
    """Stand-in for a sisyphus output_path handle: all run() needs is .get()."""

    def __init__(self, p):
        self._p = str(p)

    def get(self):
        return self._p


def _run_resolver(run_dir: Path, overlay_kind: str, step: int, out: Path):
    """Drive the REAL ResolveOverlayCheckpoint.run() without constructing a Sisyphus job.

    Sisyphus's Job.__new__ intercepts construction (it is the unpickling hook), and the constructor
    calls self.output_path(), which needs a live graph. But run() only ever touches
    self.{run_dir,overlay_kind,step,out_weights,out_config,_link}, so we hand it a namespace with
    exactly those and call the real unbound method -- the code under test is the shipped run(), not
    a reimplementation of its rules.
    """
    weights_name = {"lora": "lora.safetensors", "full": "model.safetensors"}[overlay_kind]
    job = SimpleNamespace(
        run_dir=_Out(run_dir),
        overlay_kind=overlay_kind,
        step=step,
        out_weights=_Out(out / weights_name),
        out_config=_Out(out / "config.json") if overlay_kind == "lora" else None,
        # _link is a staticmethod, so this binds as a plain 2-arg function, exactly as run() calls it.
        _link=ResolveOverlayCheckpoint._link,
    )
    ResolveOverlayCheckpoint.run(job)


def _make_full_ft_checkpoint(root: Path, step: int) -> None:
    """A full finetune writes model.safetensors + trainer_state.pt. No config, no lora."""
    d = root / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"\0" * 16)
    (d / "trainer_state.pt").write_bytes(b"\0" * 16)


def check_wrong_kind_is_refused_not_dangled():
    """The exact a11_full failure: a full-FT checkpoint asked for the LoRA layout."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run_dir = td / "run_dir"
        _make_full_ft_checkpoint(run_dir, 1250)
        out = td / "out"
        out.mkdir()
        try:
            _run_resolver(run_dir, "lora", 1250, out)
        except AssertionError as e:
            msg = str(e)
            assert "lora.safetensors" in msg or "config.json" in msg, msg
            assert "model.safetensors" in msg, (
                "the message must list what the checkpoint DOES hold -- that is the line that "
                f"turns this into a one-look diagnosis: {msg}"
            )
        else:
            # This is how it shipped: two dangling links and a successful exit.
            dangling = [p.name for p in out.iterdir() if p.is_symlink() and not p.exists()]
            raise SystemExit(
                f"FAIL: the resolver reported success with dangling symlinks {dangling}. "
                f"That silence is the bug -- it releases a downstream job that then fails with "
                f"no mention of this path."
            )
    print("PASS  a wrong overlay_kind is refused here, not left to dangle into a downstream job")


def check_right_kind_still_resolves():
    """Non-vacuity: the guard above must be rejecting the MISMATCH, not the resolver as such."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run_dir = td / "run_dir"
        _make_full_ft_checkpoint(run_dir, 1250)
        out = td / "out"
        out.mkdir()
        _run_resolver(run_dir, "full", 1250, out)
        link = out / "model.safetensors"
        assert link.is_symlink(), "the resolver must produce a symlink"
        assert link.exists(), "...and it must not dangle"
        assert not (out / "config.json").exists(), (
            "a full finetune has no LoRA config; emitting one would put a file in the output that "
            "the offline driver would then try to parse"
        )
    print("PASS  the matching overlay_kind resolves to a live model.safetensors and no config")


def check_attach_evals_passes_the_runs_kind():
    """The resolver being right is useless if the caller keeps telling it 'lora'.

    That is precisely the original defect: `attach_evals` DID pass `lora_rank=None` to the backend
    spec (so the driver would not wrap LoRA), but the checkpoint resolver was still built with the
    hardcoded LoRA layout. Half-threaded is what made it look handled.
    """
    src = (SETUP / "recipe/speech_llm/full_duplex/sis_recipe/doriank/runs.py").read_text()
    body = src.split("def attach_evals(")[1].split("\ndef ")[0]
    assert "moshi_overlay_kind=run.overlay_kind" in body, (
        "attach_evals must pass the run's own overlay_kind to attach_knowledge_evals, not rely on "
        "resolve_lora's default"
    )
    assert 'lora_rank=None if run.overlay_kind == "full"' in body, (
        "the backend-spec half must stay too -- both are needed and they are read by different jobs"
    )

    # ...and resolve_lora must actually honour it rather than accept and ignore the argument.
    from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import resolve_lora

    import inspect

    sig = inspect.signature(resolve_lora)
    assert "overlay_kind" in sig.parameters, "resolve_lora must take an overlay_kind"
    assert sig.parameters["overlay_kind"].default == "lora", (
        "the default must stay 'lora' so every existing LoRA arm's ResolveOverlayCheckpoint hash "
        "is unchanged -- a re-hash here would orphan every benchmarked eval we already have"
    )
    print("PASS  attach_evals passes the run's overlay_kind, and the LoRA default is preserved")


if __name__ == "__main__":
    check_wrong_kind_is_refused_not_dangled()
    check_right_kind_still_resolves()
    check_attach_evals_passes_the_runs_kind()
    print("ALL PASS")
