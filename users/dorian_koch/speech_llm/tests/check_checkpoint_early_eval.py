"""Guard: an eval of a named checkpoint becomes runnable as soon as that checkpoint is complete, while the run trains.

Before 2026-11-05 every ``ResolveOverlayCheckpoint`` waited on the run's ``run_dir`` output, which Sisyphus marks
available only when the whole run finishes: a 2,000-step RL run got no s250 eval for days. ``resolve_lora`` now hands
the resolver ``checkpoint_run_dir(...)``: the same path (same hash) with a ``tk.Path(available=...)`` that is true once
``checkpoint_<step>/consolidated`` holds the weights and its ``config.json`` (written after them).

Driven through the real caller (``knowledge_benchmark.resolve_lora``); each acceptance paired with a rejection:
  * named step, checkpoint complete, run still training -> runnable;
  * the same with only the weights (save in progress) -> waiting; a step not yet written -> waiting;
  * step=None (latest) -> waits for the run, then runnable;
  * the resolver's hash equals the one built on the plain run dir.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_checkpoint_early_eval.py
"""

import os
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import resolve_lora  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.speech_inference import ResolveOverlayCheckpoint  # noqa: E402


class _Run:
    """Stands in for a training job: its job dir, its declared shape, and Sisyphus's finished view of its outputs."""

    tags: set = set()

    def __init__(self, job_dir: str):
        self._dir = job_dir
        self.hparams = {"max_steps": 1000, "save_every": 50}
        os.makedirs(os.path.join(job_dir, "output", "run_dir"), exist_ok=True)

    def _sis_path(self, path_type=None, *_a, **_k):
        return self._dir if path_type is None else os.path.join(self._dir, path_type)

    def _sis_id(self):
        return "stub/" + os.path.basename(self._dir)

    def __deepcopy__(self, memo):  # a real Job is shared, not copied, when Sisyphus deep-copies a consumer's args
        return self

    def _sis_finished(self):
        return os.path.exists(os.path.join(self._dir, "finished"))

    def path_available(self, path):
        return self._sis_finished()

    def finish(self):
        Path(self._dir, "finished").touch()


def _ckpt(job_dir, step, files):
    d = Path(job_dir, "output", "run_dir", "checkpoints", f"checkpoint_{step:06d}", "consolidated")
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        Path(d, f).touch()


def _resolver(run_dir, step):
    return resolve_lora(run_dir, step, "lora")[0].creator


def main():
    with tempfile.TemporaryDirectory() as tmp:
        job_dir = os.path.join(tmp, "train")
        run = _Run(job_dir)
        run_dir = tk.Path("run_dir", creator=run)
        _ckpt(job_dir, 50, ["lora.safetensors", "config.json"])
        _ckpt(job_dir, 100, ["lora.safetensors"])  # save in progress: config.json not yet written

        done = _resolver(run_dir, 50)
        assert done is ResolveOverlayCheckpoint(run_dir=run_dir, overlay_kind="lora", step=50), (
            "the resolver must hash as on the plain run dir (Sisyphus returns the cached job only for the same id)"
        )
        assert done._sis_runnable(), "a complete named checkpoint of a training run must be runnable"
        half = _resolver(run_dir, 100)
        assert not half._sis_runnable(), "weights without config.json = save in progress, must wait"
        assert not _resolver(run_dir, 150)._sis_runnable(), "a step not yet written must wait"
        latest = _resolver(run_dir, None)
        assert not latest._sis_runnable(), "latest must wait for the run to finish"
        print("PASS  while training: a complete named step is runnable; partial, unwritten and latest wait; hash kept")

        _ckpt(job_dir, 100, ["config.json"])
        assert half._sis_runnable(), "once config.json lands the step must be runnable"
        run.finish()
        assert latest._sis_runnable(), "after the run finishes latest is runnable"
        print("PASS  config.json completes a step; the finished run releases latest")


if __name__ == "__main__":
    main()
