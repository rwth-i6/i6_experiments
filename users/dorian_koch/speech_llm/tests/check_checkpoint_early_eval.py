"""Guard: an eval of a named checkpoint becomes runnable as soon as that checkpoint is complete, while the run trains.

Before 2026-11-05 every ``ResolveOverlayCheckpoint`` waited on the run's ``run_dir`` output, which Sisyphus marks
available only when the whole run finishes: a 2,000-step RL run got no s250 eval for days. Now a named step of a
``lora`` / ``full`` run is ready once its ``consolidated/config.json`` (written after the weights) exists.

Each acceptance is paired with a rejection on the same fixture:
  * named step, checkpoint complete, run still training -> runnable;
  * the same step with only the weights (save in progress) -> waiting;
  * a step not yet written -> waiting;
  * step=None (latest) -> waits for the run, then runnable;
  * personaplex_heads named step -> keeps waiting for the run.

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


def _ready(job) -> bool:
    return job._sis_all_path_available()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        job_dir = os.path.join(tmp, "train")
        run = _Run(job_dir)
        run_dir = tk.Path("run_dir", creator=run)
        _ckpt(job_dir, 50, ["lora.safetensors", "config.json"])
        _ckpt(job_dir, 100, ["lora.safetensors"])  # save in progress: config.json not yet written

        done = ResolveOverlayCheckpoint(run_dir=run_dir, overlay_kind="lora", step=50)
        assert run_dir in done._sis_inputs, "the run dir must stay an input (hash and dependency unchanged)"
        assert _ready(done), "a complete named checkpoint of a training run must be ready"
        half = ResolveOverlayCheckpoint(run_dir=run_dir, overlay_kind="lora", step=100)
        assert not _ready(half), "weights without config.json = save in progress, must wait"
        later = ResolveOverlayCheckpoint(run_dir=run_dir, overlay_kind="lora", step=150)
        assert not _ready(later), "a step not yet written must wait"
        latest = ResolveOverlayCheckpoint(run_dir=run_dir, overlay_kind="lora", step=None)
        assert not _ready(latest), "latest must wait for the run to finish"
        heads = ResolveOverlayCheckpoint(run_dir=run_dir, overlay_kind="personaplex_heads", step=50)
        assert not _ready(heads), "personaplex_heads has no early rule, must wait for the run"
        print("PASS  while training: complete named step ready; partial, unwritten, latest and heads wait")

        _ckpt(job_dir, 100, ["config.json"])
        assert _ready(half), "once config.json lands the step must be ready"
        run.finish()
        assert _ready(latest) and _ready(heads), "after the run finishes everything is ready"
        print("PASS  config.json completes a step; the finished run releases latest and the other layouts")


if __name__ == "__main__":
    main()
