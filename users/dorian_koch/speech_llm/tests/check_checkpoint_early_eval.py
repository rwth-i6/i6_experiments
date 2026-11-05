"""Guard: a run that exposes its checkpoints as outputs gets its evals started as each checkpoint is written.

Sisyphus releases a job's outputs when the job finishes, so an eval waiting on ``run_dir`` of a 2,000-step RL run got no
s250 result for days. ``RLFinetuneV2`` at ``RL2_POLICY_2026_11_05``+ and ``SpeechFinetune`` at ``POLICY_2026_11_05``+ expose checkpoint ``step`` as its own output
(``checkpoint(step)``) and declares it available once written (``path_available``, as ``ReturnnTrainingJob`` does for
its epochs); ``resolve_lora`` hands that output to ``ResolveOverlayCheckpoint(checkpoint=...)``. Older runs keep the
run-dir route and their hashes.

Drives the real ``CheckpointOutputs.path_available`` / ``checkpoint`` of ``RLFinetuneV2`` and ``SpeechFinetune`` (on an instance built without ``__init__``) through the
real ``resolve_lora``; each acceptance paired with a rejection:
  * complete checkpoint, run still training -> runnable; weights without ``config.json`` -> waiting;
  * a run without checkpoint outputs -> the run-dir resolver, waiting while the run trains;
  * the resolver on a run dir hashes as before (``checkpoint`` excluded from the hash when None).

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_checkpoint_early_eval.py
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

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import resolve_lora  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.speech_inference import ResolveOverlayCheckpoint  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.finetune import SpeechFinetune  # noqa: E402
from speech_llm.full_duplex.rl_spec.job import RLFinetuneV2  # noqa: E402


class _Stub:
    """A training job whose job dir is a temp dir; built with ``object.__new__`` (no graph, no hashing)."""

    tags: set = set()

    @classmethod
    def make(cls, job_dir: str, outputs: bool):
        self = object.__new__(cls)
        self._dir = job_dir
        self.policy = "stub"
        self.checkpoint_outputs = outputs
        self.train = SimpleNamespace(save_every=50, max_steps=1000)
        self.hparams = {"max_steps": 1000, "save_every": 50}
        self.out_rundir = tk.Path("run_dir", creator=self)
        os.makedirs(os.path.join(job_dir, "output", "run_dir"), exist_ok=True)
        return self

    def _sis_path(self, path_type=None, *_a, **_k):
        return self._dir if path_type is None else os.path.join(self._dir, path_type)

    def _sis_id(self):
        return f"stub/{type(self).__name__}/{os.path.basename(self._dir)}"  # distinct ids: Sisyphus caches jobs by id

    def __deepcopy__(self, memo):  # a real Job is shared, not copied, when Sisyphus deep-copies a consumer's args
        return self

    def _sis_finished(self):
        return os.path.exists(os.path.join(self._dir, "finished"))

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class _RL(_Stub, RLFinetuneV2):
    pass


class _SFT(_Stub, SpeechFinetune):
    pass


def _ckpt(job_dir, step, files):
    d = Path(job_dir, "output", "run_dir", "checkpoints", f"checkpoint_{step:06d}", "consolidated")
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        Path(d, f).touch()


def _resolver(run, step):
    return resolve_lora(tk.Path("run_dir", creator=run), step, "lora")[0].creator


def check(Run, tmp):
    print(f"== {Run.__mro__[2].__name__}")
    if True:
        new = Run.make(os.path.join(tmp, "new"), outputs=True)
        _ckpt(new._dir, 50, ["lora.safetensors", "config.json"])
        _ckpt(new._dir, 100, ["lora.safetensors"])  # save in progress: config.json not yet written

        done = _resolver(new, 50)
        assert done.checkpoint is not None and done.run_dir is None, "a run with outputs must use its checkpoint"
        assert done._sis_runnable(), "a complete checkpoint of a training run must be runnable"
        half = _resolver(new, 100)
        assert not half._sis_runnable(), "weights without config.json = save in progress, must wait"
        assert not _resolver(new, 150)._sis_runnable(), "a step not yet written must wait"
        print("PASS  checkpoint outputs: a complete step is runnable while training; partial and unwritten wait")

        _ckpt(new._dir, 100, ["config.json"])
        assert half._sis_runnable(), "once config.json lands the step must be runnable"
        print("PASS  config.json completes a step")

        old = Run.make(os.path.join(tmp, "old"), outputs=False)
        _ckpt(old._dir, 50, ["lora.safetensors", "config.json"])
        legacy = _resolver(old, 50)
        assert legacy.run_dir is not None and legacy.checkpoint is None, "a run without outputs keeps the run-dir route"
        assert not legacy._sis_runnable(), "the run-dir route waits for the run"
        assert legacy is ResolveOverlayCheckpoint(run_dir=tk.Path("run_dir", creator=old), overlay_kind="lora", step=50)
        rd = tk.Path("run_dir", creator=old)
        args = {"run_dir": rd, "overlay_kind": "lora", "step": 50}
        assert ResolveOverlayCheckpoint.hash({**args, "checkpoint": None}) == ResolveOverlayCheckpoint.hash(args), (
            "checkpoint=None must not enter the hash (every existing resolver would move)"
        )
        Path(old._dir, "finished").touch()
        assert legacy._sis_runnable(), "after the run finishes the run-dir resolver is runnable"
        print("PASS  runs without checkpoint outputs keep the run-dir route (waits for the run)")


def check_full_finetune(tmp):
    """A full finetune writes model.safetensors (tmp + rename) and no config.json: its existence completes the step."""
    run = _SFT.make(os.path.join(tmp, "full"), outputs=True)
    _ckpt(run._dir, 50, ["model.safetensors.tmp"])
    r = resolve_lora(tk.Path("run_dir", creator=run), 50, "full")[0].creator
    assert not r._sis_runnable(), "a full checkpoint still in its tmp file must wait"
    _ckpt(run._dir, 50, ["model.safetensors"])
    assert r._sis_runnable(), "the renamed model.safetensors completes a full checkpoint"
    print("PASS  full finetune: the renamed model.safetensors completes the step")


def main():
    for Run in (_RL, _SFT):
        with tempfile.TemporaryDirectory() as tmp:
            check(Run, tmp)
    with tempfile.TemporaryDirectory() as tmp:
        check_full_finetune(tmp)


if __name__ == "__main__":
    main()
