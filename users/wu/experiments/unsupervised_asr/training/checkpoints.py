"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/emc_train_jobs.py
(``ExtractSubmoduleCheckpointJob``, ``theta_checkpoint``).

Slicing one submodule (theta = ``recognizer.``, phi = ``reverse.``) out of an EMC training checkpoint,
e.g. for a recognizer-only posterior dump or to initialise phi of another arm from a trained one
(``build_train_config(reverse_checkpoint_path=...)``).
"""

from typing import Optional, Tuple

from sisyphus import Job, Task, tk

__all__ = ["ExtractSubmoduleCheckpointJob", "theta_checkpoint"]


class ExtractSubmoduleCheckpointJob(Job):
    """Slice one submodule out of an EMC checkpoint into a standalone checkpoint.

    An EMC training checkpoint holds ``recognizer.*``, ``reverse.*`` and ``agg.*`` in one
    ``state_dict``.  Two consumers want one submodule alone:

    * a posterior dump builds a BARE ``ConvRecognizer`` -> ``prefix = "recognizer."``;
    * an arm that preloads phi -> ``prefix = "reverse."``; the model's loader raises on both missing
      AND unexpected keys, so the sliced file must contain exactly that submodule's keys with the
      prefix stripped.

    The output is ``{"model": state_dict, "epoch": ..., "step": ...}``, which is what both the
    model's loader and RETURNN's own checkpoint loader read.
    """

    def __init__(self, *, checkpoint: "tk.Path", prefix: str, expect_min_keys: int = 1):
        super().__init__()
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.expect_min_keys = int(expect_min_keys)
        self.out_checkpoint = self.output_path("model.pt")
        self.out_stats = self.output_path("extract.stats.txt")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import torch

        state = torch.load(self.checkpoint.get_path(), map_location="cpu", weights_only=False)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        picked = {k[len(self.prefix) :]: v for k, v in sd.items() if k.startswith(self.prefix)}
        assert len(picked) >= self.expect_min_keys, (
            f"{len(picked)} key(s) under prefix {self.prefix!r} in {self.checkpoint.get_path()}; "
            f"the checkpoint holds {sorted({k.split('.')[0] for k in sd})}"
        )
        out = {
            "model": picked,
            "epoch": int(state.get("epoch", 0)) if isinstance(state, dict) else 0,
            "step": int(state.get("step", 0)) if isinstance(state, dict) else 0,
        }
        torch.save(out, self.out_checkpoint.get_path())
        lines = [
            f"source     = {self.checkpoint.get_path()}",
            f"prefix     = {self.prefix!r}",
            f"kept keys  = {len(picked)} of {len(sd)}",
            f"submodules in source = {sorted({k.split('.')[0] for k in sd})}",
            f"epoch = {out['epoch']}  step = {out['step']}",
            f"parameters = {sum(int(v.numel()) for v in picked.values() if hasattr(v, 'numel'))}",
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


def theta_checkpoint(train_job, epoch: int, *, alias: Optional[str] = None) -> Tuple[ExtractSubmoduleCheckpointJob, "object"]:
    """``(job, PtCheckpoint)`` of the recognizer (theta) alone at sub-epoch ``epoch`` of ``train_job``
    (a ``ReturnnTrainingJob``; ``epoch`` must be one of its kept checkpoints), ready for a
    recognizer-only posterior dump.  ``alias`` is added to the extract job when given."""
    from i6_core.returnn.training import PtCheckpoint

    epoch = int(epoch)
    assert epoch in train_job.out_checkpoints, (
        f"sub-epoch {epoch} is not a kept checkpoint of {train_job}: {sorted(train_job.out_checkpoints)}"
    )
    emc_checkpoint = train_job.out_checkpoints[epoch]
    path = emc_checkpoint.path if hasattr(emc_checkpoint, "path") else emc_checkpoint
    job = ExtractSubmoduleCheckpointJob(checkpoint=path, prefix="recognizer.")
    if alias:
        job.add_alias(alias)
    return job, PtCheckpoint(job.out_checkpoint)
