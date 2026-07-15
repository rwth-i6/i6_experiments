"""RETURNN training helper jobs: best-checkpoint-by-WER selection."""

from typing import Any, Dict

from sisyphus import Job, Task, tk

__all__ = ["GetBestCheckpointByWerJob"]


class GetBestCheckpointByWerJob(Job):
    """Pick the lowest-WER checkpoint among a set of already-scored recogs."""

    def __init__(self, wers: Dict[str, tk.Variable], checkpoints: Dict[str, Any]):
        assert set(wers) == set(checkpoints)
        self.wers = wers
        self.checkpoints = checkpoints
        self.out_summary = self.output_path("wer_summary.json")
        self.out_best_checkpoint = self.output_path("best_checkpoint.pt")
        self.out_best_label = self.output_var("best_label")
        self.out_best_wer = self.output_var("best_wer")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json
        import os

        table = {label: float(var.get()) for label, var in self.wers.items()}
        best_label = min(
            table, key=table.get
        )  # lowest wer wins, ties keep first-seen label
        with open(self.out_summary.get_path(), "w") as f:
            json.dump(
                {
                    "best_label": best_label,
                    "best_wer": table[best_label],
                    "wers": dict(sorted(table.items(), key=lambda kv: kv[1])),
                },
                f,
                indent=2,
            )
        ckpt = self.checkpoints[best_label]
        # accept either a tk.Path or a returnn Checkpoint (latter wraps the path in .path)
        src = ckpt.get_path() if hasattr(ckpt, "get_path") else ckpt.path.get_path()
        dst = self.out_best_checkpoint.get_path()
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        self.out_best_label.set(best_label)
        self.out_best_wer.set(table[best_label])
