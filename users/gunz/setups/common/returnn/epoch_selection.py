import glob
import os.path
from typing import Dict

from sisyphus import tk

from i6_core import returnn


def _get_epoch(path: str) -> int:
    _, ep, *_ = os.path.basename(path).split(".")
    return int(ep)


async def find_surviving_epochs(job: returnn.ReturnnTrainingJob) -> Dict[int, returnn.Checkpoint]:
    await tk.async_run(job.out_checkpoints)

    surviving_index_files = glob.glob(job.out_model_dir.join_right("epoch.*.index"))
    surviving_checkpoints = {_get_epoch(p): returnn.Checkpoint(p) for p in surviving_index_files}

    return surviving_checkpoints
