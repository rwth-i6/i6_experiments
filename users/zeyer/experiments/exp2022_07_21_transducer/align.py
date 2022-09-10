"""
alignment utils
"""

from .model import ModelWithCheckpoint, AlignmentCollection, Alignment
from .task import Task


def align(*, task: Task, model: ModelWithCheckpoint) -> AlignmentCollection:
    """alignment"""
    # TODO
    # really just a dummy...
    return AlignmentCollection(alignments={
        name: Alignment(hdf_files=[model.checkpoint.index_path])
        for name, dataset in (task.eval_datasets | {"train": task.train_dataset, "dev": task.dev_dataset}).items()
    })
