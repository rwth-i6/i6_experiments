"""
alignment utils
"""

from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, AlignmentCollection, Alignment
from i6_experiments.users.zeyer.datasets.task import Task


def align(*, task: Task, model: ModelWithCheckpoint) -> AlignmentCollection:
    """alignment"""
    # TODO
    # really just a dummy...
    return AlignmentCollection(alignments={
        name: Alignment(hdf_files=[model.checkpoint.index_path])
        for name, dataset in (
                list(task.eval_datasets.items()) +
                list({"train": task.train_dataset, "dev": task.dev_dataset}.items()))
    })
