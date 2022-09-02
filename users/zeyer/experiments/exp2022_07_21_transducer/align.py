"""
alignment utils
"""

from .model import ModelWithCheckpoint, AlignmentCollection
from .task import Task


def align(*, task: Task, model: ModelWithCheckpoint) -> AlignmentCollection:
    # TODO
    pass
