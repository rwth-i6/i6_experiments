__all__ = ["ReturnnTrainingJobArgs", "EpochPartitioning", "ReturnnRasrTrainingArgs"]

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, TypedDict, Union

from sisyphus import tk

import i6_core.rasr as rasr


@dataclass()
class ReturnnTrainingJobArgs:
    num_epochs: int
    log_verbosity: int = field(default=4)
    device: str = field(default="gpu")
    save_interval: int = field(default=1)
    keep_epochs: Optional[Union[List[int], Set[int]]] = None
    time_rqmt: float = field(default=168)
    mem_rqmt: float = field(default=14)
    cpu_rqmt: int = field(default=4)
    horovod_num_processes: Optional[int] = None
    multi_node_slots: Optional[int] = None
    returnn_python_exe: Optional[tk.Path] = None
    returnn_root: Optional[tk.Path] = None


class EpochPartitioning(TypedDict):
    dev: int
    train: int


@dataclass(frozen=True)
class ReturnnRasrTrainingArgs:
    """
    Options for writing a RASR training config. See `ReturnnRasrTrainingJob`.
    Most of them may be disregarded, i.e. the defaults can be left untouched.

    :param partition_epochs: if >1, split the full dataset into multiple sub-epochs
    :param num_classes: number of classes
    :param disregarded_classes: path to file with list of disregarded classes
    :param class_label_file: path to file with class labels
    :param buffer_size: buffer size for data loading
    :param extra_rasr_config: extra RASR config
    :param extra_rasr_post_config: extra RASR post config
    :param use_python_control: whether to use python control, usually True
    """

    partition_epochs: Optional[int] = None
    num_classes: Optional[int] = None
    disregarded_classes: Optional[tk.Path] = None
    class_label_file: Optional[tk.Path] = None
    buffer_size: int = 200 * 1024
    extra_rasr_config: Optional[rasr.RasrConfig] = None
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None
    use_python_control: bool = True
    cpu_rqmt: Optional[int] = 2
    device: Optional[str] = "gpu"
    horovod_num_processes: Optional[int] = None
    keep_epochs: Optional[bool] = None
    log_verbosity: Optional[int] = 3
    mem_rqmt: Optional[int] = 12
    num_epochs: int = 1
    save_interval: Optional[int] = 1
    time_rqmt: Optional[int] = 168
