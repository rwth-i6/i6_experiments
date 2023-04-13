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


class ReturnnRasrTrainingArgs(TypedDict):
    buffer_size: Optional[int]
    class_label_file: Optional[tk.Path]
    cpu_rqmt: Optional[int]
    device: Optional[str]
    disregarded_classes: Optional[Any]
    extra_rasr_config: Optional[rasr.RasrConfig]
    extra_rasr_post_config: Optional[rasr.RasrConfig]
    horovod_num_processes: Optional[int]
    keep_epochs: Optional[bool]
    log_verbosity: Optional[int]
    mem_rqmt: Optional[int]
    num_classes: int
    num_epochs: int
    partition_epochs: Optional[EpochPartitioning]
    save_interval: Optional[int]
    time_rqmt: Optional[int]
    use_python_control: Optional[bool]
