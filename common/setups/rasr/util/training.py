__all__ = ["ReturnnTrainingJobArgs"]

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

from sisyphus import tk


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


@dataclass()
class ReturnnTrainingData:
    bliss_corpus: tk.Path
    alignments: Dict[int, tk.Path]
    state_tying_file: tk.Path
    allophone_file: tk.Path

    def get(self):
        data = {}

        return data
