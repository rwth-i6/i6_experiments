__all__ = ["ReturnnTrainingJobArgs", "ReturnnRawAlignmentHdfTrainingData", "AllowedReturnnTrainingData"]

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

from sisyphus import tk

from i6_core.returnn.hdf import BlissToPcmHDFJob, RasrAlignmentDumpHDFJob


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
class ReturnnRawAlignmentHdfTrainingData:
    bliss_corpus: tk.Path
    alignment_caches: List[tk.Path]
    state_tying_file: tk.Path
    allophone_file: tk.Path
    returnn_root: tk.Path
    seq_ordering: str

    def get(self):
        raw_hdf_path = BlissToPcmHDFJob(
            bliss_corpus=self.bliss_corpus,
            returnn_root=self.returnn_root,
        ).out_hdf
        alignment_hdf_path = RasrAlignmentDumpHDFJob(
            alignment_caches=self.alignment_caches,
            allophone_file=self.allophone_file,
            state_tying_file=self.state_tying_file,
            returnn_root=self.returnn_root,
        ).out_hdf_files

        data = {
            "class": "MetaDataset",
            "data_map": {"classes": ("alignments", "data"), "data": ("features", "data")},
            "datasets": {
                "alignments": {
                    "class": "HDFDataset",
                    "files": alignment_hdf_path,
                    "seq_ordering": self.seq_ordering,
                },
                "features": {
                    "class": "HDFDataset",
                    "files": [raw_hdf_path],
                },
            },
            "seq_order_control_dataset": "alignments",
        }

        return data


AllowedReturnnTrainingData = Union[Dict, ReturnnRawAlignmentHdfTrainingData]
