from dataclasses import dataclass
from typing import List, Union, Callable, Optional
from sisyphus import tk

@dataclass
class ToolPaths:
    returnn_root: Optional[tk.Path] = None
    returnn_python_exe: Optional[tk.Path] = None
    rasr_binary_path: Optional[tk.Path] = None
    returnn_common_root: Optional[tk.Path] = None
    blas_lib: Optional[tk.Path] = None
    rasr_python_exe: Optional[tk.Path] = None

    def __post_init__(self) -> None:
        if self.rasr_python_exe is None:
            self.rasr_python_exe = self.returnn_python_exe


# Adapted from berger/util.py
default_tools = ToolPaths(
    returnn_root=tk.Path("/home/agruev/dependencies/returnn"),
    returnn_python_exe=tk.Path("/usr/bin/python3"),
    rasr_binary_path=tk.Path("/home/agruev/dependencies/rasr_versions/gen_seq2seq_apptainer/arch/linux-x86_64-standard"),
    returnn_common_root=tk.Path("/home/agruev/dependencies/returnn_common"),
)