from dataclasses import dataclass

from sisyphus import tk

@dataclass
class HDFBuilderArgs:
    returnn_root: tk.Path
    returnn_python_exe: tk.Path
    dc_detection: bool = False
    single_hdf: bool = False
