from dataclasses import dataclass
from i6_core.tools import CloneGitRepositoryJob
from typing import List, Union, Callable, Optional
import functools
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


default_tools = ToolPaths(
    returnn_root=tk.Path("/u/berger/software/returnn"),
    returnn_python_exe=tk.Path("/usr/bin/python3"),
    # returnn_python_exe=tk.Path("/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3.8"),
    rasr_binary_path=tk.Path("/u/berger/software/rasr_apptainer/arch/linux-x86_64-standard"),
    returnn_common_root=tk.Path("/u/berger/software/returnn_common"),
    # blas_lib=tk.Path(
    #     "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so"
    # ),
)

default_tools_v2 = ToolPaths(
    returnn_root=tk.Path("/u/hwu/repositories/returnn"),
    returnn_python_exe=tk.Path("/usr/bin/python3"),
    rasr_binary_path=tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard"),
    returnn_common_root=tk.Path("/u/berger/repositories/returnn_common"),
)

default_tools_apptek = ToolPaths(
    returnn_root=CloneGitRepositoryJob("https://github.com/rwth-i6/returnn.git").out_repository,
    returnn_python_exe=tk.Path("/usr/bin/python3"),
    rasr_binary_path=tk.Path("/home/sberger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard"),
    returnn_common_root=tk.Path(""),
)
