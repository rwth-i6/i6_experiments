"""
List of default tools and software to be defined as default independent from hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well

"""
import os

from dataclasses import dataclass
from typing import Optional

from sisyphus import tk
from i6_experiments.common.tools.audio import compile_ffmpeg_binary
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_experiments.common.tools.sctk import compile_sctk


def get_rasr_binary_path(rasr_path):
    return os.path.join(rasr_path, "arch", "linux-x86_64-standard")


u16_rasr_path_tf1 = tk.Path(
    get_rasr_binary_path("/u/raissi/dev/rasr_github/rasr_tf1_conformer"),
    hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF1",
)
u16_rasr_path_tf2 = tk.Path(
    get_rasr_binary_path("/u/raissi/dev/rasr_github/rasr_tf2"), hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF2"
)
U16_RASR_BINARY_PATHS = {"TF1": u16_rasr_path_tf2, "TF2": u16_rasr_path_tf2}
u16_returnn_launcher_tf2 = tk.Path(
    "/u/raissi/bin/apptainer-launchers/u16/returnn_tf2.3_apptainer_u16_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER_TF2",
)
U16_RETURNN_LAUNCHERS = {"TF2": u16_returnn_launcher_tf2}

##All u22 related
u22_rasr_path_onnxtorch = tk.Path(
    get_rasr_binary_path("/work/tools22/users/raissi/rasr/rasr_pytorch-onnx"),
    hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TORCHONNX",
)
u22_RASR_BINARY_PATHS = {"ONNX-TORCH": u22_rasr_path_onnxtorch}

#common
RETURNN_ROOT = tk.Path("/work/tools/users/raissi/returnn_versions/conformer", hash_overwrite="CONFORMER_RETURNN_ROOT")
RETURNN_ROOT_TORCH = tk.Path("/work/tools/users/raissi/returnn_versions/torch", hash_overwrite="TORCH_RETURNN_ROOT")

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "DEFAULT_SCTK_BINARY_PATH"

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


u16_default_tools = ToolPaths(
    returnn_root=RETURNN_ROOT,
    returnn_python_exe=U16_RETURNN_LAUNCHERS["TF2"],
    rasr_binary_path=U16_RASR_BINARY_PATHS["TF2"],
)
