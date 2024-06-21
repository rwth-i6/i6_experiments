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
#keeping same hash as u16_rasr_path_tf2
u16_rasr_path_tf2_barcelona = tk.Path(
    get_rasr_binary_path("/work/tools/users/raissi/rasr/rasr_tf2"), hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF2"
)

u16_rasr_path_ted_common = tk.Path(
    get_rasr_binary_path("/u/raissi/dev/rasr_github/rasr_tf2"),
    hash_overwrite="TEDLIUM2_DEFAULT_RASR_BINARY_PATH",
)

u16_moritz_path = tk.Path(
    "/work/tools/users/raissi/shared/mgunz/rasr_apptainer_tf2.3_u16/arch/linux-x86_64-standard",
    hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF2",
)

U16_RASR_GENERIC_SEQ2SEQ = tk.Path("/work/tools/users/raissi/rasr/generic-seq2seq-dev/arch/linux-x86_64-standard", hash_overwrite="u16")
U16_RASR_BINARY_PATHS = {"TF1": u16_rasr_path_tf2, "TF2": u16_rasr_path_tf2_barcelona, "TED_COMMON": u16_rasr_path_ted_common}
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

u22_rasr_path_tf = tk.Path(
    get_rasr_binary_path("/work/tools22/users/raissi/rasr/rasr_tf2.14"),
    hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF214",
)

u22_rasr_path_tf_test = tk.Path(
    get_rasr_binary_path("/work/tools22/users/raissi/rasr/rasr_tf2.14"),
    hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF2",
)

u22_rasr_path_ted_common = tk.Path(
    get_rasr_binary_path("/work/tools22/users/raissi/rasr/rasr_pytorch-onnx"),
    hash_overwrite="TEDLIUM2_DEFAULT_RASR_BINARY_PATH",
)

u22_RASR_BINARY_PATHS = {
    "ONNX-TORCH": u22_rasr_path_onnxtorch,
    "TED-COMMON": u22_rasr_path_ted_common,
    "TF": u22_rasr_path_tf,
    "TF-TEST": u22_rasr_path_tf_test,
}

u22_returnn_launcher_tf2 = tk.Path(
    "/u/raissi/bin/apptainer-launchers/u22/TF/returnn_tf2.14_apptainer_u22_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER_TF214",
)
U22_RETURNN_LAUNCHERS = {"TF2": u22_returnn_launcher_tf2}

# common
RETURNN_ROOT = tk.Path("/work/tools/users/raissi/returnn_versions/conformer", hash_overwrite="CONFORMER_RETURNN_ROOT")
RETURNN_ROOT_MORITZ = tk.Path(
    "/work/asr3/raissi/shared_workspaces/gunz/2023-05--thesis-baselines-tf2/i6_core/tools/git/CloneGitRepositoryJob.0TxYoqLkxbuC/output/returnn",
    hash_overwrite="CONFORMER_RETURNN_Len_FIX",
)
RETURNN_ROOT_TORCH = tk.Path("/work/tools/users/raissi/returnn_versions/torch", hash_overwrite="TORCH_RETURNN_ROOT")
RETURNN_ROOT_BW_FACTORED = tk.Path("/work/tools/users/raissi/returnn_versions/bw-factored", hash_overwrite="BW_RETURNN_ROOT")

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

u16_default_tools_slow = ToolPaths(
    returnn_root=RETURNN_ROOT,
    returnn_python_exe=U16_RETURNN_LAUNCHERS["TF2"],
    rasr_binary_path=u16_moritz_path,
)

u16_default_tools_returnn_fix = ToolPaths(
    returnn_root=RETURNN_ROOT_MORITZ,
    returnn_python_exe=U16_RETURNN_LAUNCHERS["TF2"],
    rasr_binary_path=U16_RASR_BINARY_PATHS["TED_COMMON"],
)


u16_default_tools_ted = ToolPaths(
    returnn_root=RETURNN_ROOT,
    returnn_python_exe=U16_RETURNN_LAUNCHERS["TF2"],
    rasr_binary_path=U16_RASR_BINARY_PATHS["TED_COMMON"],
)

u16_tools_factored = ToolPaths(
    returnn_root=RETURNN_ROOT_BW_FACTORED,
    returnn_python_exe=U16_RETURNN_LAUNCHERS["TF2"],
    rasr_binary_path=U16_RASR_GENERIC_SEQ2SEQ
)


u22_tools_tf = ToolPaths(
    returnn_root=RETURNN_ROOT_TORCH,
    returnn_python_exe=U22_RETURNN_LAUNCHERS["TF2"],
    rasr_binary_path=u22_RASR_BINARY_PATHS["TF"],
)

u22_tools_tf_test = ToolPaths(
    returnn_root=RETURNN_ROOT,
    returnn_python_exe=U22_RETURNN_LAUNCHERS["TF2"],
    rasr_binary_path=u22_RASR_BINARY_PATHS["TF-TEST"],
)
