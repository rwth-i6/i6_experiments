"""
List of default tools and software to be defined as default independent from hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well

"""
from sisyphus import tk
from i6_experiments.common.tools.audio import compile_ffmpeg_binary
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_experiments.common.tools.sctk import compile_sctk

import os
from typing import Optional
from dataclasses import dataclass

def get_rasr_binary_path(rasr_path):
    return os.path.join(rasr_path, "arch", "linux-x86_64-standard")

RASR_BINARY_PATH = tk.Path(
    "/u/noureldin.bayoumi/dev/rasr_master/arch/linux-x86_64-standard", hash_overwrite="local-rasr-bayoumi"
)

RASR_BINARY_PATH_SIMON = tk.Path(
    "/u/berger/repositories/rasr_versions/gen_seq2seq_apptainer/arch/linux-x86_64-standard", hash_overwrite="local-rasr-simon"
)


# RASR_BINARY_PATH = compile_rasr_binaries_i6mode(commit="907eec4f4e36c11153f6ab6b5dd7675116f909f6")  # use tested RASR
# RASR_BINARY_PATH = compile_rasr_binaries_i6mode()  #  use most recent RASR

RETURNN_EXE_PATH = tk.Path(
    "/u/noureldin.bayoumi/bin/opt/launchers/u22/returnn_tf2.8_apptainer_u22_launcher.sh", hash_overwrite="default"
)

RETURNN_ROOT = tk.Path("/u/noureldin.bayoumi/dev/returnn", hash_overwrite="returnn-v-oct-10")
RETURNN_ROOT_OLD = tk.Path("/u/noureldin.bayoumi/dev/old_i6_recipes/returnn_327f43cf", hash_overwrite="returnn-v-oct-10")
RETURNN_ROOT_SIMON = tk.Path("/u/noureldin.bayoumi/dev/returnn_simon", hash_overwrite="returnn-v-oct-10")

RETURNN_COMMON = tk.Path("/u/berger/software/returnn_common", hash_overwrite="returnn-common-berger")

u16_rasr_path_tf2 = tk.Path(
    get_rasr_binary_path("/work/asr3/raissi/shared_workspaces/bayoumi/dev/rasr_tf2"), hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF2"
)

hwu = tk.Path(
            get_rasr_binary_path("/work/asr3/raissi/shared_workspaces/bayoumi/dev/haotian"), hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF2"
            )

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

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


tools = ToolPaths(
    returnn_root=RETURNN_ROOT,
    returnn_python_exe=RETURNN_EXE_PATH,
    rasr_binary_path=RASR_BINARY_PATH,
    returnn_common_root=RETURNN_COMMON,
)


simon_rasr_tools = ToolPaths(
    returnn_root=RETURNN_ROOT,
    returnn_python_exe=RETURNN_EXE_PATH,
    rasr_binary_path=RASR_BINARY_PATH_SIMON,
    returnn_common_root=RETURNN_COMMON,
)

returnn_old_tools = ToolPaths(
    returnn_root=RETURNN_ROOT_OLD,
    returnn_python_exe=RETURNN_EXE_PATH,
    rasr_binary_path=RASR_BINARY_PATH,
    returnn_common_root=RETURNN_COMMON,
)

simon_returnn_tools = ToolPaths(
    returnn_root=RETURNN_ROOT_SIMON,
    returnn_python_exe=RETURNN_EXE_PATH,
    rasr_binary_path=RASR_BINARY_PATH,
    returnn_common_root=RETURNN_COMMON,
)

u16_default_tools = ToolPaths(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE_PATH,
        rasr_binary_path=RASR_BINARY_PATH,
        returnn_common_root=RETURNN_COMMON,
)


