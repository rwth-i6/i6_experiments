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

# RASR_BINARY_PATH = None
# RASR_BINARY_PATH = compile_rasr_binaries_i6mode(commit="907eec4f4e36c11153f6ab6b5dd7675116f909f6")  # use tested RASR

##All u16 related
u16_rasr_path_tf1 = tk.Path("/u/raissi/dev/rasr_github/rasr_tf1_conformer", hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF1")
u16_rasr_path_tf2 = tk.Path("/u/raissi/dev/rasr_github/rasr_tf2", hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF2")
U16_RASR_BINARY_PATHS = {'TF1': u16_rasr_path_tf1, 'TF2': u16_rasr_path_tf2}

u16_returnn_launcher_tf2 = tk.Path("/u/raissi/bin/apptainer-launchers/u16/returnn_tf2.3_apptainer_u16_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER_TF1")
U16_RETURNN_LAUNCHERS = {'TF2': u16_returnn_launcher_tf2}

RETURNN_ROOT = tk.Path("/work/tools/users/raissi/returnn_versions/conformer", hash_overwrite="CONFORMER_RETURNN_ROOT")

##All u22 related
u22_rasr_path_onnxtorch = tk.Path("/work/tools22/users/raissi/rasr/rasr_pytorch-onnx", hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TORCHONNX")
u22_RASR_BINARY_PATHS = {'ONNX-TORCH': u22_rasr_path_onnxtorch}




SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK

SCTK_BINARY_PATH.hash_overwrite = "DEFAULT_SCTK_BINARY_PATH"

