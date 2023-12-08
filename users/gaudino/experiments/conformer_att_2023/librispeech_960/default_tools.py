from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.sctk import compile_sctk


# RETURNN_EXE = tk.Path(
#     "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
# )
# RETURNN_CPU_EXE = tk.Path(
#     "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_generic_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
# )

# RETURNN_EXE = tk.Path(
#     "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
# )
RETURNN_CPU_EXE = tk.Path(
    "/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
)

# RETURNN_ROOT = CloneGitRepositoryJob(
#     "https://github.com/rwth-i6/returnn", commit="a9a669036b8279503da18ee4bdef2ac9498b6504"
# ).out_repository

RETURNN_ROOT = tk.Path("/u/luca.gaudino/setups/2023-02-22--conformer-swb/returnn", hash_overwrite="PROJECT_RETURNN_ROOT")
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_SCTK_BINARY_PATH"
