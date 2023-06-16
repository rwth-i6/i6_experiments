from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode



RETURNN_EXE = tk.Path("/u/lukas.rilling/bin/returnn/returnn_tf_dynamic_version_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
RETURNN_PYTORCH_EXE = tk.Path("/u/lukas.rilling/bin/returnn/returnn_pt20_experimental.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

# outdated version using DataLoader v1 and the Tensor fix for RC networks
# RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="d98a6c606d2c007e2a6771684e77a7650bb3fad6").out_repository
# RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

MINI_RETURNN_ROOT = tk.Path("/u/lukas.rilling/github/MiniReturnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")

RETURNN_COMMON = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn_common", commit="d1fc1c7dc6ae63658e5aa01dc2aad41eb2758573", checkout_folder_name="returnn_common").out_repository
RETURNN_COMMON.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_COMMON"
