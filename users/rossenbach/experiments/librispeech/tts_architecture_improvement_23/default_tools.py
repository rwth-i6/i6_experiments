from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_experiments.common.tools.sctk import compile_sctk

RETURNN_EXE = tk.Path("/u/rossenbach/bin/returnn/returnn_tf_dynamic_version_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
RETURNN_PYTORCH_EXE = tk.Path("/u/rossenbach/bin/returnn/returnn_pt20_experimental.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

# outdated version using DataLoader v1 and the Tensor fix for RC networks
RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="d98a6c606d2c007e2a6771684e77a7650bb3fad6").out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

MINI_RETURNN_ROOT = tk.Path("/u/rossenbach/src/NoReturnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")

RETURNN_COMMON = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn_common", commit="d1fc1c7dc6ae63658e5aa01dc2aad41eb2758573", checkout_folder_name="returnn_common").out_repository
RETURNN_COMMON.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_COMMON"



# Does not work because of https://github.com/rwth-i6/rasr/issues/65
# RASR_GMM_BINARY_PATH = compile_rasr_binaries_i6mode(
#     commit="c819ba3f0689950722e6fcbbde8894dca7823604",
#     configure_options=[
#         "--apptainer-setup=2023-08-09_tensorflow-2.8_onnx-1.15_v1",
#         "--set-march=haswell",
#         "--enable-module=CORE_CACHE_MANAGER",
#         "--disable-module=TENSORFLOW",
#         "--disable-module=LM_TFRNN",
#         "--disable-module=ONNX",
#     ]
# )
RASR_GMM_BINARY_PATH = tk.Path("/work/asr4/rossenbach/src/rasr_u22_gmm_old_cpu/arch/linux-x86_64-standard")
RASR_GMM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"
