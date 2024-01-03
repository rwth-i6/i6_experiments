from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.sctk import compile_sctk

# python from apptainer, use these for new pipeline
# NEW_RETURNN_EXE = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

RETURNN_EXE = tk.Path(
    "/u/atanas.gruev/bin/returnn/returnn_pt20_experimental.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)

MINI_RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/JackTemaki/MiniReturnn.git",
    commit="e93df7134f4d01d22e50dbbdfe987f7364da92e2",
).out_repository

MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

# For sclite integration
SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

########################################################################################################################

############# pytorch_conformer_ctc ##############
# RETURNN_EXE = tk.Path(
#     "/u/atanas.gruev/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh",
#     hash_overwrite="GENERIC_RETURNN_LAUNCHER",
# )
RETURNN_PYTORCH_EXE = tk.Path(
    "/u/atanas.gruev/bin/returnn/returnn_pt20_experimental.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)

RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="45fad83c785a45fa4abfeebfed2e731dd96f960c",
).out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"


# ## Local copy of the MiniReturnn (from the Engine fix to access data inside forward_step())
# MINI_RETURNN_ROOT = CloneGitRepositoryJob(
#     "https://github.com/JackTemaki/MiniReturnn.git",
#     commit="d0f2486d9c48d5502fbc1e1b141a301426736463",
# ).out_repository
# MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

RETURNN_COMMON = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="d1fc1c7dc6ae63658e5aa01dc2aad41eb2758573",
    checkout_folder_name="returnn_common",
).out_repository
RETURNN_COMMON.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_COMMON"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

