from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_core.lm.kenlm import CompileKenLMJob


RETURNN_EXE = tk.Path(
    "/u/lukas.rilling/bin/returnn/returnn_tf_dynamic_version_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
)
RETURNN_PYTORCH_EXE = tk.Path(
    "/u/lukas.rilling/bin/returnn/returnn_pt20_experimental.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
)

# outdated version using DataLoader v1 and the Tensor fix for RC networks
# RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="d98a6c606d2c007e2a6771684e77a7650bb3fad6").out_repository
# RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

MINI_RETURNN_ROOT = tk.Path("/u/lukas.rilling/github/MiniReturnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")

RETURNN_COMMON = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="d1fc1c7dc6ae63658e5aa01dc2aad41eb2758573",
    checkout_folder_name="returnn_common",
).out_repository
RETURNN_COMMON.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_COMMON"


from i6_experiments.common.tools.sctk import compile_sctk

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository
KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries
KENLM_BINARY_PATH.hash_overwrite = "TEDLIUM2_DEFAULT_KENLM_BINARY_PATH"

NISQA_REPO = CloneGitRepositoryJob("https://github.com/gabrielmittag/NISQA").out_repository.copy()
NISQA_REPO.hash_overwrite = "LIBRISPEECH_DEFAULT_NISQA_REPO"
