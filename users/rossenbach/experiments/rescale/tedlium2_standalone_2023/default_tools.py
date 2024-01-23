from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob


# python from apptainer
RETURNN_EXE = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
MINI_RETURNN_ROOT = tk.Path("/u/rossenbach/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

from i6_experiments.common.tools.sctk import compile_sctk

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK
SCTK_BINARY_PATH.hash_overwrite = "TEDLIUM2_DEFAULT_SCTK_BINARY_PATH"

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob, CreateBinaryLMJob

kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository
KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries
KENLM_BINARY_PATH.hash_overwrite = "TEDLIUM2_DEFAULT_KENLM_BINARY_PATH"
