from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.sctk import compile_sctk

RETURNN_PYTORCH_ASR_SEARCH_EXE = tk.Path("/usr/bin/python3", hash_overwrite="ASR_SEARCH_FIX")

MINI_RETURNN_ROOT = tk.Path("/u/lukas.rilling/github/MiniReturnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12").copy()  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

NISQA_REPO = CloneGitRepositoryJob("https://github.com/gabrielmittag/NISQA").out_repository.copy()
NISQA_REPO.hash_overwrite = "LIBRISPEECH_DEFAULT_NISQA_REPO"
