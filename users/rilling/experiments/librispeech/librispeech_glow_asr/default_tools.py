from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob


# python from apptainer
RETURNN_EXE = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
MINI_RETURNN_ROOT = tk.Path("/u/lukas.rilling/github/MiniReturnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")

from i6_experiments.common.tools.sctk import compile_sctk

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

