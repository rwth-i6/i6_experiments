from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.sctk import compile_sctk


RETURNN_EXE = tk.Path("/u/zeineldeen/bin/returnn_tf_ubuntu22_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
RETURNN_CPU_EXE = RETURNN_EXE

RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="b8187f47437a368eda8ad369631521515ee5e209"
).out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_SCTK_BINARY_PATH"
