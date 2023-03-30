from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.sctk import compile_sctk


RETURNN_EXE = tk.Path(
    "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
)
RETURNN_CPU_EXE = tk.Path(
    "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_generic_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
)

RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="b96842963caa1b218d1d7cb76d7ed81867079bdf"
).out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_SCTK_BINARY_PATH"

NICK_LOGMEL_RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", branch="nick-fminfmax-mel"
).out_repository
