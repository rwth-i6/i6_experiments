from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.sctk import compile_sctk


RETURNN_EXE = tk.Path("/u/zeineldeen/bin/returnn_tf_ubuntu22_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
RETURNN_CPU_EXE = RETURNN_EXE

RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="3f7bd3b9c712c6b92ed7ccd8dd925a595cf73716"
).out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_SCTK_BINARY_PATH"

# updated version
# a95905b26459aba2184f9f6db0866a316e3d09a4
RETURNN_ROOT_V2 = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="b1e64c915b934e4e031dfc3755f776d1b4c173c9"
).out_repository
RETURNN_ROOT_V2.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"
