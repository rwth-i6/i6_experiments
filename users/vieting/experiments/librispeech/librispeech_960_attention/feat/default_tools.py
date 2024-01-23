from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.sctk import compile_sctk


RETURNN_EXE = tk.Path(
    "/u/rossenbach/bin/returnn/returnn_tf_dynamic_version_mkl_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)
RETURNN_CPU_EXE = RETURNN_EXE

RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="8f5d2a51ddc3ca862bf5cadcf19069159d59607d"
).out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_SCTK_BINARY_PATH"
