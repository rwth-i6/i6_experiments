"""
List of default tools and software to be defined as default independent from hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well.
"""
from sisyphus import tk
from i6_experiments.common.tools.audio import compile_ffmpeg_binary
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_experiments.common.tools.sctk import compile_sctk
from i6_core.tools.git import CloneGitRepositoryJob

PACKAGE = __package__

RASR_BINARY_PATH = tk.Path("/work/asr4/rossenbach/rescale/pytorch_mixed_precision/onnx_extended_rasr/arch/linux-x86_64-standard")
RASR_BINARY_PATH.hash_overwrite = "TEDLIUM2_DEFAULT_RASR_BINARY_PATH"

SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "TEDLIUM2_DEFAULT_SCTK_BINARY_PATH"

SRILM_PATH = tk.Path("/work/tools/users/luescher/srilm-1.7.3/bin/i686-m64/")
SRILM_PATH.hash_overwrite = "TEDLIUM2_DEFAULT_SRILM_PATH"

RETURNN_EXE = tk.Path(
    "/usr/bin/python3",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)

RETURNN_RC_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="d7689b945b2fe781b3c79fbef9d82f018c7b11e8",
).out_repository
RETURNN_RC_ROOT.hash_overwrite = "TEDLIUM2_DEFAULT_RETURNN_RC_ROOT"