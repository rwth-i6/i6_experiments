"""
List of default tools and software to be defined as default independent from hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well

"""
from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.audio import compile_ffmpeg_binary
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.vieting.tools.conda import InstallMinicondaJob, CreateCondaEnvJob

RASR_BINARY_PATH = tk.Path(  # tested with /work/tools22/users/vieting/u20_tf28_sis.sif
    "/work/asr4/vieting/programs/rasr/20230707/rasr/arch/linux-x86_64-standard",
    hash_overwrite="SWITCHBOARD_DEFAULT_RASR_BINARY_PATH",
)

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK
SCTK_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_SCTK_BINARY_PATH"

RETURNN_EXE = tk.Path(  # tested with /work/tools22/users/vieting/u20_tf28_sis.sif
    "/usr/bin/python3",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)

RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="6d2945a85cc95df5349a59541d84f172dd55cc20",
).out_repository
RETURNN_ROOT.hash_overwrite = "SWITCHBOARD_DEFAULT_RETURNN_ROOT"
