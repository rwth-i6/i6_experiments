"""
List of default tools and software to be defined as default independent of hashing by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well
"""
import os.path

from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.sctk import compile_sctk

RASR_BINARY_PATH = tk.Path(  # tested with /work/tools22/users/vieting/u20_tf28_sis.sif
    "/work/asr4/vieting/programs/rasr/20230707/rasr/arch/linux-x86_64-standard",
    hash_overwrite="SWITCHBOARD_DEFAULT_RASR_BINARY_PATH",
)

RASR_BINARY_PATH_PRECISION = tk.Path(  # tested with /work/tools22/users/vieting/u20_tf28_sis.sif
    "/work/asr4/vieting/programs/rasr/20240209/rasr/arch/linux-x86_64-standard",
    hash_overwrite="SWITCHBOARD_PRECISION_RASR_BINARY_PATH",  # adds fix for segment start and end time precision
)

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_SCTK_BINARY_PATH"

RETURNN_EXE = tk.Path(  # tested with /work/tools22/users/vieting/u20_tf28_sis.sif
    "/usr/bin/python3",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)

patch_file = os.path.join(os.path.dirname(__file__), "helpers/transducer_returnn_fixes.patch")
with open(patch_file) as f:
    patch = f.read()
RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="780e4254d726b02848cb57558f34be66a12ce042",
    patches=[patch],
    checkout_folder_name="returnn",
).out_repository
RETURNN_ROOT.hash_overwrite = "SWITCHBOARD_TRANSDUCER_RETURNN_ROOT"
