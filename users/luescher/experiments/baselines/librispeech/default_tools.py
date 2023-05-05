"""
List of default tools and software to be defined as default independent of hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well

"""
from sisyphus import tk
from i6_experiments.common.tools.sctk import compile_sctk

RASR_BINARY_PATH = tk.Path(
    "/work/tools22/users/luescher/rasr/apptainer_ubuntu22.04_tf2.8_230504/arch/linux-x86_64-standard",
    hash_overwrite="LIBRISPEECH_APPTAINER_UBUNTU22_RASR_BINARY_PATH",
)

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"
