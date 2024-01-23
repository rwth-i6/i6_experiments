"""
List of default tools and software to be defined as default independent from hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well.
"""
from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.sctk import compile_sctk

RASR_BINARY_PATH = tk.Path("/u/hilmes/dev/rasr_onnx_115_new/arch/linux-x86_64-standard")
assert RASR_BINARY_PATH, "Please set a specific RASR_BINARY_PATH before running the pipeline"
RASR_BINARY_PATH.hash_overwrite = "TEDLIUM2_DEFAULT_RASR_BINARY_PATH"

RETURNN_EXE = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    #commit="0963d5b0ad55145a092c1de9bba100c94ee8600c",
    commit="cd3156150fc00f1c6c20a169851bff375823f5d9"
).out_repository
RETURNN_ROOT.hash_overwrite = "TEDLIUM2_DEFAULT_RETURNN_ROOT"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "TEDLIUM2_DEFAULT_SCTK_BINARY_PATH"

SRILM_PATH = tk.Path("/work/tools22/users/luescher/srilm-1.7.3-app-u22/bin/i686-m64")
SRILM_PATH.hash_overwrite = "TEDLIUM2_DEFAULT_SRILM_PATH"
