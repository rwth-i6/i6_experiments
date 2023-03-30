"""
Tools
"""

from sisyphus import tk, gs
from i6_core.tools.git import CloneGitRepositoryJob


RETURNN_EXE = tk.Path(
    "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
)
RETURNN_CPU_EXE = tk.Path(
    "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_generic_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
)

# RETURNN_ROOT = CloneGitRepositoryJob(
#   "https://github.com/rwth-i6/returnn", commit="cc7a2a559e24a109702f66bc08c7ac9247d09ef2").out_repository
# RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

if getattr(gs, "RETURNN_ROOT_PREFER_GS", False):
    RETURNN_ROOT = getattr(gs, "RETURNN_ROOT")
    if not isinstance(RETURNN_ROOT, tk.Path):
        RETURNN_ROOT = tk.Path(RETURNN_ROOT)
else:
    RETURNN_ROOT = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn", commit="3a67da87c2fd8783c5c2469d72cf1319b5b45837"
    ).out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"
