from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob


RETURNN_EXE = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="DEFAULT_RETURNN_EXE")
RETURNN_CPU_EXE = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_generic_launcher.sh", hash_overwrite="DEFAULT_RETURNN_EXE")

RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="45fad83c785a45fa4abfeebfed2e731dd96f960c").out_repository
RETURNN_ROOT.hash_overwrite = "DEFAULT_RETURNN_ROOT"
