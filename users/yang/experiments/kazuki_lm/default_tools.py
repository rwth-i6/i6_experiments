from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob


RETURNN_EXE = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

# RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="45fad83c785a45fa4abfeebfed2e731dd96f960c").out_repository

# run with LM vocab fix
RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="9e7b7f24b090f4a4909c9be3e276194f026a6932").out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"


