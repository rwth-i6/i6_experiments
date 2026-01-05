from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob


RETURNN_EXE = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="4a762b55e8d8d6ca93602c29dd4e40572c98a971").out_repository
RETURNN_ROOT.hash_overwrite = "DOMAIN_TESTING_2024_TF_RETURNN_ROOT"
