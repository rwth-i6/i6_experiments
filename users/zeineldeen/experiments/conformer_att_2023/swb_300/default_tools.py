from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob


RETURNN_EXE = tk.Path("/u/zeineldeen/bin/returnn_tf_ubuntu22_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
RETURNN_CPU_EXE = RETURNN_EXE

# old: b8187f47437a368eda8ad369631521515ee5e209
RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="3f7bd3b9c712c6b92ed7ccd8dd925a595cf73716"
).out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"
