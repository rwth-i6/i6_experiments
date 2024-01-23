"""
List of default tools and software to be defined as default independent of hashing by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well
"""
from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob

RASR_BINARY_PATH_APPTAINER = tk.Path(  # tested with /work/asr4/berger/apptainer/images/i6_tf_2_12_mkl.sif
    "/work/asr4/vieting/programs/rasr/20230707/rasr/arch/linux-x86_64-standard",
    hash_overwrite="LIBRISPEECH_APPTAINER_RASR_BINARY_PATH",
)

RETURNN_EXE = tk.Path(
    # "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh",
    "/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3.8",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)
RETURNN_CPU_EXE = tk.Path(
    "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_generic_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)
RETURNN_EXE_APPTAINER = tk.Path(  # tested with /work/asr4/hilmes/apptainer/u22_tf+torch_new.sif
    "/usr/bin/python3",
    hash_overwrite="APPTAINER_RETURNN_LAUNCHER",
)

RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="9dfeb9a8492d6ca239a20c6a38bed7e673402984",
).out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"
