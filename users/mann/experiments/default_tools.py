from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.tools.compile import MakeJob

from i6_experiments.common.setups.rasr import HybridSystem
from i6_experiments.users.jxu.experiments.hybrid.switchboard.default_tools import RASR_BINARY_PATH, RETURNN_ROOT
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode

RETURNN_EXE = tk.Path(
    # "/u/mann/bin/returnn_tf2.3.4_mkl_launcher.sh",
    "/work/tools/asr/python/3.8.0_tf_2.3.4-haswell+cuda10.1+mkl/bin/python",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)

BLAS_LIB = tk.Path(
    "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so",
    hash_overwrite="TF23_MKL_BLAS",
)

# RASR_BINARY_PATH = compile_rasr_binaries_i6mode(
#     commit="a1218e1",
#     configure_options=["--apptainer-setup=2023-05-08_tensorflow-2.8_v1"]
# )
RASR_BINARY_PATH = tk.Path("/work/tools/asr/rasr/20211217_tf23_cuda101_mkl/arch/linux-x86_64-standard")
# RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"
# RASR_BINARY_PATH.hash_overwrite = "DEFAULT_RASR_BINARY_PATH"
# RASR_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_RASR_BINARY_PATH"

def SystemWithDefaultTools():
    hybrid_nn_system = HybridSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        blas_lib=BLAS_LIB,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH,
    )
    return hybrid_nn_system

def SwbSystemWithDefaultTools():
    RASR_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_RASR_BINARY_PATH"
    hybrid_nn_system = HybridSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        blas_lib=BLAS_LIB,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH,
    )
    return hybrid_nn_system