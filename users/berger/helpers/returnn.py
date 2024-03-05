from i6_core import returnn
from sisyphus import tk
from i6_experiments.users.berger.util import ToolPaths


def get_native_lstm_op(tool_paths: ToolPaths) -> tk.Path:
    # DO NOT USE BLAS ON I6, THIS WILL SLOW DOWN RECOGNITION ON OPTERON MACHNIES BY FACTOR 4
    compile_job = returnn.CompileNativeOpJob(
        "NativeLstm2",
        returnn_root=tool_paths.returnn_root,
        returnn_python_exe=tool_paths.returnn_python_exe,
        blas_lib=tool_paths.blas_lib,
    )

    return compile_job.out_op
