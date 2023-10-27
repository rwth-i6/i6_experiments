from i6_core import returnn
from sisyphus import tk
from i6_experiments.users.berger.util import ToolPaths
from i6_core.returnn import ReturnnConfig
from returnn_common.nn.naming import ReturnnDimTagsProxy

def get_native_lstm_op(tool_paths: ToolPaths) -> tk.Path:
    # DO NOT USE BLAS ON I6, THIS WILL SLOW DOWN RECOGNITION ON OPTERON MACHNIES BY FACTOR 4
    compile_job = returnn.CompileNativeOpJob(
        "NativeLstm2",
        returnn_root=tool_paths.returnn_root,
        returnn_python_exe=tool_paths.returnn_python_exe,
        blas_lib=tool_paths.blas_lib,
    )

    return compile_job.out_op

def serialize_dim_tags(config: ReturnnConfig) -> ReturnnConfig:
    """
    Serialize dim tags in a given RETURNN config.
    Copied from i6_private/users/vieting/helpers/returnn.py
    """
    dim_tags_proxy = ReturnnDimTagsProxy()
    config_serialized = dim_tags_proxy.collect_dim_tags_and_transform_config(
        config.config
    )
    if dim_tags_proxy.py_code_str():
        config.config["network"] = _replace_proxies_by_code_wrappers(
            config_serialized["network"]
        )
        config.config["extern_data"] = _replace_proxies_by_code_wrappers(
            config_serialized["extern_data"]
        )
        python_prolog_ext = (
            "from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim\n\n"
            + dim_tags_proxy.py_code_str()
        )
        if config.python_prolog is None:
            config.python_prolog = python_prolog_ext
        elif isinstance(config.python_prolog, str):
            config.python_prolog += "\n" + python_prolog_ext
        elif isinstance(config.python_prolog, dict):
            if "dim_tags" in config.python_prolog:
                config.python_prolog["dim_tags"] += "\n" + python_prolog_ext
            else:
                config.python_prolog["dim_tags"] = python_prolog_ext
        elif isinstance(config.python_prolog, list):
            config.python_prolog.append(python_prolog_ext)
        else:
            raise NotImplementedError
    return config

