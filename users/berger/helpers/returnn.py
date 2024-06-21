from typing import Any, Optional
from i6_core import returnn
from i6_core.returnn.config import CodeWrapper
from sisyphus import tk
from i6_core.returnn import ReturnnConfig


def get_native_lstm_op(
    returnn_root: tk.Path, returnn_python_exe: tk.Path, blas_lib: Optional[tk.Path] = None
) -> tk.Path:
    # DO NOT USE BLAS ON I6, THIS WILL SLOW DOWN RECOGNITION ON OPTERON MACHNIES BY FACTOR 4
    compile_job = returnn.CompileNativeOpJob(
        "NativeLstm2",
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        blas_lib=blas_lib,
    )

    return compile_job.out_op


def _replace_proxies_by_code_wrappers(obj: Any) -> Any:
    """
    A ReturnnDimTagsProxy.DimRefProxy can currently not be hashed and sisyphus' extract_paths() also does not work,
    because the parent attribute contains a set which again contains the original object which leads to recursion errors.
    We could fix this in ReturnnDimTagsProxy.DimRefProxy, but for now just replace them with a CodeWrapper.
    """
    from returnn_common.nn.naming import ReturnnDimTagsProxy

    if isinstance(obj, (ReturnnDimTagsProxy.SetProxy, ReturnnDimTagsProxy.DimRefProxy)):
        return CodeWrapper(str(obj))
    elif isinstance(obj, dict):
        return {k: _replace_proxies_by_code_wrappers(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)([_replace_proxies_by_code_wrappers(x) for x in obj])
    return obj


def serialize_dim_tags(config: ReturnnConfig) -> ReturnnConfig:
    """
    Serialize dim tags in a given RETURNN config.
    Copied from i6_private/users/vieting/helpers/returnn.py
    """
    from returnn_common.nn.naming import ReturnnDimTagsProxy

    dim_tags_proxy = ReturnnDimTagsProxy()
    config_serialized = dim_tags_proxy.collect_dim_tags_and_transform_config(config.config)
    if dim_tags_proxy.py_code_str():
        if "network" in config.config:
            config.config["network"] = _replace_proxies_by_code_wrappers(config_serialized["network"])
        if "extern_data" in config.config:
            config.config["extern_data"] = _replace_proxies_by_code_wrappers(config_serialized["extern_data"])
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
