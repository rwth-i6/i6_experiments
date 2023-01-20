__all__ = ["compile_tf_graph_from_returnn_config"]

import copy
from typing import Any, Dict, Union

from i6_core import returnn

from sisyphus import tk


def compile_tf_graph_from_returnn_config(
    returnn_config: Union[returnn.ReturnnConfig, Dict[str, Any]],
    *,
    output_format: str = "meta",
    python_prolog: Union[None, str, tuple, list, dict] = None,
    python_epilog: Union[None, str, tuple, list, dict] = None,
    returnn_root: Union[None, str, tk.Path] = None,
    returnn_python_exe: Union[None, str, tk.Path] = None,
):
    if isinstance(returnn_config, returnn.ReturnnConfig):
        tf_returnn_config = copy.copy(returnn_config.config)
    else:
        tf_returnn_config = copy.copy(returnn_config)

    tf_returnn_config["train"] = {
        "class": "ExternSprintDataset",
        "partitionEpoch": 1,
        "sprintConfigStr": "",
        "sprintTrainerExecPath": None,
    }

    tf_returnn_config["dev"] = {
        "class": "ExternSprintDataset",
        "partitionEpoch": 1,
        "sprintConfigStr": "",
        "sprintTrainerExecPath": None,
    }

    conf = returnn.ReturnnConfig(
        tf_returnn_config,
        python_prolog=python_prolog
        if python_prolog is not None
        else returnn_config.python_prolog,
        python_epilog=python_epilog
        if python_epilog is not None
        else returnn_config.python_epilog,
    )
    compile_job = returnn.CompileTFGraphJob(
        conf,
        output_format=output_format,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
    )

    return compile_job.out_graph
