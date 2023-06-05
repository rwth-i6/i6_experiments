__all__ = ["compile_tf_graph_from_returnn_config"]

import copy
import typing
from typing import Any, Dict, Union

from i6_core import returnn

from sisyphus import tk


def compile_tf_graph_from_returnn_config(
    returnn_config: Union[returnn.ReturnnConfig, Dict[str, Any]],
    *,
    output_format: str = "meta",
    returnn_root: Union[None, str, tk.Path] = None,
    returnn_python_exe: Union[None, str, tk.Path] = None,
    alias: typing.Optional[str] = None,
):
    if isinstance(returnn_config, returnn.ReturnnConfig):
        tf_returnn_config = copy.copy(returnn_config)
    else:
        tf_returnn_config = returnn.ReturnnConfig(copy.copy(returnn_config))

    conf = returnn.ReturnnConfig(
        {
            "train": {
                "class": "ExternSprintDataset",
                "partitionEpoch": 1,
                "sprintConfigStr": "",
                "sprintTrainerExecPath": None,
            },
            "dev": {
                "class": "ExternSprintDataset",
                "partitionEpoch": 1,
                "sprintConfigStr": "",
                "sprintTrainerExecPath": None,
            },
        }
    )
    tf_returnn_config.update(conf)

    compile_job = returnn.CompileTFGraphJob(
        conf,
        output_format=output_format,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
    )

    if alias:
        compile_job.add_alias(alias)

    return compile_job.out_graph
