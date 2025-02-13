from sisyphus import tk, Path
from typing import Dict, List, Union, Optional
import copy

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.training import PtCheckpoint

from .config_builder import AEDConfigBuilder
from .recog import _returnn_score_step, _returnn_v2_get_forward_callback
from .model.ctc import ctc_model_rescore


def rescore(
        config_builder: Union[AEDConfigBuilder],
        corpus_key: str,
        checkpoint: Optional[PtCheckpoint],
        returnn_root: Path,
        returnn_python_exe: Path,
        alias: str,
):
  config_opts = {
    "dataset_opts": {
      "corpus_key": corpus_key,
      "use_multi_proc": True,
    },
    "rescore_step_func": _returnn_score_step,
    "forward_callback": _returnn_v2_get_forward_callback,
    "rescore_def": ctc_model_rescore,
  }

  rescore_config = config_builder.get_rescore_config(opts=config_opts)

  analyze_gradients_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=rescore_config,
    returnn_root=returnn_root,
    returnn_python_exe=returnn_python_exe,
    output_files=["output.py.gz"],
    mem_rqmt=12,
    time_rqmt=2,
    device="cpu",
    cpu_rqmt=3,
  )
  analyze_gradients_job.rqmt["sbatch_args"] = ["--exclude", "cn-260"]
  analyze_gradients_job.add_alias(f"{alias}/analysis/rescore/")
  tk.register_output(analyze_gradients_job.get_one_alias(), analyze_gradients_job.out_files["output.py.gz"])

  return analyze_gradients_job
