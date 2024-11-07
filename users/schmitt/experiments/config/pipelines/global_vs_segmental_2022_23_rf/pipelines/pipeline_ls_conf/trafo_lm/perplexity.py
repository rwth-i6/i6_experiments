from typing import Tuple, Optional, List, Union, Dict

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.training import PtCheckpoint

from sisyphus import Path, tk

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.lm import LibrispeechTrafoLmConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo.forward import _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo.forward import model_forward


def calculate_perplexity(
        config_builder: LibrispeechTrafoLmConfigBuilderRF,
        corpus_keys: Tuple[str, ...],
        checkpoint: PtCheckpoint,
        alias: str,
):

  forward_config = config_builder.get_forward_config(
    opts={
      "dataset_opts": {"corpus_keys": corpus_keys},
      "forward_def": model_forward,
      "forward_step_func": _returnn_v2_forward_step,
      "forward_callback": _returnn_v2_get_forward_callback,
      "batch_size": 5_000,
    }
  )
  forward_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=forward_config,
    returnn_root=RETURNN_CURRENT_ROOT,
    returnn_python_exe=RETURNN_EXE_NEW,
    output_files=["scores.py.gz"],
    mem_rqmt=6,
    time_rqmt=1,
  )
  forward_job.add_alias(f"{alias}/perplexity")
  tk.register_output(forward_job.get_one_alias(), forward_job.out_files["scores.py.gz"])

