from typing import Tuple, Optional, List, Union, Dict
import copy

from i6_core.returnn.training import PtCheckpoint
from i6_core.returnn.forward import ReturnnForwardJob, ReturnnForwardJobV2
from sisyphus import Path, tk

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechCtcAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.recog import _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.ctc.realignment import model_realign_, _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.transducer.realign import generic_returnn_realignment


def ctc_returnn_realignment(
        alias: str,
        config_builder: LibrispeechCtcAttConformerConfigBuilderRF,
        checkpoint: Union[PtCheckpoint, Dict],
        checkpoint_alias: str,
        plot: bool = False,
        batch_size: int = 15_000,
        time_rqmt: int = 1,
        corpus_key: str = "dev-other",
        concurrent: int = 1,
):
  return generic_returnn_realignment(
    alias=alias,
    config_builder=config_builder,
    realign_def=model_realign_,
    forward_step_func=_returnn_v2_forward_step,
    forward_callback=_returnn_v2_get_forward_callback,
    checkpoint=checkpoint,
    checkpoint_alias=checkpoint_alias,
    plot=plot,
    batch_size=batch_size,
    time_rqmt=time_rqmt,
    corpus_key=corpus_key,
    use_multi_proc_dataset=True,
    load_ignore_missing=True,
    concurrent=concurrent,
    dump_scores=False,
  )
