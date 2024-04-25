import copy
from typing import Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_CURRENT_ROOT, RETURNN_EXE, RETURNN_EXE_NEW, RETURNN_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT
from i6_experiments.users.schmitt.alignment.alignment import AlignmentStatisticsJob
from i6_experiments.users.schmitt.alignment.att_weights import AttentionWeightStatisticsJob
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob, PlotAttentionWeightsJobV2

from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.text.processing import WriteToTextFileJob
from i6_core.returnn.training import Checkpoint

from sisyphus import tk


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline_rf():
    for train_alias, checkpoint in train.train_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(1,),
      time_rqmt=4,
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
      )
