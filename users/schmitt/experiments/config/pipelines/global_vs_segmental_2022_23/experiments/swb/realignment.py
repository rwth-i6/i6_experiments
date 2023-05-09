from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import SegmentalLabelDefinition, GlobalLabelDefinition, LabelDefinition

# experiments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.recognition import SegmentalReturnnDecodingExperiment, RasrDecodingExperiment, DecodingExperiment, GlobalReturnnDecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.realignment import RasrRealignmentExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.analysis import AlignmentComparer
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb import default_tags_for_analysis
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.search_errors import SegmentalSearchErrorExperiment, GlobalSearchErrorExperiment, SearchErrorExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.search_errors import SegmentalSearchErrorExperiment, GlobalSearchErrorExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_recog_config as get_segmental_recog_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.global_ import get_recog_config as get_global_recog_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_compile_config as get_segmental_compile_config

from i6_core.returnn.training import Checkpoint

from sisyphus import *

from typing import Dict, Optional, List, Tuple, Type
from abc import ABC, abstractmethod


def run_rasr_segmental_realignment(
        dependencies: SegmentalLabelDefinition,
        variant_params: Dict,
        base_alias: str,
        checkpoint: Checkpoint,
        corpus_key: str,
        length_scale: float,
        length_norm: bool = False,
):
  """
  Set rqmts and maximum segment length here, depending on the corpus and whether silence is used. Realignment with
  explicit silence takes longer because, in addition to blank, silence can be output at every position.
  """
  if corpus_key == "train":
    if dependencies.model_hyperparameters.sil_idx is None:
      time_rqmt = 15
      mem_rqmt = 12
      max_segment_len = 20
      concurrent = 8
      label_pruning = 12.0
    else:
      time_rqmt = 30
      mem_rqmt = 12
      max_segment_len = 12
      concurrent = 8
      label_pruning = 5.0
  else:
    if dependencies.model_hyperparameters.sil_idx is None:
      time_rqmt = 2
      mem_rqmt = 5
      max_segment_len = 20
      concurrent = 1
      label_pruning = 12.0
    else:
      time_rqmt = 3
      mem_rqmt = 6
      max_segment_len = 12
      concurrent = 1
      label_pruning = 5.0

  base_alias = "%s/rasr_realign_length_scale%0.1f/%s" % (base_alias, length_scale, corpus_key)

  realignment = RasrRealignmentExperiment(
    dependencies=dependencies,
    variant_params=variant_params,
    base_alias=base_alias,
    checkpoint=checkpoint,
    corpus_key=corpus_key,
    length_norm=length_norm,
    max_segment_len=max_segment_len,
    concurrent=concurrent,
    length_scale=length_scale,
    time_rqmt=time_rqmt,
    mem_rqmt=mem_rqmt,
    label_pruning=label_pruning
  ).run()

  if corpus_key == "cv":
    AlignmentComparer(
      hdf_align_path1=realignment,
      blank_idx1=dependencies.model_hyperparameters.blank_idx,
      name1="realignment",
      vocab_path1=dependencies.vocab_path,
      hdf_align_path2=realignment,
      blank_idx2=dependencies.model_hyperparameters.blank_idx,
      name2="realignment",
      vocab_path2=dependencies.vocab_path,
      seq_tags=default_tags_for_analysis,
      corpus_key="cv",
      base_alias=base_alias).run()

  return realignment
