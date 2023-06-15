from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition

# experiments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.recognition import DecodingExperiment, GlobalReturnnDecodingExperiment

from i6_core.returnn.training import Checkpoint

from sisyphus import *

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.librispeech.recognition.base import LibrispeechSearchPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.config_builder.config_builder import GlobalConfigBuilder

from typing import Dict, Optional, List


class ReturnnLabelSyncBeamSearchPipeline(LibrispeechSearchPipeline):
  def __init__(
          self,
          dependencies: GlobalLabelDefinition,
          beam_size: int,
          **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)
    self.dependencies = dependencies
    self.beam_size = beam_size

  @property
  def alias(self) -> str:
    return "%s/returnn_label_wise_beam_search" % self.base_alias

  def get_decoding_experiment(self, corpus_key) -> DecodingExperiment:
    returnn_config = GlobalConfigBuilder(
      dependencies=self.dependencies,
      variant_params=self.variant_params
    ).get_recog_config(search_corpus_key=corpus_key)

    return GlobalReturnnDecodingExperiment(
      dependencies=self.dependencies,
      returnn_config=returnn_config,
      variant_params=self.variant_params,
      checkpoint=self.checkpoint,
      dump_best_traces=False,
      corpus_key=corpus_key,
      base_alias=self.alias)
