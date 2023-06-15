from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition

# experiments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.recognition import DecodingExperiment


from i6_core.returnn.training import Checkpoint

from sisyphus import *

from typing import Dict
from abc import ABC, abstractmethod


class LibrispeechSearchPipeline(ABC):
  """
  """
  def __init__(
          self,
          dependencies: LabelDefinition,
          variant_params: Dict,
          base_alias: str,
          checkpoint: Checkpoint,
          corpus_key: str):

    self.corpus_key = corpus_key
    self.dependencies = dependencies
    self.variant_params = variant_params
    self.base_alias = base_alias
    self.checkpoint = checkpoint

  @abstractmethod
  def get_decoding_experiment(self, corpus_key) -> DecodingExperiment:
    pass

  @property
  @abstractmethod
  def alias(self) -> str:
    pass

  def decode(self, corpus_key: str):
    decoding_exp = self.get_decoding_experiment(corpus_key=corpus_key)
    decoding_exp.run_eval(use_hub5_score_job=False)
    return decoding_exp.best_traces

  def run(self):
    self.decode(corpus_key=self.corpus_key)

