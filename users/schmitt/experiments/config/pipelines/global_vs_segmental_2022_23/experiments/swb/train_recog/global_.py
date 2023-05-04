from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import SegmentalLabelDefinition, GlobalLabelDefinition, LabelDefinition

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.training import TrainExperiment, SegmentalTrainExperiment, GlobalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.recognition.segmental import run_returnn_simple_segmental_decoding, run_rasr_segmental_decoding
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.realignment import run_rasr_segmental_realignment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.recognition.global_ import run_returnn_label_sync_decoding
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.train_recog.base import TrainRecogPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.analysis import GlobalAttentionWeightsPlotter

from i6_core.returnn.training import Checkpoint

from abc import abstractmethod, ABC
from typing import Dict, List, Type, Optional, Tuple
from sisyphus import Path


class GlobalTrainRecogPipeline(TrainRecogPipeline):
  def __init__(self, dependencies: GlobalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    if self.recog_type != "standard":
      raise NotImplementedError

    self.dependencies = dependencies

  def run_training(self, import_model_train_epoch1: Optional[Checkpoint], train_alias: str) -> Dict[int, Checkpoint]:
    return GlobalTrainExperiment(
      dependencies=self.dependencies,
      variant_params=self.variant_params,
      num_epochs=self.num_epochs,
      base_alias=self.base_alias,
      import_model_train_epoch1=import_model_train_epoch1).run_training()

  def run_recog(self, checkpoints: Dict[int, Checkpoint]):
    for i, (epoch, checkpoint) in enumerate(checkpoints.items()):
      base_alias = "%s/epoch_%d" % (self.base_alias, epoch)

      variant_params = self._remove_pretrain_from_config(epoch=epoch)

      run_returnn_label_sync_decoding(
        dependencies=self.dependencies,
        variant_params=variant_params,
        base_alias=base_alias,
        checkpoint=checkpoint,
        test_corpora_keys=["dev"],
        calc_search_errors=True,
        search_error_corpus_key="cv")

  def run(self):
    train_alias = "train"
    self.checkpoints["train"] = self.run_training(
      import_model_train_epoch1=self.import_model_train_epoch1,
      initial_lr=self.import_model_train_epoch1_initial_lr if self.,
      train_alias=train_alias,
    )
    if self.do_recog:
      self.run_recog(checkpoints=self.checkpoints["train"])
