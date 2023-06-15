from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.librispeech.training import GlobalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.librispeech.recognition.global_ import ReturnnLabelSyncBeamSearchPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.librispeech.train_recog.base import TrainRecogPipeline

from i6_core.returnn.training import Checkpoint

from typing import Dict, List, Type, Optional, Tuple
from sisyphus import Path


class GlobalTrainRecogPipeline(TrainRecogPipeline):
  def __init__(self, dependencies: GlobalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    if self.recog_type != "standard":
      raise NotImplementedError

    self.dependencies = dependencies

    self.base_alias = self._get_base_alias(base_alias=self.base_alias)

  def run_training(
          self,
          import_model_train_epoch1: Optional[Checkpoint],
          train_alias: str,
          initial_lr: Optional[float] = None,
  ) -> Tuple[Dict[int, Checkpoint], Path]:
    return GlobalTrainExperiment(
      dependencies=self.dependencies,
      variant_params=self.variant_params,
      num_epochs=self.num_epochs,
      base_alias=self.base_alias,
      import_model_train_epoch1=import_model_train_epoch1,
      initial_lr=initial_lr
    ).run_training()

  def run_recog(self, checkpoints: Dict[int, Checkpoint]):
    for i, (epoch, checkpoint) in enumerate(checkpoints.items()):
      base_alias = "%s/epoch_%d" % (self.base_alias, epoch)

      ReturnnLabelSyncBeamSearchPipeline(
        dependencies=self.dependencies,
        variant_params=self.variant_params,
        base_alias=base_alias,
        checkpoint=checkpoint,
        corpus_key="dev-other",
        beam_size=12
      ).run()

  def run(self):
    train_alias = "train"
    self.checkpoints["train"], lr_file_path = self.run_training(
      import_model_train_epoch1=self.import_model_train_epoch1,
      train_alias=train_alias,
      initial_lr=self.import_model_train_epoch1_initial_lr
    )

    if self.do_recog:
      self.run_recog(checkpoints=self.checkpoints["train"])
