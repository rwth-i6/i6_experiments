from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import SegmentalLabelDefinition, GlobalLabelDefinition, LabelDefinition

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.training import TrainExperiment, SegmentalTrainExperiment, GlobalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.recognition.segmental import run_returnn_simple_segmental_decoding, run_rasr_segmental_decoding
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.realignment import run_rasr_segmental_realignment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.recognition.global_ import run_returnn_label_sync_decoding
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.analysis import AttentionWeightPlotter

from i6_core.returnn.training import Checkpoint

from abc import abstractmethod, ABC
from typing import Dict, List, Type, Optional, Tuple
from sisyphus import Path


class TrainRecogPipeline(ABC):
  def __init__(
          self,
          model_type: str,
          variant_name: str,
          variant_params: Dict,
          dependencies: LabelDefinition,
          num_epochs: Tuple,
          base_alias: str,
          recog_type: str = "standard",
          do_recog: bool = True,
          returnn_recog_epochs: Optional[Tuple] = None,
          import_model_train_epoch1: Optional[Checkpoint] = None,
          import_model_train_epoch1_alias: Optional[str] = None):

    self.model_type = model_type,
    self.variant_name = variant_name
    self.variant_params = variant_params
    self.dependencies = dependencies
    self.num_epochs = num_epochs
    self.import_model_train_epoch1 = import_model_train_epoch1
    assert import_model_train_epoch1 is None or type(import_model_train_epoch1_alias) == str
    self.import_model_train_epoch1_alias = import_model_train_epoch1_alias

    assert not do_recog or recog_type is not None
    self.recog_type = recog_type
    self.do_recog = do_recog

    self.returnn_recog_epochs = returnn_recog_epochs if returnn_recog_epochs is not None else self.num_epochs
    for epoch in self.returnn_recog_epochs:
      assert epoch in self.num_epochs, "Cannot do RETURNN recog on epoch %d because it is not set in num_epochs"

    self.base_alias = base_alias
    if import_model_train_epoch1_alias is None:
      self.base_alias = "%s/%s" % (self.base_alias, "no_import")
    else:
      self.base_alias = "%s/import_%s" % (self.base_alias, self.import_model_train_epoch1_alias)

    self.checkpoints = {}

  @abstractmethod
  def run_recog(self, checkpoints: Dict[int, Checkpoint]):
    pass

  @abstractmethod
  def run_training(self, import_model_train_epoch1: Optional[Checkpoint], train_alias: str) -> Dict[int, Checkpoint]:
    pass

  def run(self):
    train_alias = "train"
    self.checkpoints["train"] = self.run_training(
      import_model_train_epoch1=self.import_model_train_epoch1,
      train_alias=train_alias
    )
    if self.do_recog:
      self.run_recog(checkpoints=self.checkpoints["train"])
