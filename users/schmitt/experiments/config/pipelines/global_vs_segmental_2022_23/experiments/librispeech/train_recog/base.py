from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition

from i6_core.returnn.training import Checkpoint

from abc import abstractmethod, ABC
from typing import Dict, List, Type, Optional, Tuple
from sisyphus import Path
import copy


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
          import_model_train_epoch1_alias: Optional[str] = None,
          import_model_train_epoch1_initial_lr: Optional[float] = None
    ):

    self.model_type = model_type,
    self.variant_name = variant_name
    self.variant_params = variant_params
    self.dependencies = dependencies
    self.num_epochs = num_epochs
    self.import_model_train_epoch1 = import_model_train_epoch1
    assert import_model_train_epoch1 is None or type(import_model_train_epoch1_alias) == str
    self.import_model_train_epoch1_alias = import_model_train_epoch1_alias
    assert import_model_train_epoch1_initial_lr is None or import_model_train_epoch1 is not None, "Setting `import_model_train_epoch1_initital_lr` when not importing a model won't have an effect"
    self.import_model_train_epoch1_initial_lr = import_model_train_epoch1_initial_lr

    assert not do_recog or recog_type is not None
    self.recog_type = recog_type
    self.do_recog = do_recog

    self.returnn_recog_epochs = returnn_recog_epochs if returnn_recog_epochs is not None else self.num_epochs
    for epoch in self.returnn_recog_epochs:
      assert epoch in self.num_epochs, "Cannot do RETURNN recog on epoch %d because it is not set in num_epochs"

    self.checkpoints = {}

    self.base_alias = base_alias

  def _get_base_alias(self, base_alias) -> str:
    if self.import_model_train_epoch1_alias is None:
      base_alias = "%s/%s" % (base_alias, "no_import")
    else:
      base_alias = "%s/import_%s" % (base_alias, self.import_model_train_epoch1_alias)
    if self.import_model_train_epoch1_initial_lr is not None:
      base_alias = "%s_initial-lr-%f" % (base_alias, self.import_model_train_epoch1_initial_lr)

    return base_alias

  @abstractmethod
  def run_recog(self, checkpoints: Dict[int, Checkpoint]):
    pass

  @abstractmethod
  def run_training(
          self,
          import_model_train_epoch1: Optional[Checkpoint],
          train_alias: str,
          initial_lr: Optional[float] = None,
  ) -> Tuple[Dict[int, Checkpoint], Path]:
    pass

  @abstractmethod
  def run(self):
    pass
