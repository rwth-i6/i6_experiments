from abc import ABC
from typing import Optional


class ModelHyperparameters(ABC):
  def __init__(
          self,
          sos_idx: int,
          target_num_labels: int,
          target_num_labels_wo_blank: int,
          sil_idx: Optional[int]
  ):
    self.sil_idx = sil_idx
    self.sos_idx = sos_idx
    self.target_num_labels = target_num_labels
    self.target_num_labels_wo_blank = target_num_labels_wo_blank


class GlobalModelHyperparameters(ModelHyperparameters):
  pass


class SegmentalModelHyperparameters(ModelHyperparameters):
  def __init__(self, blank_idx, **kwargs):
    super().__init__(**kwargs)

    self.blank_idx = blank_idx
    self.target_num_labels_wo_blank = self.target_num_labels - 1
