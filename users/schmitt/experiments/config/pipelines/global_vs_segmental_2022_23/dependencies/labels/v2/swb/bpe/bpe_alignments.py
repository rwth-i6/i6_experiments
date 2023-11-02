from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import \
  RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import \
  SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.bpe.bpe import \
  SWBBPE1030, SWBBPE534
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.general import \
  SWBLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import \
  SegmentalModelHyperparameters
from i6_experiments.users.schmitt.rasr.convert import BPEJSONVocabToRasrFormatsJob

from typing import Dict
import copy

from sisyphus import *


class SwbBpe1030RnaAlignment(SWBBPE1030, SWBLabelDefinition, SegmentalLabelDefinition):
  """
    This alignment is generated by the RNA full-sum model from Albert's "Improved Transducer".
    See: https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer.
  """

  def __init__(self):
    super().__init__()

    self.hdf_targets = {
      "train": Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-train.hdf",
        cached=True),
      "devtrain": Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-train.hdf",
        cached=True),
      "cv": Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-dev.hdf",
        cached=True)
    }

  @property
  def alias(self) -> str:
    return "bpe"

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=1031, sil_idx=None, blank_idx=1030)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    json_to_rasr_job = BPEJSONVocabToRasrFormatsJob(
      self.vocab_path, blank_idx=self.model_hyperparameters.blank_idx)

    return RasrFormats(
      state_tying_path=json_to_rasr_job.out_state_tying,
      allophone_path=json_to_rasr_job.out_allophones,
      label_file_path=json_to_rasr_job.out_rasr_label_file,
      decoding_lexicon_path=Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe-with-sil_lexicon"),
      realignment_lexicon_path=Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/my_new_lex"),
      blank_allophone_state_idx=4119
    )

  @property
  def alignment_paths(self):
    return {
      "train": Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-train.hdf",
        cached=True),
      "devtrain": Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-train.hdf",
        cached=True),
      "cv": Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-dev.hdf",
        cached=True)
    }

  @property
  def segment_paths(self):
    return {
      "train": Path(
        "/u/schmitt/experiments/transducer/config/dependencies/seg_train", cached=True
      ),
      "devtrain": Path(
        "/u/schmitt/experiments/transducer/config/dependencies/seg_train_head3000", cached=True
      ),
      "cv": Path(
        "/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000", cached=True
      )
    }

  @property
  def oggzip_paths(self) -> Dict:
    oggzip_paths = copy.deepcopy(super().oggzip_paths)
    oggzip_paths["cv"] = oggzip_paths["train"]
    return oggzip_paths


class SwbBpe534CtcAlignment(SWBBPE534, SWBLabelDefinition, SegmentalLabelDefinition):
  """
    This alignment is generated by the RNA full-sum model from Albert's "Improved Transducer".
    See: https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer.
  """

  @property
  def alias(self) -> str:
    return "bpe"

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=535, sil_idx=None, blank_idx=534)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    pass

  @property
  def alignment_paths(self):
    return {}