from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition, SegmentalLabelDefinition, GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_train_config as get_segmental_train_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_recog_config as get_segmental_recog_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.global_ import get_train_config as get_global_train_config

from i6_experiments.users.schmitt.returnn.tools import CalcSearchErrorJob
from i6_experiments.users.schmitt.datasets.extern_sprint import get_dataset_dict

from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn.forward import ReturnnForwardJob

from abc import ABC, abstractmethod
from typing import Dict
import copy

from sisyphus import *


class SearchErrorExperiment(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          variant_params: Dict,
          checkpoint: Checkpoint,
          search_targets: Path,
          ref_targets: Path,
          corpus_key: str,
          base_alias: str
    ):
    self.ref_targets = ref_targets
    self.corpus_key = corpus_key
    self.dependencies = dependencies
    self.variant_params = variant_params
    self.checkpoint = checkpoint
    self.search_targets = search_targets
    self.alias = "%s/search_errors_%s" % (base_alias, corpus_key)

  @property
  @abstractmethod
  def model_type(self):
    pass

  @property
  @abstractmethod
  def label_name(self):
    pass

  @property
  @abstractmethod
  def blank_idx(self):
    pass

  @property
  @abstractmethod
  def returnn_config(self):
    pass

  def create_calc_search_error_job(self):
    calc_search_err_job = CalcSearchErrorJob(
      returnn_config=self.returnn_config,
      rasr_config=self.dependencies.rasr_config_paths["feature_extraction"][self.corpus_key],
      rasr_nn_trainer_exe=RasrExecutables.nn_trainer_path,
      segment_file=self.dependencies.segment_paths[self.corpus_key],
      blank_idx=self.blank_idx,
      model_type=self.model_type,
      label_name=self.label_name,
      search_targets=self.search_targets,
      ref_targets=self.ref_targets,
      max_seg_len=-1, length_norm=False)
    calc_search_err_job.add_alias(self.alias)
    tk.register_output(self.alias, calc_search_err_job.out_search_errors)


class SegmentalSearchErrorExperiment(SearchErrorExperiment):
  def __init__(self, dependencies: SegmentalLabelDefinition, length_scale: float, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies
    self.length_scale = length_scale

  @property
  def model_type(self):
    return "seg"

  @property
  def label_name(self):
    return "alignment"

  @property
  def blank_idx(self):
    return self.dependencies.model_hyperparameters.blank_idx

  @property
  def returnn_config(self):
    return get_segmental_train_config(
      dependencies=self.dependencies,
      alignments=None,
      variant_params=self.variant_params,
      load=self.checkpoint,
      length_scale=self.length_scale)


class GlobalSearchErrorExperiment(SearchErrorExperiment):
  def __init__(self, dependencies: GlobalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  @property
  def model_type(self):
    return "glob"

  @property
  def label_name(self):
    return "bpe"

  @property
  def blank_idx(self):
    # here, we just return a random idx, as it is not used in the case of a global att model
    return 0

  @property
  def returnn_config(self):
      return get_global_train_config(self.dependencies, self.variant_params, load=self.checkpoint)


class SearchErrorExperimentV2(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          variant_params: Dict,
          checkpoint: Checkpoint,
          search_targets: Path,
          ref_targets: Path,
          corpus_key: str,
          base_alias: str
    ):
    self.ref_targets = ref_targets
    self.corpus_key = corpus_key
    self.dependencies = dependencies
    self.variant_params = variant_params
    self.checkpoint = checkpoint
    self.search_targets = search_targets
    self.alias = "%s/search_errors_%s" % (base_alias, corpus_key)

  @property
  @abstractmethod
  def model_type(self):
    pass

  @property
  @abstractmethod
  def label_name(self):
    pass

  @property
  @abstractmethod
  def blank_idx(self):
    pass

  @property
  @abstractmethod
  def returnn_config(self) -> ReturnnConfig:
    pass

  def create_calc_search_error_job(self):
    returnn_config_ext = copy.deepcopy(self.returnn_config)
    # returnn_config_ext.config["eval"] = copy.deepcopy(returnn_config_ext.config["search_data"])
    returnn_config_ext.config["eval"] = get_dataset_dict(
      rasr_config_path="/u/schmitt/experiments/segmental_models_2021_22/work/i6_core/rasr/config/WriteRasrConfigJob.T7WSrzHjGcrs/output/rasr.config",
      rasr_nn_trainer_exe="/u/schmitt/src/rasr/arch/linux-x86_64-standard/nn-trainer.linux-x86_64-standard",
      target_hdf="/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-dev.hdf",
      segment_path="/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000",
      partition_epoch=1,
      seq_order_seq_lens_file=None,
      seq_ordering="default",
    )
    returnn_config_ext.config["extern_data"]["targetb"] = copy.deepcopy(returnn_config_ext.config["extern_data"]["alignment"])
    returnn_config_ext.config["eval"]["data_map"]["targetb"] = ("targets", "data")
    del returnn_config_ext.config["eval"]["data_map"]["targets"]
    # returnn_config_ext.config["forward_use_search"] = True
    del returnn_config_ext.config["search_data"]
    del returnn_config_ext.config["task"]
    returnn_config_ext.config["network"]["output"]["unit"]["output"]["cheating"] = "exclusive"


    dump_forward_job = ReturnnForwardJob(
      model_checkpoint=self.checkpoint,
      returnn_config=returnn_config_ext,
      returnn_python_exe=RETURNN_EXE,
      returnn_root=RETURNN_ROOT,
      eval_mode=True,
      hdf_outputs=["output"]
    )
    dump_forward_job.add_alias(self.alias)
    tk.register_output("dump_forward_test", dump_forward_job.out_hdf_files["output"])

    # calc_search_err_job = CalcSearchErrorJob(
    #   returnn_config=self.returnn_config,
    #   rasr_config=self.dependencies.rasr_config_paths["feature_extraction"][self.corpus_key],
    #   rasr_nn_trainer_exe=RasrExecutables.nn_trainer_path,
    #   segment_file=self.dependencies.segment_paths[self.corpus_key],
    #   blank_idx=self.blank_idx,
    #   model_type=self.model_type,
    #   label_name=self.label_name,
    #   search_targets=self.search_targets,
    #   ref_targets=self.ref_targets,
    #   max_seg_len=-1, length_norm=False)
    # calc_search_err_job.add_alias(self.alias)
    # tk.register_output(self.alias, calc_search_err_job.out_search_errors)


class SegmentalSearchErrorExperimentV2(SearchErrorExperimentV2):
  def __init__(self, dependencies: SegmentalLabelDefinition, length_scale: float, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies
    self.length_scale = length_scale

  @property
  def model_type(self):
    return "seg"

  @property
  def label_name(self):
    return "alignment"

  @property
  def blank_idx(self):
    return self.dependencies.model_hyperparameters.blank_idx

  @property
  def returnn_config(self) -> ReturnnConfig:
    return get_segmental_recog_config(
      dependencies=self.dependencies,
      variant_params=self.variant_params,
      corpus_key="cv",
      dump_output=False,
      length_scale=1.,
      beam_size=12,
      use_recomb=False,
    )
