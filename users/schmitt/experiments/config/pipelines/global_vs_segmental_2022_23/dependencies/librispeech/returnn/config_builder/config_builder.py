from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition, LabelDefinition
from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict
from i6_experiments.users.schmitt.conformer_pretrain import get_network
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask
from i6_experiments.users.schmitt.dynamic_lr import dynamic_lr_str

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

import os
import re
from abc import abstractmethod, ABC
from typing import Dict


class ConfigBuilder(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          variant_params: Dict,
          initial_lr=None,
          import_model=None,
          import_model_train_epoch1=None,
  ):
    self.dependencies = dependencies
    self.variant_params = variant_params

    self.post_config_dict = dict(
      cleanup_old_models=True
    )

    self.config_dict = dict(
      use_tensorflow=True,
      log_batch_size=True,
      truncation=-1,
      tf_log_memory_usage=True,
      debug_print_layer_output_template=True
    )

    self.config_dict.update(dict(
      batching="random",
      max_seqs=200,
      max_seq_length={"targets": 75},
    ))

    if import_model is not None:
      self.config_dict["load"] = import_model
    if import_model_train_epoch1 is not None:
      self.config_dict["import_model_train_epoch1"] = import_model_train_epoch1

    self.config_dict.update(dict(
      gradient_clip=0.0,
      gradient_noise=0.0,
      adam=True,
      optimizer_epsilon=1e-8,
      accum_grad_multiple_step=2,
      learning_rate=initial_lr if initial_lr is not None else 0.0009,
      learning_rate_control="constant",
      learning_rates=[8e-5] * 35
    ))

    self.python_prolog = [
      "from returnn.tf.util.data import DimensionTag",
      "import os",
      "import numpy as np",
      "from subprocess import check_output, CalledProcessError",
      "import sys",
      "sys.setrecursionlimit(4000)",
      _mask,
      random_mask,
      transform,
      speed_pert,
    ]

    self.python_epilog = [
      dynamic_lr_str.format(
        initial_lr=8.999999999999999e-05,
        peak_lr=0.0009,
        final_lr=1e-06,
        cycle_ep=915,
        total_ep=2035,
        n_step=1350
      )
    ]

  @abstractmethod
  def get_net_dict(self):
    pass

  @abstractmethod
  def get_train_dataset_dict(self):
    pass

  @abstractmethod
  def get_cv_dataset_dict(self):
    pass

  @abstractmethod
  def get_eval_datasets_dict(self):
    pass

  @abstractmethod
  def get_search_dataset_dict(self, corpus_key: str):
    pass

  @abstractmethod
  def get_extern_data_dict(self):
    return dict(data=dict(
      dim=1,
      same_dim_tags_as=dict(
        t=CodeWrapper("DimensionTag(kind=DimensionTag.Types.Spatial, description='time', dimension=None)")
      )
    ))

  def get_train_datasets(self):
    return dict(
      extern_data=self.get_extern_data_dict(),
      train=self.get_train_dataset_dict(),
      dev=self.get_cv_dataset_dict(),
      eval_datasets=self.get_eval_datasets_dict()
    )

  def get_search_dataset(self, search_corpus_key: str):
    return dict(
      extern_data=self.get_extern_data_dict(),
      search_data=self.get_search_dataset_dict(corpus_key=search_corpus_key)
    )

  def get_train_config(self):
    self.config_dict.update(
      dict(
        task="train",
        batch_size=2400000
      )
    )

    self.config_dict.update(self.get_train_datasets())

    from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict
    networks_dict = networks_dict

    return ReturnnConfig(
      config=self.config_dict,
      post_config=self.post_config_dict,
      python_prolog=self.python_prolog,
      python_epilog=self.python_epilog,
      staged_network_dict=networks_dict
    )

  def get_recog_config(self, search_corpus_key: str):
    self.config_dict.update(dict(
      task="search",
      batch_size=2400000,
      search_output_layer="decision"
    ))

    self.config_dict.update(self.get_search_dataset(search_corpus_key=search_corpus_key))

    from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict
    self.config_dict["network"] = networks_dict[36]

    return ReturnnConfig(
      config=self.config_dict,
      post_config=self.post_config_dict,
      python_prolog=self.python_prolog,
      python_epilog=self.python_epilog,
      staged_network_dict=None)


class GlobalConfigBuilder(ConfigBuilder):
  def __init__(
          self,
          dependencies: GlobalLabelDefinition,
          variant_params: Dict
  ):
    super().__init__(dependencies=dependencies, variant_params=variant_params, **variant_params["config"])

  def get_net_dict(self):
    pass

  def get_default_dataset_opts(self, corpus_key):
    return dict(
      oggzip_path_list=self.dependencies.oggzip_paths[corpus_key],
      bpe_file=self.dependencies.bpe_codes_path,
      vocab_file=self.dependencies.vocab_path,
      segment_file=self.dependencies.segment_paths[corpus_key]
    )

  def get_train_dataset_dict(self):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=20,
      pre_process=CodeWrapper("speed_pert"),
      seq_ordering="laplace:.1000",
      epoch_wise_filter={(1, 5): {"max_mean_len": 1000}},
      **self.get_default_dataset_opts("train")
    )

  def get_cv_dataset_dict(self):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts("cv")
    )

  def get_eval_datasets_dict(self):
    return {
      "devtrain": get_dataset_dict(
        fixed_random_subset=3000,
        partition_epoch=1,
        pre_process=None,
        seq_ordering="sorted_reverse",
        epoch_wise_filter=None,
        **self.get_default_dataset_opts("devtrain")
      )
    }

  def get_search_dataset_dict(self, corpus_key: str):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts(corpus_key)
    )

  def get_extern_data_dict(self):
    extern_data_dict = super().get_extern_data_dict()
    extern_data_dict.update(dict(
      targets=dict(dim=self.dependencies.model_hyperparameters.target_num_labels, sparse=True)
    ))

    return extern_data_dict
