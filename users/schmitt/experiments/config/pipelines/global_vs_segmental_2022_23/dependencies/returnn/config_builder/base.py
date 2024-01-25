from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition, SegmentalLabelDefinition, LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBSprintCorpora, SWBOggZipCorpora
from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict as get_oggzip_dataset_dict
from i6_experiments.users.schmitt.datasets.concat import get_concat_dataset_dict
from i6_experiments.users.schmitt.datasets.extern_sprint import get_dataset_dict as get_extern_sprint_dataset_dict
from i6_experiments.users.schmitt.conformer_pretrain import get_network
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask
from i6_experiments.users.schmitt.augmentation.alignment import shift_alignment_boundaries_func_str
from i6_experiments.users.schmitt.dynamic_lr import dynamic_lr_str
from i6_experiments.users.schmitt.chunking import custom_chunkin_func_str
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder import network_builder, ilm_correction
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn import custom_construction_algos
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from sisyphus import Path

import os
import re
from abc import abstractmethod, ABC
from typing import Dict, Optional, List
import copy
import numpy as np


class ConfigBuilder(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          variant_params: Dict,
  ):
    self.dependencies = dependencies
    self.variant_params = variant_params

    self.post_config_dict = {
      "cleanup_old_models": True
    }

    self.python_epilog = []

    self.config_dict = {
      "use_tensorflow": True,
      "log_batch_size": True,
      "truncation": -1,
      "tf_log_memory_usage": True,
      "debug_print_layer_output_template": True
    }

    self.config_dict.update({
      "batching": "random",
      "max_seqs": 200,
      # "max_seq_length": {"targets": 75},
    })

    self.config_dict.update({
      "gradient_clip": 0.0,
      "gradient_noise": 0.0,
      "adam": True,
      "optimizer_epsilon": 1e-8,
      "accum_grad_multiple_step": 2,
    })

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
    ]

  def get_train_config(self, opts: Dict, python_epilog: Optional[Dict] = None):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog if python_epilog is None else python_epilog)

    config_dict.update(self.get_train_datasets(dataset_opts=opts.get("dataset_opts", {})))

    if opts.get("preload_from_files"):
      net_dict = self.get_final_net_dict(config_dict=config_dict, python_prolog=python_prolog)
      assert net_dict is not None
      config_dict["preload_from_files"] = opts["preload_from_files"]
      networks_dict = None
    elif opts.get("import_model_train_epoch1"):
      net_dict = self.get_final_net_dict(config_dict=config_dict, python_prolog=python_prolog)
      assert net_dict is not None
      config_dict.update({
        "import_model_train_epoch1": opts["import_model_train_epoch1"],
        "load_ignore_missing_vars": True,
      })
      networks_dict = None
    else:
      networks_dict = copy.deepcopy(self.get_networks_dict("train", config_dict=config_dict, python_prolog=python_prolog))
      net_dict = self.get_net_dict("train", config_dict=config_dict, python_prolog=python_prolog)
    if opts.get("align_augment", False):
      self.add_align_augment(net_dict=net_dict, networks_dict=networks_dict, python_prolog=python_prolog)

    if opts.get("only_train_length_model", False):
      assert "preload_from_files" in opts or "import_model_train_epoch1" in opts, "It does not make sense to only train the length model if you are not importing from some checkpoint"
      self.edit_network_only_train_length_model(net_dict)

    if opts.get("no_ctc_loss", False):
      # remove the ctc layer from the network(s)
      if networks_dict is not None:
        for network in networks_dict:
          network.pop("ctc", None)
      if net_dict is not None:
        net_dict.pop("ctc", None)

    if net_dict is not None:
      config_dict["network"] = net_dict

    if "cleanup_old_models" in opts:
      post_config_dict["cleanup_old_models"] = opts["cleanup_old_models"]

    if "tf_session_opts" in opts:
      post_config_dict["tf_session_opts"] = opts["tf_session_opts"]

    if opts.get("lr_opts"):
      config_dict.update(self.get_lr_settings(lr_opts=opts["lr_opts"]))
    else:
      config_dict.update(self.get_lr_settings(lr_opts=self.get_default_lr_opts()))

    if "batch_size" in opts:
      config_dict["batch_size"] = opts["batch_size"]
    else:
      config_dict["batch_size"] = self.get_default_batch_size()

    if "max_seq_length" in opts:
      config_dict["max_seq_length"] = opts["max_seq_length"]

    self.get_custom_construction_algo(config_dict=config_dict, python_prolog=python_prolog)

    return ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
      staged_network_dict=networks_dict
    )

  def get_recog_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    config_dict.update(dict(
      task="search",
      search_output_layer="decision"
    ))

    config_dict.update(
      self.get_search_dataset(
        search_corpus_key=opts["search_corpus_key"],
        dataset_opts=opts.get("dataset_opts", {})
      ))

    if opts.get("load_ignore_missing_vars"):
      config_dict["load_ignore_missing_vars"] = True

    net_dict = self.get_net_dict("search", config_dict=config_dict, python_prolog=python_prolog)
    if net_dict is not None:
      if opts.get("load_ignore_missing_vars"):
        # dirty fix
        # for some reason, RETURNN tries to initialize the weights of the ctc layer (even though it is not used
        # during search) and this leads to an error because the targets of the ctc layer are not set correctly
        # later, just remove the ctc layer in general during recog but keep for now because of hashes
        net_dict.pop("ctc")

      config_dict["network"] = net_dict

    if opts.get("batch_size"):
      config_dict["batch_size"] = opts["batch_size"]
    else:
      config_dict["batch_size"] = 2400000

    return ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
      staged_network_dict=self.get_networks_dict("search", config_dict=config_dict, python_prolog=python_prolog))

  def get_eval_config(self, eval_corpus_key: str, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    config_dict.update(
      self.get_eval_dataset(
        eval_corpus_key=eval_corpus_key,
        dataset_opts=opts.get("dataset_opts", {})
      ))

    if opts.get("use_train_net", False):
      net_dict = self.get_final_net_dict(config_dict=config_dict, python_prolog=python_prolog)
    else:
      net_dict = self.get_net_dict("search", config_dict=config_dict, python_prolog=python_prolog)

    assert net_dict is not None
    config_dict["network"] = net_dict

    if "batch_size" in opts:
      config_dict["batch_size"] = opts["batch_size"]
    else:
      config_dict["batch_size"] = self.get_default_batch_size()

    return ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
      staged_network_dict=self.get_networks_dict("search", config_dict=config_dict, python_prolog=python_prolog))

  def get_ctc_align_config(self, corpus_key: str, opts: Dict):
    returnn_config = self.get_eval_config(eval_corpus_key=corpus_key, opts=opts)

    assert "ctc" in returnn_config.config["network"]
    returnn_config.config["network"].update({
      "ctc_forced_align": {
        "align_target": opts["align_target"],
        "class": "forced_align",
        "from": "ctc",
        "input_type": "prob",
        "topology": "rna",
      },
      "ctc_forced_align_dump": {
        "class": "hdf_dump",
        "filename": opts["hdf_filename"],
        "from": "ctc_forced_align",
        "is_output_layer": True,
      },
    })

    return returnn_config

  def get_compile_tf_graph_config(self, opts: Dict):
    pass

  @abstractmethod
  def get_recog_config_for_forward_job(self, opts: Dict):
    pass

  @abstractmethod
  def get_dump_att_weight_config(self, corpus_key: str, opts: Dict):
    pass

  @abstractmethod
  def get_dump_length_model_probs_config(self, corpus_key: str, opts: Dict):
    pass

  @abstractmethod
  def get_dump_scores_config(self, corpus_key: str, opts: Dict):
    pass

  def get_custom_construction_algo(self, config_dict, python_prolog):
    pass

  @abstractmethod
  def edit_network_only_train_length_model(self, net_dict: Dict):
    pass

  def add_align_augment(self, net_dict, networks_dict, python_prolog):
    raise NotImplementedError

  def get_lr_settings(self, lr_opts):
    lr_settings = {}
    if lr_opts["type"] == "newbob":
      lr_opts.pop("type")
      lr_settings.update(lr_opts)
      # lr_settings.update({
      #   "learning_rate": 0.001,
      #   "learning_rate_control": "newbob_multi_epoch",
      #   "learning_rate_control_error_measure": lr_opts["learning_rate_control_error_measure"],
      #   "learning_rate_control_min_num_epochs_per_new_lr": 3,
      #   "learning_rate_control_relative_error_relative_lr": True,
      #   "newbob_learning_rate_decay": 0.7,
      #   "newbob_multi_num_epochs": 6,
      #   "newbob_multi_update_interval": 1,
      #   "min_learning_rate": 2e-05,
      #   "use_learning_rate_control_always": True
      # })
    elif lr_opts["type"] == "const_then_linear":
      const_lr = lr_opts["const_lr"]
      const_frac = lr_opts["const_frac"]
      final_lr = lr_opts["final_lr"]
      num_epochs = lr_opts["num_epochs"]
      lr_settings.update({
        "learning_rates": [const_lr] * int((num_epochs*const_frac)) + list(np.linspace(const_lr, final_lr, num_epochs - int((num_epochs*const_frac)))),
      })
    elif lr_opts["type"] == "const":
      const_lr = lr_opts["const_lr"]
      lr_settings.update({
        "learning_rate": const_lr,
      })
    else:
      raise NotImplementedError

    return lr_settings

  @abstractmethod
  def get_default_lr_opts(self):
    pass

  @abstractmethod
  def get_default_batch_size(self):
    pass

  @abstractmethod
  def get_final_net_dict(self, config_dict, python_prolog):
    pass

  def get_default_dataset_opts(self, corpus_key: str, dataset_opts: Dict):
    hdf_targets = dataset_opts.get("hdf_targets", self.dependencies.hdf_targets)
    segment_paths = dataset_opts.get("segment_paths", self.dependencies.segment_paths)
    if self.variant_params["dataset"]["feature_type"] == "raw":
      return {
        "oggzip_path_list": self.variant_params["dataset"]["corpus"].oggzip_paths[corpus_key],
        "bpe_file": self.dependencies.bpe_codes_path,
        "vocab_file": self.dependencies.vocab_path,
        "segment_file": segment_paths.get(corpus_key, None),
        "hdf_targets": hdf_targets.get(corpus_key, None)
      }
    else:
      assert self.variant_params["dataset"]["feature_type"] == "gammatone"
      return {
        "rasr_config_path": self.variant_params["dataset"]["corpus"].rasr_feature_extraction_config_paths[corpus_key],
        "rasr_nn_trainer_exe": RasrExecutables.nn_trainer_path,
        "segment_path": self.dependencies.segment_paths[corpus_key],
        "target_hdf": hdf_targets.get(corpus_key, None)
      }

  def get_train_dataset_dict(self, dataset_opts: Dict):
    if self.variant_params["dataset"]["feature_type"] == "raw":
      return get_oggzip_dataset_dict(
        fixed_random_subset=None,
        partition_epoch=self.variant_params["dataset"]["corpus"].partition_epoch,
        pre_process=CodeWrapper("speed_pert") if self.variant_params["config"].get("speed_pert") else None,
        seq_ordering=self.variant_params["config"]["train_seq_ordering"],
        epoch_wise_filter=self.variant_params["config"].get("epoch_wise_filter", None),
        **self.get_default_dataset_opts("train", dataset_opts)
      )
    else:
      assert self.variant_params["dataset"]["feature_type"] == "gammatone"
      return get_extern_sprint_dataset_dict(
        partition_epoch=self.variant_params["dataset"]["corpus"].partition_epoch,
        seq_ordering="laplace:227",
        seq_order_seq_lens_file=Path("/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"),
        **self.get_default_dataset_opts("train", dataset_opts)
      )

  def get_cv_dataset_dict(self, dataset_opts: Dict):
    if self.variant_params["dataset"]["feature_type"] == "raw":
      return get_oggzip_dataset_dict(
        fixed_random_subset=None,
        partition_epoch=1,
        pre_process=None,
        seq_ordering="sorted_reverse",
        epoch_wise_filter=None,
        **self.get_default_dataset_opts("cv", dataset_opts)
      )
    else:
      assert self.variant_params["dataset"]["feature_type"] == "gammatone"
      return get_extern_sprint_dataset_dict(
        partition_epoch=1,
        seq_ordering="default",
        seq_order_seq_lens_file=None,
        **self.get_default_dataset_opts("cv", dataset_opts)
      )

  def get_devtrain_dataset_dict(self, dataset_opts: Dict):
    if self.variant_params["dataset"]["feature_type"] == "raw":
      return get_oggzip_dataset_dict(
        fixed_random_subset=3000,
        partition_epoch=1,
        pre_process=None,
        seq_ordering="sorted_reverse",
        epoch_wise_filter=None,
        **self.get_default_dataset_opts("devtrain", dataset_opts)
      )
    else:
      assert self.variant_params["dataset"]["feature_type"] == "gammatone"
      return get_extern_sprint_dataset_dict(
        partition_epoch=1,
        seq_ordering="default",
        seq_order_seq_lens_file=None,
        **self.get_default_dataset_opts("devtrain", dataset_opts)
      )

  def get_search_dataset_dict(self, corpus_key: str, dataset_opts: Dict):
    if self.variant_params["dataset"]["feature_type"] == "raw":
      dataset_dict = get_oggzip_dataset_dict(
        fixed_random_subset=None,
        partition_epoch=1,
        pre_process=None,
        seq_ordering="sorted_reverse",
        epoch_wise_filter=None,
        **self.get_default_dataset_opts(corpus_key, dataset_opts)
      )

      concat_num = dataset_opts.get("concat_num")  # type: Optional[int]
      if concat_num:
        dataset_dict = get_concat_dataset_dict(
          original_dataset_dict=dataset_dict,
          seq_len_file=self.variant_params["dataset"]["corpus"].seq_len_files[corpus_key],
          seq_list_file=self.variant_params["dataset"]["corpus"].segment_paths[corpus_key + "_concat-%d" % concat_num]
        )

      return dataset_dict
    else:
      assert self.variant_params["dataset"]["feature_type"] == "gammatone"
      return get_extern_sprint_dataset_dict(
        partition_epoch=1,
        seq_ordering=None,
        seq_order_seq_lens_file=None,
        **self.get_default_dataset_opts(corpus_key, dataset_opts)
      )

  def get_eval_dataset_dict(self, corpus_key: str, dataset_opts: Dict):
    if self.variant_params["dataset"]["feature_type"] == "raw":
      return get_oggzip_dataset_dict(
        fixed_random_subset=None,
        partition_epoch=1,
        pre_process=None,
        seq_ordering="sorted_reverse",
        epoch_wise_filter=None,
        seq_postfix=dataset_opts.get("seq_postfix", self.dependencies.model_hyperparameters.sos_idx),
        **self.get_default_dataset_opts(corpus_key, dataset_opts)
      )
    else:
      assert self.variant_params["dataset"]["feature_type"] == "gammatone"
      return get_extern_sprint_dataset_dict(
        partition_epoch=1,
        seq_ordering=None,
        seq_order_seq_lens_file=None,
        **self.get_default_dataset_opts(corpus_key, dataset_opts)
      )

  def get_extern_data_dict(self):
    extern_data_dict = {}

    if self.variant_params["dataset"]["feature_type"] == "raw":
      extern_data_dict["data"] = {
        "dim": 1,
        "same_dim_tags_as": {
          "t": CodeWrapper("DimensionTag(kind=DimensionTag.Types.Spatial, description='time', dimension=None)")
        }
      }
    else:
      assert self.variant_params["dataset"]["feature_type"] == "gammatone"
      extern_data_dict["data"] = {
        "dim": 40,
        "same_dim_tags_as": {
          "t": CodeWrapper("DimensionTag(kind=DimensionTag.Types.Spatial, description='time', dimension=None)")
        }
      }

    extern_data_dict["targets"] = {
      "dim": self.dependencies.model_hyperparameters.target_num_labels,
      "sparse": True
    }

    return extern_data_dict

  @abstractmethod
  def get_net_dict(self, task: str, config_dict, python_prolog):
    pass

  @abstractmethod
  def get_networks_dict(self, task: str, config_dict, python_prolog):
    pass

  def get_train_datasets(self, dataset_opts: Dict):
    return dict(
      extern_data=self.get_extern_data_dict(),
      train=self.get_train_dataset_dict(dataset_opts),
      dev=self.get_cv_dataset_dict(dataset_opts),
      eval_datasets={"devtrain": self.get_devtrain_dataset_dict(dataset_opts)}
    )

  def get_search_dataset(self, search_corpus_key: str, dataset_opts: Dict):
    return dict(
      extern_data=self.get_extern_data_dict(),
      search_data=self.get_search_dataset_dict(corpus_key=search_corpus_key, dataset_opts=dataset_opts)
    )

  def get_eval_dataset(self, eval_corpus_key: str, dataset_opts: Dict):
    return dict(
      extern_data=self.get_extern_data_dict(),
      eval=self.get_eval_dataset_dict(corpus_key=eval_corpus_key, dataset_opts=dataset_opts)
    )


class SWBBlstmConfigBuilder(ConfigBuilder, ABC):
  def get_default_batch_size(self):
    return 10000

  def get_final_net_dict(self, config_dict, python_prolog):
    return self.get_net_dict("train", config_dict=config_dict, python_prolog=python_prolog)


class ConformerConfigBuilder(ConfigBuilder, ABC):
  def get_final_net_dict(self, config_dict, python_prolog):
    networks_dict = self.get_networks_dict("train", config_dict=config_dict, python_prolog=python_prolog)
    last_epoch = sorted(list(networks_dict.keys()))[-1]
    return networks_dict[last_epoch]


class SwbConformerConfigBuilder(ConfigBuilder, ABC):
  def get_default_batch_size(self):
    return 1200000

  def get_default_lr_opts(self):
    raise NotImplementedError


class LibrispeechConformerConfigBuilder(ConfigBuilder, ABC):
  def get_default_batch_size(self):
    return 2400000

  def get_default_lr_opts(self):
    raise NotImplementedError
