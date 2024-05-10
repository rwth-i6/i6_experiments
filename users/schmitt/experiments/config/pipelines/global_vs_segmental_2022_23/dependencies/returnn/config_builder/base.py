from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict as get_oggzip_dataset_dict
from i6_experiments.users.schmitt.datasets.concat import get_concat_dataset_dict
from i6_experiments.users.schmitt.datasets.extern_sprint import get_dataset_dict as get_extern_sprint_dataset_dict
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask
from i6_experiments.users.schmitt.dynamic_lr import dynamic_lr_str
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder import network_builder

from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn.training import AverageTFCheckpointsJob, GetBestEpochJob, Checkpoint, GetBestTFCheckpointJob

from sisyphus import Path

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
      networks_dict = copy.deepcopy(
        self.get_networks_dict(
          "train",
          config_dict=config_dict,
          python_prolog=python_prolog,
          use_get_global_config=True
        )
      )
      net_dict = self.get_net_dict("train", config_dict=config_dict, python_prolog=python_prolog)
    if opts.get("align_augment", False):
      self.add_align_augment(net_dict=net_dict, networks_dict=networks_dict, python_prolog=python_prolog)

    if opts.get("dataset_opts", {}).get("use_speed_pert"):
      python_prolog.append(speed_pert_str)

    if opts.get("only_train_length_model", False):
      assert "preload_from_files" in opts or "import_model_train_epoch1" in opts, "It does not make sense to only train the length model if you are not importing from some checkpoint"
      self.edit_network_only_train_length_model(net_dict)

    if opts.get("no_ctc_loss", False):
      # remove the ctc layer from the network(s)
      if networks_dict is not None:
        for _, network in networks_dict.items():
          network.pop("ctc", None)
      if net_dict is not None:
        net_dict.pop("ctc", None)

    if net_dict is not None:
      if opts.get("freeze_encoder"):
        self.edit_network_freeze_encoder(net_dict)
      decoder_version = self.variant_params["network"]["decoder_version"]
      if decoder_version is not None:
        self.edit_network_modify_decoder(
          version=decoder_version,
          net_dict=net_dict,
          train=True,
          target_num_labels=self.dependencies.model_hyperparameters.target_num_labels_wo_blank
        )
      config_dict["network"] = net_dict

    if opts.get("cleanup_old_models"):
      post_config_dict["cleanup_old_models"] = opts["cleanup_old_models"]
    else:
      post_config_dict["cleanup_old_models"] = self.get_default_cleanup_old_models()

    if "tf_session_opts" in opts:
      post_config_dict["tf_session_opts"] = opts["tf_session_opts"]

    if opts.get("lr_opts"):
      config_dict.update(self.get_lr_settings(lr_opts=opts["lr_opts"], python_epilog=python_epilog))
    else:
      config_dict.update(self.get_lr_settings(lr_opts=self.get_default_lr_opts()))

    if opts.get("batch_size") is not None:
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
      staged_network_dict={
        k: f"from returnn.config import get_global_config\nnetwork = {str(v)}" for k, v in networks_dict.items()
      } if networks_dict is not None else None
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
      net_dict.pop("ctc")  # not needed during recognition
      # set beam size
      net_dict["output"]["unit"]["output"]["beam_size"] = opts.get("beam_size", self.get_default_beam_size())

      if opts.get("use_same_static_padding"):
        self.edit_network_use_same_static_padding(net_dict)

      decoder_version = self.variant_params["network"]["decoder_version"]
      if decoder_version is not None:
        self.edit_network_modify_decoder(
          version=decoder_version,
          net_dict=net_dict,
          train=False,
          target_num_labels=self.dependencies.model_hyperparameters.target_num_labels_wo_blank
        )

      config_dict["network"] = net_dict
    else:
      raise ValueError("net_dict is None!")

    if opts.get("batch_size"):
      config_dict["batch_size"] = opts["batch_size"]
    else:
      config_dict["batch_size"] = 2400000

    if opts.get("max_seqs"):
      config_dict["max_seqs"] = opts["max_seqs"]

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

    if opts.get("network_epoch"):
      # select network corresponding to given epoch
      networks_dict = copy.deepcopy(
        self.get_networks_dict(
          "train",
          config_dict=config_dict,
          python_prolog=python_prolog,
          use_get_global_config=True
        )
      )
      network_epoch = None
      for epoch in networks_dict:
        if epoch <= opts["network_epoch"]:
          network_epoch = epoch
        else:
          break
      net_dict = networks_dict[network_epoch]
    elif opts.get("use_train_net", False):
      # select network from training (final net dict normally)
      net_dict = self.get_final_net_dict(config_dict=config_dict, python_prolog=python_prolog)
    else:
      net_dict = self.get_net_dict("search", config_dict=config_dict, python_prolog=python_prolog)

    decoder_version = self.variant_params["network"]["decoder_version"]
    if decoder_version is not None:
      self.edit_network_modify_decoder(
        version=decoder_version,
        net_dict=net_dict,
        train=False,
        target_num_labels=self.dependencies.model_hyperparameters.target_num_labels_wo_blank
      )

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
    returnn_config.config["network"].update(
      network_builder.get_ctc_forced_align_hdf_dump(align_target=opts["align_target"], filename=opts["hdf_filename"])
    )

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
  def get_dump_scores_config(self, corpus_key: str, opts: Dict):
    pass

  def get_custom_construction_algo(self, config_dict, python_prolog):
    pass

  @abstractmethod
  def edit_network_only_train_length_model(self, net_dict: Dict):
    pass

  @abstractmethod
  def edit_network_freeze_encoder(self, net_dict: Dict):
    pass

  def edit_network_modify_decoder(self, version: int, net_dict: Dict, train: bool, target_num_labels: int):
    raise NotImplementedError

  def edit_network_use_same_static_padding(self, net_dict: Dict):
    raise NotImplementedError

  def add_align_augment(self, net_dict, networks_dict, python_prolog):
    raise NotImplementedError

  def edit_network_freeze_layers_excluding(
          self,
          net_dict: Dict,
          layers_to_exclude: List[str],
  ):
    if "class" in net_dict:
      net_dict["trainable"] = False

    for item in net_dict:
      if isinstance(net_dict[item], dict):
        if item in layers_to_exclude:
          continue
        self.edit_network_freeze_layers_excluding(net_dict[item], layers_to_exclude)

  def edit_network_freeze_layers_including(
          self,
          net_dict: Dict,
          layers_to_include: List[str],
          prefix_to_include: str = "",
  ):
    if "class" in net_dict:
      net_dict["trainable"] = False

    for item in net_dict:
      if isinstance(net_dict[item], dict):
        if item in layers_to_include or item.startswith(prefix_to_include):
          self.edit_network_freeze_layers_including(net_dict[item], layers_to_include, prefix_to_include)

  def get_recog_checkpoints(
          self, model_dir: Path, learning_rates: Path, key: str, checkpoints: Dict[int, Checkpoint], n_epochs: int):
    # last checkpoint
    last_checkpoint = checkpoints[n_epochs]

    # best checkpoint
    best_checkpoint = GetBestTFCheckpointJob(
      model_dir=model_dir, learning_rates=learning_rates, key=key, index=0
    ).out_checkpoint

    # avg checkpoint
    best_n = 4
    best_epochs = []
    for i in range(best_n):
      best_epochs.append(GetBestEpochJob(
        model_dir=model_dir,
        learning_rates=learning_rates,
        key=key,
        index=i
      ).out_epoch)
    best_avg_checkpoint = AverageTFCheckpointsJob(
      model_dir=model_dir,
      epochs=best_epochs,
      returnn_python_exe=self.variant_params["returnn_python_exe"],
      returnn_root=self.variant_params["returnn_root"]
    ).out_checkpoint

    checkpoints = {"last": last_checkpoint, "best": best_checkpoint, "best-4-avg": best_avg_checkpoint}

    return checkpoints

  def get_lr_settings(self, lr_opts, python_epilog: Optional[List] = None):
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
    elif lr_opts["type"] == "dynamic_lr":
      assert python_epilog is not None, "python_epilog must be provided to insert dynamic_learning_rate function"
      python_epilog.append(dynamic_lr_str.format(**lr_opts["dynamic_lr_opts"]))
    elif lr_opts["type"] == "const":
      const_lr = lr_opts["const_lr"]
      lr_settings.update({
        "learning_rate": const_lr,
      })
    else:
      raise NotImplementedError

    return lr_settings

  @staticmethod
  def get_default_cleanup_old_models():
    return {
      "keep_best_n": 4,
      "keep_last_n": 1,
      "keep": []
    }

  @staticmethod
  def get_default_beam_size():
    return 12

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
    oggzip_paths = dataset_opts.get("oggzip_paths", self.variant_params["dataset"]["corpus"].oggzip_paths)
    if self.variant_params["dataset"]["feature_type"] == "raw":
      return {
        "oggzip_path_list": oggzip_paths[corpus_key],
        "bpe_file": self.dependencies.bpe_codes_path,
        "vocab_file": self.dependencies.vocab_path,
        "segment_file": segment_paths.get(corpus_key, None),
        "hdf_targets": hdf_targets.get(corpus_key, None),
        "peak_normalization": dataset_opts.get("peak_normalization", True),
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
        pre_process=CodeWrapper("speed_pert") if dataset_opts.get("use_speed_pert") else None,
        seq_ordering=self.variant_params["config"]["train_seq_ordering"],
        epoch_wise_filter=dataset_opts.get("epoch_wise_filter", None),
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

  def get_extern_data_dict(self, dataset_opts: Dict):
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

    if dataset_opts.get("targets_dtype") is not None:
      extern_data_dict["targets"]["dtype"] = dataset_opts["targets_dtype"]

    return extern_data_dict

  @abstractmethod
  def get_net_dict(self, task: str, config_dict, python_prolog):
    pass

  @abstractmethod
  def get_networks_dict(self, task: str, config_dict, python_prolog, use_get_global_config: bool = False):
    pass

  def get_train_datasets(self, dataset_opts: Dict):
    return dict(
      extern_data=self.get_extern_data_dict(dataset_opts),
      train=self.get_train_dataset_dict(dataset_opts),
      dev=self.get_cv_dataset_dict(dataset_opts),
      eval_datasets={"devtrain": self.get_devtrain_dataset_dict(dataset_opts)}
    )

  def get_search_dataset(self, search_corpus_key: str, dataset_opts: Dict):
    return dict(
      extern_data=self.get_extern_data_dict(dataset_opts),
      search_data=self.get_search_dataset_dict(corpus_key=search_corpus_key, dataset_opts=dataset_opts)
    )

  def get_eval_dataset(self, eval_corpus_key: str, dataset_opts: Dict):
    return dict(
      extern_data=self.get_extern_data_dict(dataset_opts),
      eval=self.get_eval_dataset_dict(corpus_key=eval_corpus_key, dataset_opts=dataset_opts)
    )


class SWBBlstmConfigBuilder(ConfigBuilder, ABC):
  def get_default_batch_size(self):
    return 10000

  def get_final_net_dict(self, config_dict, python_prolog):
    return self.get_net_dict("train", config_dict=config_dict, python_prolog=python_prolog)

  def edit_network_freeze_encoder(self, net_dict: Dict):
    raise NotImplementedError


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

  def edit_network_freeze_encoder(self, net_dict: Dict):
    raise NotImplementedError


class LibrispeechConformerConfigBuilder(ConfigBuilder, ABC):
  def get_default_batch_size(self):
    return 2400000

  def get_default_lr_opts(self):
    raise NotImplementedError

  def edit_network_freeze_encoder(self, net_dict: Dict):
    self.edit_network_freeze_layers_including(
      net_dict,
      layers_to_include=["subsample_conv0", "subsample_conv1", "conv0", "source_linear"],
      prefix_to_include="conformer"
    )

  def edit_network_use_same_static_padding(self, net_dict: Dict):
    net_dict["subsample_conv0"]["padding"] = "same_static"
    net_dict["subsample_conv1"]["padding"] = "same_static"
