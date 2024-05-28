from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict as get_oggzip_dataset_dict
from i6_experiments.users.schmitt.datasets.concat import get_concat_dataset_dict
from i6_experiments.users.schmitt.datasets.extern_sprint import get_dataset_dict as get_extern_sprint_dataset_dict
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt import dynamic_lr
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef, serialize_model_def
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef
from i6_experiments.users.schmitt.returnn_frontend.utils.serialization import get_import_py_code

from i6_experiments.common.setups import serialization
from i6_experiments.common.setups.returnn.serialization import get_serializable_config

from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn.training import AverageTorchCheckpointsJob, GetBestEpochJob, Checkpoint, GetBestPtCheckpointJob
from i6_core.util import instanciate_delayed

from returnn_common import nn

from sisyphus import Path

from abc import ABC
from typing import Dict, Optional, List, Callable
import copy
import numpy as np


class ConfigBuilderRF(ABC):
  def __init__(
          self,
          variant_params: Dict,
          model_def: ModelDef,
          get_model_func: Callable,
          use_att_ctx_in_state: bool = True,
  ):
    self.variant_params = variant_params
    self.model_def = model_def
    self.get_model_func = get_model_func

    self.post_config_dict = dict(
      torch_dataloader_opts={"num_workers": 1},
    )

    self.python_epilog = []

    self.config_dict = dict(
      backend="torch",
      log_batch_size=True,
      torch_log_memory_usage=True,
      debug_print_layer_output_template=True,
      max_seqs=200,
      optimizer={"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-6},
      accum_grad_multiple_step=4,
      default_input="data",
      target="targets",
    )

    self.use_att_ctx_in_state = use_att_ctx_in_state
    if not use_att_ctx_in_state:
      self.config_dict["use_att_ctx_in_state"] = use_att_ctx_in_state

    self.python_prolog = []

  def get_train_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.pop("dataset_opts", {})
    config_dict.update(self.get_train_datasets(dataset_opts=dataset_opts))
    extern_data_raw = self.get_extern_data_dict(dataset_opts)
    extern_data_raw = instanciate_delayed(extern_data_raw)

    if dataset_opts.pop("use_speed_pert", None):
      python_prolog += [
        "import sys",
        'sys.path.append("/work/asr4/zeineldeen/py_envs/py_3.10_tf_2.9/lib/python3.10/site-packages")'
      ]
      config_dict["speed_pert"] = speed_pert

    if opts.get("cleanup_old_models"):
      post_config_dict["cleanup_old_models"] = opts.pop("cleanup_old_models")

    config_dict.update(self.get_lr_settings(lr_opts=opts.pop("lr_opts"), python_epilog=python_epilog))
    config_dict["batch_size"] = opts.pop("batch_size", 15_000) * self.batch_size_factor

    train_def = opts.pop("train_def")
    train_step_func = opts.pop("train_step_func")

    remaining_opt_keys = [
      "aux_loss_layers", "preload_from_files", "accum_grad_multiple_step", "optimizer", "batching",
      "torch_distributed", "pos_emb_dropout", "rf_att_dropout_broadcast", "grad_scaler", "gradient_clip_global_norm",
      "spec_augment_steps", "torch_amp"
    ]
    config_dict.update(
      {k: opts.pop(k) for k in remaining_opt_keys if k in opts}
    )

    python_epilog.append(
      serialization.Collection(
        [
          serialization.NonhashedCode(get_import_py_code()),
          serialization.NonhashedCode(
            nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
          ),
          *serialize_model_def(self.model_def),
          serialization.Import(self.get_model_func, import_as="get_model"),
          serialization.Import(train_def, import_as="_train_def", ignore_import_as_for_hash=True),
          serialization.Import(train_step_func, import_as="train_step"),
          serialization.PythonEnlargeStackWorkaroundNonhashedCode,
          serialization.PythonCacheManagerFunctionNonhashedCode,
          serialization.PythonModelineNonhashedCode
        ]
      )
    )

    returnn_train_config = ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
    )

    # serialize remaining functions, e.g. dynamic learning rate
    return get_serializable_config(returnn_train_config, serialize_dim_tags=False)

  def get_recog_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.get("dataset_opts", {})
    config_dict.update(dict(
      task="forward",
      search_output_layer="decision",
      batching=opts.get("batching", "random")
    ))

    config_dict.update(
      self.get_search_dataset(
        search_corpus_key=opts["search_corpus_key"],
        dataset_opts=dataset_opts
      ))
    extern_data_raw = self.get_extern_data_dict(dataset_opts)
    extern_data_raw = instanciate_delayed(extern_data_raw)

    config_dict["batch_size"] = opts.get("batch_size", 15_000) * self.batch_size_factor

    python_epilog.append(
      serialization.Collection(
        [
          serialization.NonhashedCode(get_import_py_code()),
          serialization.NonhashedCode(
            nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
          ),
          *serialize_model_def(self.model_def),
          serialization.Import(self.get_model_func, import_as="get_model"),
          serialization.Import(opts["recog_def"], import_as="_recog_def", ignore_import_as_for_hash=True),
          serialization.Import(opts["forward_step_func"], import_as="forward_step"),
          serialization.Import(opts["forward_callback"], import_as="forward_callback"),
          serialization.PythonEnlargeStackWorkaroundNonhashedCode,
          serialization.PythonCacheManagerFunctionNonhashedCode,
          serialization.PythonModelineNonhashedCode
        ]
      )
    )

    returnn_train_config = ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
    )

    # serialize remaining functions, e.g. dynamic learning rate
    return get_serializable_config(returnn_train_config, serialize_dim_tags=False)


  def get_recog_checkpoints(
          self, model_dir: Path, learning_rates: Path, key: str, checkpoints: Dict[int, Checkpoint], n_epochs: int):
    # last checkpoint
    last_checkpoint = checkpoints[n_epochs]

    # best checkpoint
    best_checkpoint = GetBestPtCheckpointJob(
      model_dir=model_dir, learning_rates=learning_rates, key=key, index=0
    ).out_checkpoint

    # avg checkpoint
    best_n = 4
    best_checkpoints = []
    for i in range(best_n):
      best_checkpoints.append(GetBestPtCheckpointJob(
        model_dir=model_dir, learning_rates=learning_rates, key=key, index=i
      ).out_checkpoint)
    best_avg_checkpoint = AverageTorchCheckpointsJob(
      checkpoints=best_checkpoints,
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
    elif lr_opts["type"] == "const_then_linear":
      const_lr = lr_opts["const_lr"]
      const_frac = lr_opts["const_frac"]
      final_lr = lr_opts["final_lr"]
      num_epochs = lr_opts["num_epochs"]
      lr_settings.update({
        "learning_rates": [const_lr] * int((num_epochs*const_frac)) + list(np.linspace(const_lr, final_lr, num_epochs - int((num_epochs*const_frac)))),
      })
    elif lr_opts["type"] == "dyn_lr_piecewise_linear":
      _lrlin_oclr_steps_by_bs_nep = {
        (8, 125): [139_000, 279_000, 310_000],  # ~2485steps/ep, 125 eps -> 310k steps in total
        (8, 250): [279_000, 558_000, 621_000],  # ~2485steps/ep, 250 eps -> 621k steps in total
        (8, 500): [558_000, 1_117_000, 1_242_000],  # ~2485steps/ep, 500 eps -> 1.242k steps in total
        (10, 500): [443_000, 887_000, 986_000],  # ~1973 steps/epoch, total steps after 500 epochs: ~986k
        (15, 500): [295_000, 590_000, 652_000],  # total steps after 500 epochs: ~652k
        (20, 1000): [438_000, 877_000, 974_000],  # total steps after 1000 epochs: 974.953
        (20, 2000): [878_000, 1_757_000, 1_952_000],  # total steps after 2000 epochs: 1.952.394
        (30, 2000): [587_000, 1_174_000, 1_305_000],  # total steps after 2000 epochs: 1.305.182
        (40, 2000): [450_000, 900_000, 982_000],  # total steps after 2000 epochs: 982.312
      }
      peak_lr = lr_opts.get("peak_lr", 1e-3)
      return dict(
        dynamic_learning_rate=dynamic_lr.dyn_lr_piecewise_linear,
        learning_rate=1.0,
        learning_rate_piecewise_steps=_lrlin_oclr_steps_by_bs_nep[(lr_opts["batch_size"] // 1000, lr_opts["num_epochs"])],
        learning_rate_piecewise_values=[peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3],
      )
    elif lr_opts["type"] == "dyn_lr_lin_warmup_invsqrt_decay":
      return dict(
        dynamic_learning_rate=dynamic_lr.dyn_lr_lin_warmup_invsqrt_decay,
        learning_rate_warmup_steps=lr_opts["learning_rate_warmup_steps"],
        learning_rate_invsqrt_norm=lr_opts["learning_rate_invsqrt_norm"],
        learning_rate=lr_opts["learning_rate"],
      )
    elif lr_opts["type"] == "const":
      const_lr = lr_opts["const_lr"]
      lr_settings.update({
        "learning_rate": const_lr,
      })
    else:
      raise NotImplementedError

    return lr_settings

  @staticmethod
  def get_default_beam_size():
    return 12

  def get_default_dataset_opts(self, corpus_key: str, dataset_opts: Dict):
    hdf_targets = dataset_opts.get("hdf_targets", self.variant_params["dependencies"].hdf_targets)
    segment_paths = dataset_opts.get("segment_paths", self.variant_params["dependencies"].segment_paths)
    oggzip_paths = dataset_opts.get("oggzip_paths", self.variant_params["dataset"]["corpus"].oggzip_paths)
    if self.variant_params["dataset"]["feature_type"] == "raw":
      return {
        "oggzip_path_list": oggzip_paths[corpus_key],
        "bpe_file": self.variant_params["dependencies"].bpe_codes_path,
        "vocab_file": self.variant_params["dependencies"].vocab_path,
        "segment_file": segment_paths.get(corpus_key, None),
        "hdf_targets": hdf_targets.get(corpus_key, None),
        "peak_normalization": dataset_opts.get("peak_normalization", True),
      }
    else:
      assert self.variant_params["dataset"]["feature_type"] == "gammatone"
      return {
        "rasr_config_path": self.variant_params["dataset"]["corpus"].rasr_feature_extraction_config_paths[corpus_key],
        "rasr_nn_trainer_exe": RasrExecutables.nn_trainer_path,
        "segment_path": self.variant_params["dependencies"].segment_paths[corpus_key],
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
        seq_postfix=dataset_opts.get("seq_postfix", self.variant_params["dependencies"].model_hyperparameters.sos_idx),
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
    from returnn.tensor import Dim, batch_dim

    extern_data_dict = {}
    time_dim = Dim(description="time", dimension=None, kind=Dim.Types.Spatial)
    if self.variant_params["dataset"]["feature_type"] == "raw":
      audio_dim = Dim(description="audio", dimension=1, kind=Dim.Types.Feature)
      extern_data_dict["data"] = {"dim_tags": [batch_dim, time_dim, audio_dim]}
    else:
      assert self.variant_params["dataset"]["feature_type"] == "gammatone"
      audio_dim = Dim(description="audio", dimension=40, kind=Dim.Types.Feature)
      extern_data_dict["data"] = {"dim_tags": [batch_dim, time_dim, audio_dim]}

    out_spatial_dim = Dim(description="out_spatial", dimension=None, kind=Dim.Types.Spatial)
    vocab_dim = Dim(
      description="vocab",
      dimension=self.variant_params["dependencies"].model_hyperparameters.target_num_labels,
      kind=Dim.Types.Spatial
    )
    extern_data_dict["targets"] = {
      "dim_tags": [batch_dim, out_spatial_dim],
      "sparse_dim": vocab_dim,
    }

    return extern_data_dict

  def get_train_datasets(self, dataset_opts: Dict):
    return dict(
      train=self.get_train_dataset_dict(dataset_opts),
      dev=self.get_cv_dataset_dict(dataset_opts),
      eval_datasets={"devtrain": self.get_devtrain_dataset_dict(dataset_opts)}
    )

  def get_search_dataset(self, search_corpus_key: str, dataset_opts: Dict):
    return dict(
      forward_data=self.get_search_dataset_dict(corpus_key=search_corpus_key, dataset_opts=dataset_opts)
    )

  def get_eval_dataset(self, eval_corpus_key: str, dataset_opts: Dict):
    return dict(
      extern_data=self.get_extern_data_dict(dataset_opts),
      eval=self.get_eval_dataset_dict(corpus_key=eval_corpus_key, dataset_opts=dataset_opts)
    )

  @property
  def batch_size_factor(self):
    raise NotImplementedError


class LibrispeechConformerConfigBuilderRF(ConfigBuilderRF, ABC):
  @property
  def batch_size_factor(self):
    return 160


class GlobalAttConfigBuilderRF(LibrispeechConformerConfigBuilderRF):
  def __init__(
          self,
          use_weight_feedback: bool = True,
          **kwargs
  ):
    super(GlobalAttConfigBuilderRF, self).__init__(**kwargs)

    self.config_dict.update(dict(
      max_seq_length_default_target=75,
    ))

    if not use_weight_feedback:
      self.config_dict["use_weight_feedback"] = use_weight_feedback

  def get_extern_data_dict(self, dataset_opts: Dict):
    extern_data_dict = super(GlobalAttConfigBuilderRF, self).get_extern_data_dict(dataset_opts)
    extern_data_dict["targets"]["vocab"] = {
      "bpe_file": self.variant_params["dependencies"].bpe_codes_path,
      "vocab_file": self.variant_params["dependencies"].vocab_path,
      "unknown_label": None,
      "bos_label": self.variant_params["dependencies"].model_hyperparameters.sos_idx,
      "eos_label": self.variant_params["dependencies"].model_hyperparameters.sos_idx,
    }

    return extern_data_dict


class SegmentalAttConfigBuilderRF(LibrispeechConformerConfigBuilderRF):
  def __init__(
          self,
          center_window_size: int,
          blank_decoder_version: Optional[int] = None,
          use_joint_model: bool = False,
          use_weight_feedback: bool = True,
          label_decoder_state: str = "nb-lstm",
          **kwargs
  ):
    super(SegmentalAttConfigBuilderRF, self).__init__(**kwargs)

    self.config_dict.update(dict(
      center_window_size=center_window_size,
    ))

    if use_joint_model:
      assert not blank_decoder_version, "Either use joint model or separate label and blank model"

    if blank_decoder_version is not None and blank_decoder_version != 1:
      self.config_dict["blank_decoder_version"] = blank_decoder_version
    if use_joint_model:
      self.config_dict["use_joint_model"] = use_joint_model
    if not use_weight_feedback:
      self.config_dict["use_weight_feedback"] = use_weight_feedback
    if label_decoder_state != "nb-lstm":
      self.config_dict["label_decoder_state"] = label_decoder_state

  def get_train_config(self, opts: Dict):
    train_config = super(SegmentalAttConfigBuilderRF, self).get_train_config(opts)

    if opts.get("alignment_augmentation_opts"):
      train_config.config["alignment_augmentation_opts"] = opts["alignment_augmentation_opts"]

    return train_config

  def get_recog_config(self, opts: Dict):
    recog_config = super(SegmentalAttConfigBuilderRF, self).get_recog_config(opts)

    recog_config.config["non_blank_vocab"] = {
      "bpe_file": self.variant_params["dependencies"].bpe_codes_path,
      "vocab_file": self.variant_params["dependencies"].vocab_path,
      "unknown_label": None,
      "bos_label": self.variant_params["dependencies"].model_hyperparameters.sos_idx,
      "eos_label": self.variant_params["dependencies"].model_hyperparameters.sos_idx,
    }

    return recog_config
