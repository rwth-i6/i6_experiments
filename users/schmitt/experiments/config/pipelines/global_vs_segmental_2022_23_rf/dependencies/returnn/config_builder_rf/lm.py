from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef, serialize_model_def
from i6_experiments.users.schmitt.returnn_frontend.utils.serialization import get_import_py_code
from i6_experiments.users.schmitt.datasets.bpe_lm import build_lm_training_datasets, LMDatasetSettings
from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict as get_oggzip_dataset_dict
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import ConfigBuilderRF

from i6_experiments.common.setups import serialization
from i6_experiments.common.setups.returnn.serialization import get_serializable_config

from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn.training import AverageTorchCheckpointsJob, GetBestEpochJob, Checkpoint, GetBestPtCheckpointJob
from i6_core.util import instanciate_delayed
from returnn.tf.updater import accum_grad_multiple_step

from returnn_common import nn

from sisyphus import Path

from abc import ABC
from typing import Dict, Optional, List, Callable
import copy
import numpy as np


class LmConfigBuilderRF(ABC):
  def __init__(
          self,
          variant_params: Dict,
          model_def: ModelDef,
          get_model_func: Callable,
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
      default_input="data",
      target="data",
      behavior_version=21,
    )

    self.python_prolog = []

    if variant_params["dependencies"].bpe_codes_path is None:
      # this means we use sentencepiece
      # append venv to path so that it finds the package (if it's not in the apptainer)
      self.python_prolog += [
        "import sys",
        "sys.path.append('/work/asr3/zeyer/schmitt/venvs/tf_env/lib/python3.10/site-packages')",
      ]

  def get_train_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.pop("dataset_opts", {})

    config_dict.update(self.get_train_datasets(dataset_opts=dataset_opts))
    extern_data_raw = self.get_extern_data_dict()
    extern_data_raw = instanciate_delayed(extern_data_raw)

    if opts.get("cleanup_old_models"):
      post_config_dict["cleanup_old_models"] = opts.pop("cleanup_old_models")

    config_dict["batch_size"] = opts.pop("batch_size", 5_000)

    config_dict.update(dict(
      accum_grad_multiple_step=2,
      batching="laplace:.1000",
      gradient_clip_global_norm=5.0,
      gradient_noise=0.0,
      grad_scaler=None,
      newbob_learning_rate_decay=0.8,
      newbob_relative_error_threshold=0,
      max_seq_length={"data": 75},
      max_seqs=100,
      calculate_exp_loss=True,
      pos_emb_dropout=0.1,
      rf_att_dropout_broadcast=False,
      optimizer={
        "class": "adamw",
        "epsilon": 1e-16,
        "weight_decay": 0.01,
        "weight_decay_modules_blacklist": [
          "rf.Embedding",
          "rf.LearnedRelativePositionalEncoding",
        ],
      },
    ))

    config_dict.update(ConfigBuilderRF.get_lr_settings(lr_opts=opts.pop("lr_opts"), python_epilog=python_epilog))

    train_def = opts.pop("train_def")
    train_step_func = opts.pop("train_step_func")

    serialization_list = [
      serialization.NonhashedCode(get_import_py_code()),
      serialization.NonhashedCode(nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)),
      *serialize_model_def(self.model_def), serialization.Import(self.get_model_func, import_as="get_model"),
      serialization.Import(train_def, import_as="_train_def", ignore_import_as_for_hash=True),
      serialization.Import(train_step_func, import_as="train_step"),
      serialization.PythonEnlargeStackWorkaroundNonhashedCode,
      serialization.PythonModelineNonhashedCode
    ]

    if opts.get("use_python_cache_manager", True):
      serialization_list.append(serialization.PythonCacheManagerFunctionNonhashedCode)

    python_epilog.append(
      serialization.Collection(serialization_list)
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
        dataset_opts=dataset_opts
      ))
    extern_data_raw = self.get_extern_data_dict(dataset_opts, config_dict)
    extern_data_raw = instanciate_delayed(extern_data_raw)

    config_dict["batch_size"] = opts.get("batch_size", 15_000) * self.batch_size_factor

    config_dict["beam_search_opts"] = {
      "beam_size": opts.get("beam_size", 12),
    }

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

  def get_search_dataset_dict(self, corpus_key: str, dataset_opts: Dict):
    raise NotImplementedError

  def get_extern_data_dict(self):
    from returnn.tensor import Dim, batch_dim

    extern_data_dict = {}

    out_spatial_dim = Dim(description="out_spatial", dimension=None, kind=Dim.Types.Spatial)

    extern_data_dict["data"] = {"dim_tags": [batch_dim, out_spatial_dim]}

    non_blank_target_dimension = self.variant_params["dependencies"].model_hyperparameters.target_num_labels_wo_blank
    non_blank_target_dim = Dim(description="non_blank_target_dim", dimension=non_blank_target_dimension, kind=Dim.Types.Spatial)
    extern_data_dict["data"]["sparse_dim"] = non_blank_target_dim

    return extern_data_dict

  def get_train_datasets(self, dataset_opts: Dict):
    train_data = build_lm_training_datasets(
      prefix="lm_train_data",
      librispeech_key="train-other-960",
      bpe_size=1000,
      settings=LMDatasetSettings(
        train_partition_epoch=20,
        train_seq_ordering="laplace:.1000",
      )
    )

    cv_data = get_oggzip_dataset_dict(
        fixed_random_subset=None,
        partition_epoch=1,
        pre_process=None,
        seq_ordering="sorted_reverse",
        epoch_wise_filter=None,
        seq_postfix=dataset_opts.get("seq_postfix", self.variant_params["dependencies"].model_hyperparameters.sos_idx),
        oggzip_path_list=self.variant_params["dataset"]["corpus"].oggzip_paths["cv"],
        segment_file=self.variant_params["dependencies"].segment_paths.get("cv", None),
        bpe_file=self.variant_params["dependencies"].bpe_codes_path,
        vocab_file=self.variant_params["dependencies"].vocab_path,
      )

    devtrain_data = get_oggzip_dataset_dict(
        fixed_random_subset=None,
        partition_epoch=1,
        pre_process=None,
        seq_ordering="sorted_reverse",
        epoch_wise_filter=None,
        seq_postfix=dataset_opts.get("seq_postfix", self.variant_params["dependencies"].model_hyperparameters.sos_idx),
        oggzip_path_list=self.variant_params["dataset"]["corpus"].oggzip_paths["devtrain"],
        segment_file=self.variant_params["dependencies"].segment_paths.get("devtrain", None),
        bpe_file=self.variant_params["dependencies"].bpe_codes_path,
        vocab_file=self.variant_params["dependencies"].vocab_path,
      )

    datasets = dict(
      train=train_data.train.as_returnn_opts(),
      dev=cv_data,
      devtrain=devtrain_data
      # dev=train_data.cv.as_returnn_opts(),
      # devtrain=train_data.devtrain.as_returnn_opts()
    )

    return datasets

  def get_search_dataset(self, dataset_opts: Dict):
    raise NotImplementedError


  def get_vocab_dict_for_tensor(self):
    if self.variant_params["dependencies"].bpe_codes_path is None:
      return {
        "model_file": self.variant_params["dependencies"].model_path,
        "class": "SentencePieces",
      }
    else:
      return {
        "bpe_file": self.variant_params["dependencies"].bpe_codes_path,
        "vocab_file": self.variant_params["dependencies"].vocab_path,
        "unknown_label": None,
        "bos_label": self.variant_params["dependencies"].model_hyperparameters.sos_idx,
        "eos_label": self.variant_params["dependencies"].model_hyperparameters.sos_idx,
      }


class LibrispeechLstmLmConfigBuilderRF(LmConfigBuilderRF, ABC):
  pass


class LibrispeechTrafoLmConfigBuilderRF(LmConfigBuilderRF, ABC):
  pass
