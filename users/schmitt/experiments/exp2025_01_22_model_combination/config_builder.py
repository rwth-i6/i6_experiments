from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict as get_oggzip_dataset_dict
from i6_experiments.users.schmitt.specaugment import speed_pert, cutoff_initial_silence, speed_pert_w_flip
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef, serialize_model_def
from i6_experiments.users.schmitt.returnn_frontend.utils.serialization import get_import_py_code
from i6_experiments.common.setups import serialization
from i6_experiments.common.setups.returnn.serialization import get_serializable_config
from .tools_paths import RETURNN_EXE, RETURNN_ROOT

from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn.training import AverageTorchCheckpointsJob, GetBestEpochJob, PtCheckpoint, GetBestPtCheckpointJob
from i6_core.util import instanciate_delayed

from returnn_common import nn
import returnn.frontend as rf

from sisyphus import Path

from abc import ABC
from typing import Dict, Optional, List, Callable
import copy
import numpy as np
import re


class ConfigBuilder:
  def __init__(
          self,
          dataset,
          vocab_opts: Dict,
          model_def: Callable,
          get_model_func: Callable,
          behavior_version: Optional[int] = None,
          feature_dimension: Optional[int] = None,
          feature_extraction: Optional[str] = "log-mel-on-the-fly",
          batch_size_factor: int = 160,
  ):
    self.dataset = dataset
    self.model_def = model_def
    self.get_model_func = get_model_func
    self.vocab_opts = vocab_opts
    self.feature_dimension = feature_dimension
    self.feature_extraction = feature_extraction
    self.batch_size_factor = batch_size_factor

    self.python_epilog = []

    self.config_dict = dict(
      backend="torch",
      max_seqs=200,
      default_input="data",
      target="targets",
      behavior_version=21,
    )
    self.post_config_dict = dict(
      log_batch_size=True,
      torch_log_memory_usage=True,
      debug_print_layer_output_template=True,
      torch_dataloader_opts={"num_workers": 1},
    )

    if behavior_version is not None:
      self.config_dict["behavior_version"] = behavior_version

    self.python_prolog = []

  def get_python_epilog_serialization(self, extern_data_raw, *extra_imports):
    return serialization.Collection(
      [
        serialization.NonhashedCode(get_import_py_code()),
        serialization.NonhashedCode(
          nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
        ),
        *serialize_model_def(self.model_def),
        serialization.Import(self.get_model_func, import_as="get_model"),
        *extra_imports,
        serialization.PythonEnlargeStackWorkaroundNonhashedCode,
        serialization.PythonCacheManagerFunctionNonhashedCode,
        serialization.PythonModelineNonhashedCode
      ]
    )

  def get_analyze_encoder_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.get("dataset_opts", {})
    config_dict.update(dict(
      task="forward",
      batching=opts.get("batching", "random")
    ))

    config_dict.update(dict(forward_data=self.get_dataset(dataset_opts=dataset_opts, type_='search')))
    extern_data_raw = self.get_extern_data_dict()
    extern_data_raw = instanciate_delayed(extern_data_raw)

    config_dict["batch_size"] = opts.get("batch_size", 15_000) * self.batch_size_factor

    config_dict.update({
      "ref_alignment_hdf": opts["ref_alignment_hdf"],
      "ref_alignment_vocab_path": opts["ref_alignment_vocab_path"],
      "ref_alignment_blank_idx": opts["ref_alignment_blank_idx"],
      "json_vocab_path": opts["json_vocab_path"],
    })

    python_epilog.append(
      self.get_python_epilog_serialization(
        extern_data_raw,
        serialization.Import(
          opts["analyze_encoder_def"], import_as="_analyze_encoder_def", ignore_import_as_for_hash=True),
        serialization.Import(opts["forward_step_func"], import_as="forward_step"),
        serialization.Import(opts["forward_callback"], import_as="forward_callback"),
      )
    )

    returnn_analyze_encoder_config = ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
    )

    # serialize remaining functions, e.g. dynamic learning rate
    return get_serializable_config(returnn_analyze_encoder_config, serialize_dim_tags=False)


  @staticmethod
  def get_lr_settings(lr_opts):
    if lr_opts["type"] == "dyn_lr_piecewise_linear_epoch-wise":
      peak_lr = lr_opts.get("peak_lr", 1e-3)
      initial_lr = lr_opts.get("init_lr", peak_lr * 1e-2)
      lr2 = lr_opts.get("lr2", initial_lr)
      final_lr = lr_opts.get("final_lr", peak_lr * 1e-3)
      cyc_ep = int(0.45 * lr_opts["num_epochs"])
      return dict(
        learning_rates=list(
          np.linspace(initial_lr, peak_lr, cyc_ep)  # go up
        ) + list(
            np.linspace(peak_lr, lr2, cyc_ep)  # go down
        ) + list(
          np.linspace(lr2, final_lr, lr_opts["num_epochs"] - 2 * cyc_ep)  # cool down
        )
      )
    else:
      raise NotImplementedError

  def get_default_dataset_opts(self, corpus_key: str, dataset_opts: Dict):
    raise NotImplementedError

  def get_train_dataset_dict(self, dataset_opts: Dict):
    raise NotImplementedError

  def get_eval_dataset_dict(self, corpus_key: str, dataset_opts: Dict):
    raise NotImplementedError

  def get_extern_data_dict(self):
    raise NotImplementedError

  def get_dataset(self, dataset_opts: Dict, type_: str):
    if type_ == "train":
      dataset_dict = self.get_train_dataset_dict(dataset_opts)
    elif type_ == "cv":
      dataset_dict = self.get_eval_dataset_dict("cv", dataset_opts)
    elif type_ == "devtrain":
      dataset_dict = self.get_eval_dataset_dict("devtrain", dataset_opts)
    else:
      assert type_ == "search"
      dataset_dict = self.get_eval_dataset_dict(dataset_opts["corpus_key"], dataset_opts)

    if dataset_opts.get("use_multi_proc", True):
      dataset_dict = {
        "class": "MultiProcDataset",
        "buffer_size": 10,
        "num_workers": 4,
        "dataset": dataset_dict
      }

    return dataset_dict

  def get_train_datasets(self, dataset_opts: Dict):
    datasets = dict(
      train=self.get_dataset(dataset_opts, type_='train'),
      eval_datasets={
        "devtrain": self.get_dataset(dataset_opts, type_='devtrain'),
        "dev": self.get_dataset(dataset_opts, type_='cv'),
      }
    )

    return datasets

  @staticmethod
  def get_vocab_dict_for_tensor(vocab_opts):
    return {
      "bpe_file": vocab_opts["bpe_codes_path"],
      "vocab_file": vocab_opts["vocab_path"],
      "unknown_label": None,
      "bos_label": vocab_opts["bos_idx"],
      "eos_label": vocab_opts["eos_idx"],
    }


class AEDConfigBuilder(ConfigBuilder):
  def get_train_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.pop("dataset_opts", {})

    config_dict.update(self.get_train_datasets(dataset_opts=dataset_opts))
    extern_data_raw = self.get_extern_data_dict()
    extern_data_raw = instanciate_delayed(extern_data_raw)

    if dataset_opts.pop("use_speed_pert", None):
      if "import sys" not in python_prolog:
        python_prolog.append("import sys")
      python_prolog += [
        'sys.path.append("/work/asr4/zeineldeen/py_envs/py_3.10_tf_2.9/lib/python3.10/site-packages")'
      ]
      config_dict["speed_pert"] = speed_pert

    if dataset_opts.pop("cutoff_initial_silence", None):
      config_dict["cutoff_initial_silence"] = cutoff_initial_silence

    if opts.get("hard_att_opts", None) is not None:
      config_dict["hard_att_opts"] = opts["hard_att_opts"]

    if opts.get("cleanup_old_models"):
      post_config_dict["cleanup_old_models"] = opts.pop("cleanup_old_models")

    config_dict.update(self.get_lr_settings(lr_opts=opts.pop("lr_opts")))
    config_dict["batch_size"] = opts.pop("batch_size", 15_000) * self.batch_size_factor

    train_def = opts.pop("train_def")
    train_step_func = opts.pop("train_step_func")

    remaining_opt_keys = [
      "aux_loss_layers",
      "preload_from_files",
      "accum_grad_multiple_step",
      "optimizer",
      "batching",
      "torch_distributed",
      "pos_emb_dropout",
      "rf_att_dropout_broadcast",
      "grad_scaler",
      "gradient_clip_global_norm",
      "specaugment_steps",
      "torch_amp",
      "gradient_clip",
      "gradient_noise",
      "max_seq_length",
      "weight_dropout",
      "att_dropout",
      "att_weight_dropout",
      "target_embed_dropout",
      "disable_enc_self_att_until_epoch",
      "random_seed",
    ]
    config_dict.update(
      {k: opts.pop(k) for k in remaining_opt_keys if k in opts}
    )

    python_epilog.append(
      self.get_python_epilog_serialization(
        extern_data_raw,
        serialization.Import(train_def, import_as="_train_def", ignore_import_as_for_hash=True),
        serialization.Import(train_step_func, import_as="train_step"),
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
    config_dict.update(dict(forward_data=self.get_dataset(dataset_opts=dataset_opts, type_='search')))
    extern_data_raw = self.get_extern_data_dict()
    extern_data_raw = instanciate_delayed(extern_data_raw)

    config_dict["batch_size"] = opts.get("batch_size", 15_000) * self.batch_size_factor

    config_dict["beam_search_opts"] = {
      "beam_size": opts.get("beam_size", 12),
    }

    behavior_version = opts.get("behavior_version")
    if behavior_version is not None:
      config_dict["behavior_version"] = behavior_version

    length_normalization_exponent = opts.get("length_normalization_exponent")
    if length_normalization_exponent != 1.0:
      config_dict["beam_search_opts"]["length_normalization_exponent"] = length_normalization_exponent

    python_epilog.append(
      self.get_python_epilog_serialization(
        extern_data_raw,
        serialization.Import(opts["recog_def"], import_as="_recog_def", ignore_import_as_for_hash=True),
        serialization.Import(opts["forward_step_func"], import_as="forward_step"),
        serialization.Import(opts["forward_callback"], import_as="forward_callback")
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

  def get_rescore_config(self, opts: Dict):
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
    config_dict.update(dict(forward_data=self.get_dataset(dataset_opts=dataset_opts, type_='search')))
    extern_data_raw = self.get_extern_data_dict()
    extern_data_raw = instanciate_delayed(extern_data_raw)

    config_dict["batch_size"] = opts.get("batch_size", 15_000) * self.batch_size_factor

    behavior_version = opts.get("behavior_version")
    if behavior_version is not None:
      config_dict["behavior_version"] = behavior_version

    python_epilog.append(
      self.get_python_epilog_serialization(
        extern_data_raw,
        serialization.Import(opts["rescore_def"], import_as="_rescore_def", ignore_import_as_for_hash=True),
        serialization.Import(opts["rescore_step_func"], import_as="forward_step"),
        serialization.Import(opts["forward_callback"], import_as="forward_callback")
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

  @staticmethod
  def get_recog_checkpoints(
          model_dir: Path, learning_rates: Path, key: str, checkpoints: Dict[int, PtCheckpoint]
  ) -> Dict[str, PtCheckpoint]:
    # last checkpoint
    last_checkpoint = checkpoints[max(checkpoints.keys())]

    # best checkpoint
    best_checkpoint = GetBestPtCheckpointJob(
      model_dir=model_dir, learning_rates=learning_rates, key=key, index=0
    ).out_checkpoint

    # avg checkpoint
    best_n = 4
    best_checkpoints = []
    for i in range(best_n):
      best_nth_chckpointjob = GetBestPtCheckpointJob(
        model_dir=model_dir, learning_rates=learning_rates, key=key, index=i
      )
      best_checkpoints.append(best_nth_chckpointjob.out_checkpoint)

    best_avg_checkpoint = AverageTorchCheckpointsJob(
      checkpoints=best_checkpoints,
      returnn_python_exe=RETURNN_EXE,
      returnn_root=RETURNN_ROOT,
    ).out_checkpoint

    checkpoints = {"last": last_checkpoint, "best": best_checkpoint, "best-4-avg": best_avg_checkpoint}

    return checkpoints

  def get_default_dataset_opts(self, corpus_key: str, dataset_opts: Dict):
    segment_paths = dataset_opts.get("segment_paths", self.dataset.segment_paths)
    oggzip_paths = dataset_opts.get("oggzip_paths", self.dataset.oggzip_paths)
    hdf_features = dataset_opts.get("hdf_features", {})
    seq_order_control_dataset = dataset_opts.get("seq_order_control_dataset", {})

    opts = {
      "oggzip_path_list": oggzip_paths[corpus_key],
      "segment_file": segment_paths.get(corpus_key, None),
      "hdf_targets": None,
      "peak_normalization": dataset_opts.get("peak_normalization", True),
      "hdf_features": hdf_features.get(corpus_key, None),
      "seq_order_control_dataset": seq_order_control_dataset.get(corpus_key, "zip_dataset"),
    }

    opts.update({
      "bpe_file": self.vocab_opts["bpe_codes_path"],
      "vocab_file": self.vocab_opts["vocab_path"],
      "model_file": None,
    })

    return opts

  def get_train_dataset_dict(self, dataset_opts: Dict):
    return get_oggzip_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=self.dataset.partition_epoch,
      pre_process=CodeWrapper("speed_pert") if dataset_opts.get("use_speed_pert") else None,
      post_process=CodeWrapper("cutoff_initial_silence") if dataset_opts.get("cutoff_initial_silence") else None,
      seq_ordering="laplace:.1000",
      epoch_wise_filter=dataset_opts.get("epoch_wise_filter", None),
      seq_postfix=dataset_opts.get("seq_postfix", self.vocab_opts["eos_idx"]),
      **self.get_default_dataset_opts("train", dataset_opts)
    )

  def get_eval_dataset_dict(self, corpus_key: str, dataset_opts: Dict):
    if corpus_key == "devtrain":
      fixed_random_subset = 3000
    else:
      fixed_random_subset = None

    return get_oggzip_dataset_dict(
      fixed_random_subset=fixed_random_subset,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      seq_postfix=dataset_opts.get("seq_postfix", self.vocab_opts["eos_idx"]),
      **self.get_default_dataset_opts(corpus_key, dataset_opts)
    )

  def get_extern_data_dict(self):
    from returnn.tensor import Dim, batch_dim

    extern_data_dict = {}
    time_dim = Dim(description="time", dimension=None, kind=Dim.Types.Spatial)
    if self.feature_extraction == "log-mel-on-the-fly":
      audio_dim = Dim(description="audio", dimension=1, kind=Dim.Types.Feature)
      extern_data_dict["data"] = {"dim_tags": [batch_dim, time_dim, audio_dim]}
    else:
      assert self.feature_extraction is None
      feature_dim = Dim(description="features", dimension=self.feature_dimension, kind=Dim.Types.Feature)
      extern_data_dict["data"] = {"dim_tags": [batch_dim, time_dim, feature_dim]}

    out_spatial_dim = Dim(description="out_spatial", dimension=None, kind=Dim.Types.Spatial)
    extern_data_dict["targets"] = {"dim_tags": [batch_dim, out_spatial_dim]}

    non_blank_target_dimension = self.vocab_opts["num_labels"]
    non_blank_target_dim = Dim(
      description="non_blank_target_dim", dimension=non_blank_target_dimension, kind=Dim.Types.Spatial)
    extern_data_dict["targets"]["sparse_dim"] = non_blank_target_dim
    vocab = self.get_vocab_dict_for_tensor(self.vocab_opts)
    extern_data_dict["targets"]["vocab"] = vocab

    return extern_data_dict


class TinaAlignmentModelConfigBuilder(ConfigBuilder):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.post_config_dict["torch_dataloader_opts"] = {"num_workers": 0}

  def get_eval_dataset_dict(self, corpus_key: str, dataset_opts: Dict):
    assert corpus_key == "train", f"corpus_key {corpus_key} not supported"

    segment_paths = dataset_opts.get("segment_paths", self.dataset.segment_paths)

    # we are only interested in forwarding the input
    # so we use the same dataset for data and targets
    sprint_cache_path = "/u/raissi/setups/librispeech/960h/work/i6_core/features/extraction/FeatureExtractionJob.Gammatone.XS6hEq8ovdgv/output/gt.cache.bundle"
    return {
      "class": "SprintCacheDataset",
      "data": {
          "data": {"filename": sprint_cache_path},
          "targets": {"filename": sprint_cache_path, "data_type": "feat"},
      },
      "seq_list_filter_file": segment_paths.get(corpus_key, None),
      "seq_ordering": "random"
    }

  def get_extern_data_dict(self):
    from returnn.tensor import Dim, batch_dim

    extern_data_dict = {}
    time_dim = Dim(description="time", dimension=None, kind=Dim.Types.Spatial)

    assert self.feature_extraction is None
    feature_dim = Dim(description="features", dimension=self.feature_dimension, kind=Dim.Types.Feature)
    extern_data_dict["data"] = {"dim_tags": [batch_dim, time_dim, feature_dim]}
    # set target dims to same as input dims
    # for now, we are just interested in forwarding the input
    extern_data_dict["targets"] = {"dim_tags": [batch_dim, time_dim, feature_dim]}

    # out_spatial_dim = Dim(description="out_spatial", dimension=None, kind=Dim.Types.Spatial)
    # extern_data_dict["targets"] = {"dim_tags": [batch_dim, out_spatial_dim]}
    #
    # target_dimension = self.vocab_opts["num_labels"]
    # target_dim = Dim(
    #   description="target_dim", dimension=target_dimension, kind=Dim.Types.Spatial)
    # extern_data_dict["targets"]["sparse_dim"] = target_dim
    # # vocab = self.get_vocab_dict_for_tensor(self.vocab_opts)
    # # extern_data_dict["targets"]["vocab"] = vocab

    return extern_data_dict
