from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict as get_oggzip_dataset_dict
from i6_experiments.users.schmitt.datasets.concat import get_concat_dataset_dict
from i6_experiments.users.schmitt.datasets.variable import (
  get_interpolation_alignment_dataset, get_interpolation_alignment_scores_dataset, get_realignment_dataset
)
from i6_experiments.users.schmitt.datasets.extern_sprint import get_dataset_dict as get_extern_sprint_dataset_dict
from i6_experiments.users.schmitt.specaugment import speed_pert, cutoff_initial_silence, speed_pert_w_flip
from i6_experiments.users.schmitt import dynamic_lr
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef, serialize_model_def
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef
from i6_experiments.users.schmitt.returnn_frontend.utils.serialization import get_import_py_code

from i6_experiments.common.setups import serialization
from i6_experiments.common.setups.returnn.serialization import get_serializable_config
from i6_experiments.users.schmitt.custom_load_params import load_missing_params

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
          label_decoder_state: str = "nb-lstm",
          use_current_frame_in_readout: bool = False,
          use_current_frame_in_readout_w_gate: bool = False,
          use_current_frame_in_readout_random: bool = False,
          use_correct_dim_tags: bool = False,
          num_label_decoder_layers: int = 1,
          target_embed_dimension: int = 640,
          readout_dimension: int = 1024,
          ilm_dimension: int = 1024,
          conformer_w_abs_pos_enc: bool = False,
          conformer_wo_rel_pos_enc: bool = False,
          conformer_wo_final_layer_norm_per_layer: bool = False,
          conformer_num_layers: int = 12,
          conformer_wo_convolution: bool = False,
          conformer_conv_w_zero_padding: bool = False,
          conv_frontend_w_zero_padding: bool = False,
          conformer_out_dim: int = 512,
          use_trafo_att: bool = False,
          use_readout: bool = True,
  ):
    assert (use_current_frame_in_readout_random ^ use_current_frame_in_readout_w_gate) or (
                  use_current_frame_in_readout_random ^ use_current_frame_in_readout) or (
      not use_current_frame_in_readout_random and not use_current_frame_in_readout_w_gate and not use_current_frame_in_readout
    ), "Either none or exactly one can be true"


    self.variant_params = variant_params
    self.model_def = model_def
    self.get_model_func = get_model_func
    self.use_correct_dim_tags = use_correct_dim_tags

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

    self.label_decoder_state = label_decoder_state
    if label_decoder_state != "nb-lstm":
      self.config_dict["label_decoder_state"] = label_decoder_state

    self.use_current_frame_in_readout = use_current_frame_in_readout
    if use_current_frame_in_readout:
      self.config_dict["use_current_frame_in_readout"] = use_current_frame_in_readout

    if use_current_frame_in_readout_w_gate:
      self.config_dict["use_current_frame_in_readout_w_gate"] = use_current_frame_in_readout_w_gate

    if use_current_frame_in_readout_random:
      self.config_dict["use_current_frame_in_readout_random"] = use_current_frame_in_readout_random

    if num_label_decoder_layers != 1:
      self.config_dict["num_label_decoder_layers"] = num_label_decoder_layers

    if target_embed_dimension != 640:
      self.config_dict["target_embed_dim"] = target_embed_dimension
    if readout_dimension != 1024:
      self.config_dict["readout_dimension"] = readout_dimension
    if ilm_dimension != 1024:
      self.config_dict["ilm_dimension"] = ilm_dimension
    if not use_readout:
      self.config_dict["use_readout"] = False

    if conformer_w_abs_pos_enc:
      self.config_dict["conformer_w_abs_pos_enc"] = True
    if conformer_wo_rel_pos_enc:
      self.config_dict["conformer_wo_rel_pos_enc"] = True
    if conformer_wo_final_layer_norm_per_layer:
      self.config_dict["conformer_wo_final_layer_norm_per_layer"] = True
    if conformer_num_layers != 12:
      self.config_dict["conformer_num_layers"] = conformer_num_layers
    if conformer_wo_convolution:
      self.config_dict["conformer_wo_convolution"] = True
    if conformer_out_dim != 512:
      self.config_dict["conformer_out_dim"] = conformer_out_dim
    if conformer_conv_w_zero_padding:
      self.config_dict["conformer_conv_w_zero_padding"] = True
    if conv_frontend_w_zero_padding:
      self.config_dict["conv_frontend_w_zero_padding"] = True

    if use_trafo_att:
      self.config_dict["use_trafo_att"] = True

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

    if opts.get("full_sum_alignment_interpolation_factor", 0.0) > 0.0:
      dataset_opts["add_alignment_interpolation_datasets"] = True

    if opts.get("training_do_realignments", False):
      config_dict["train_partition_epoch"] = self.variant_params["dataset"]["corpus"].partition_epoch
      config_dict["training_do_realignments"] = True
      dataset_opts["add_realignment_dataset"] = True

    config_dict.update(self.get_train_datasets(dataset_opts=dataset_opts))
    extern_data_raw = self.get_extern_data_dict(dataset_opts, config_dict)
    extern_data_raw = instanciate_delayed(extern_data_raw)

    if dataset_opts.pop("use_speed_pert", None):
      if "import sys" not in python_prolog:
        python_prolog.append("import sys")
      python_prolog += [
        # "import sys",
        'sys.path.append("/work/asr4/zeineldeen/py_envs/py_3.10_tf_2.9/lib/python3.10/site-packages")'
      ]
      if dataset_opts.pop("use_speed_pert_w_flip", False):
        config_dict["speed_pert"] = speed_pert_w_flip
      else:
        config_dict["speed_pert"] = speed_pert

    if dataset_opts.pop("cutoff_initial_silence", None):
      config_dict["cutoff_initial_silence"] = cutoff_initial_silence

    if opts.get("cleanup_old_models"):
      post_config_dict["cleanup_old_models"] = opts.pop("cleanup_old_models")

    if opts.get("train_mini_lstm_opts") is not None:
      config_dict["use_mini_att"] = True

    config_dict.update(self.get_lr_settings(lr_opts=opts.pop("lr_opts"), python_epilog=python_epilog))
    config_dict["batch_size"] = opts.pop("batch_size", 15_000) * self.batch_size_factor

    train_def = opts.pop("train_def")
    train_step_func = opts.pop("train_step_func")

    remaining_opt_keys = [
      "aux_loss_layers",
      "aux_loss_type",
      "aux_loss_scales",
      "aux_loss_focal_loss_factors",
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

    # need to randomly init new input part of readout matrix
    if self.config_dict.get("use_current_frame_in_readout"):
      preload_dict = config_dict.get("preload_from_files", {}).get("pretrained_global_att_params", {})
      if preload_dict and "custom_missing_load_func" not in preload_dict:
        preload_dict["custom_missing_load_func"] = load_missing_params

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

    if dataset_opts.get("remove_targets_from_extern_data", False):
      # remove targets from extern_data and instead insert the dimension (integer) and the vocab separately
      # into the config dict. the corresponding dim_tags are then build in the get_model function.
      target_sparse_dim = extern_data_raw["targets"]["sparse_dim"]
      targets_vocab = extern_data_raw["targets"].get("vocab", None)
      del extern_data_raw["targets"]

      if isinstance(self, GlobalAttConfigBuilderRF):
        dim_tag_name = "non_blank_target_dimension"
      else:
        if dataset_opts["target_is_alignment"]:
          dim_tag_name = "align_target_dimension"
        else:
          dim_tag_name = "non_blank_target_dimension"

      config_dict[dim_tag_name] = target_sparse_dim.dimension
      if targets_vocab is not None:
        config_dict["vocab"] = targets_vocab

    config_dict["batch_size"] = opts.get("batch_size", 15_000) * self.batch_size_factor

    config_dict["beam_search_opts"] = {
      "beam_size": opts.get("beam_size", 12),
    }

    lm_opts = opts.get("lm_opts", None)  # type: Optional[Dict]
    if lm_opts is not None:
      assert lm_opts.get("type", "trafo") == "trafo"

      config_dict["external_lm"] = {
        "class": "TransformerDecoder",
        "vocab_dim": 10_025,
        "model_dim": 1024,
        "embed_dim": 128,
        "num_layers": 24,
        "decoder_layer_opts": {"self_att_opts": {"with_bias": False, "att_dropout_broadcast": False}},
        "input_embedding_scale": 1.0,
        "share_embedding": False,
        "logits_with_bias": True,
        "input_dropout": 0.1,
      }

      if "preload_from_files" not in config_dict:
        config_dict["preload_from_files"] = {}
      config_dict["preload_from_files"]["external_lm"] = {
        "filename": "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_experiments/users/schmitt/returnn_frontend/convert/checkpoint/ConvertTfCheckpointToRfPtJob.7haAE0Cx93dA/output/model/network.023.pt",
        "prefix": "language_model.",
        "ignore_missing": False,
      }

      config_dict["beam_search_opts"]["external_lm_scale"] = lm_opts["scale"]

    ilm_correction_opts = opts.get("ilm_correction_opts", None)
    if ilm_correction_opts is not None:
      assert ilm_correction_opts["type"] == "mini_att"

      config_dict["use_mini_att"] = True

      if "preload_from_files" not in config_dict:
        config_dict["preload_from_files"] = {}
      config_dict["preload_from_files"]["mini_lstm"] = {
        "filename": ilm_correction_opts["mini_att_checkpoint"],
        "prefix": "do_not_load_",
        "var_name_mapping": {
          layer: f"do_not_load_{layer}" for layer in (
            "label_decoder.mini_att.bias",
            "label_decoder.mini_att.weight",
            "label_decoder.mini_att_lstm.bias",
            "label_decoder.mini_att_lstm.rec_weight",
            "label_decoder.mini_att_lstm.ff_weight",
          )} if "lstm" in self.label_decoder_state else {
          layer: f"do_not_load_{layer}" for layer in (
            "label_decoder.mini_att.bias",
            "label_decoder.mini_att.weight",
            "label_decoder.mini_att_linear.bias",
            "label_decoder.mini_att_linear.weight",
          )
        }
      }

      config_dict["beam_search_opts"].update({
        "ilm_type": "mini_att",
        "ilm_correction_scale": ilm_correction_opts["scale"],
      })

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

  def get_dump_att_weight_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.get("dataset_opts", {})
    config_dict.update(dict(
      task="forward",
      batching=opts.get("batching", "random")
    ))

    config_dict.update(
      self.get_search_dataset(
        dataset_opts=dataset_opts
      ))
    extern_data_raw = self.get_extern_data_dict(dataset_opts, config_dict)
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
          serialization.Import(
            opts["dump_att_weight_def"], import_as="_dump_att_weight_def", ignore_import_as_for_hash=True),
          serialization.Import(opts["forward_step_func"], import_as="forward_step"),
          serialization.Import(opts["forward_callback"], import_as="forward_callback"),
          serialization.PythonEnlargeStackWorkaroundNonhashedCode,
          serialization.PythonCacheManagerFunctionNonhashedCode,
          serialization.PythonModelineNonhashedCode
        ]
      )
    )

    returnn_dump_att_weight_config = ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
    )

    # serialize remaining functions, e.g. dynamic learning rate
    return get_serializable_config(returnn_dump_att_weight_config, serialize_dim_tags=False)

  def get_analyze_gradients_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.get("dataset_opts", {})
    config_dict.update(dict(
      task="forward",
      batching=opts.get("batching", "random")
    ))

    if opts.get("plot_encoder_gradient_graph", False):
      config_dict["plot_encoder_gradient_graph"] = True
    if opts.get("plot_encoder_layers", False):
      config_dict["plot_encoder_layers"] = True
    if opts.get("plot_log_gradients", False):
      config_dict["plot_log_gradients"] = True

    if isinstance(self, CtcConfigBuilderRF):
      dataset_opts["seq_postfix"] = None

    config_dict.update(
      self.get_search_dataset(
        dataset_opts=dataset_opts
      ))
    extern_data_raw = self.get_extern_data_dict(dataset_opts, config_dict)
    extern_data_raw = instanciate_delayed(extern_data_raw)

    config_dict["batch_size"] = opts.get("batch_size", 15_000) * self.batch_size_factor

    config_dict.update({
      "ref_alignment_hdf": opts["ref_alignment_hdf"],
      "ref_alignment_vocab_path": opts["ref_alignment_vocab_path"],
      "ref_alignment_blank_idx": opts["ref_alignment_blank_idx"],
      "json_vocab_path": opts["json_vocab_path"],
    })

    python_epilog.append(
      serialization.Collection(
        [
          serialization.NonhashedCode(get_import_py_code()),
          serialization.NonhashedCode(
            nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
          ),
          *serialize_model_def(self.model_def),
          serialization.Import(self.get_model_func, import_as="get_model"),
          serialization.Import(
            opts["analyze_gradients_def"], import_as="_analyze_gradients_def", ignore_import_as_for_hash=True),
          serialization.Import(opts["forward_step_func"], import_as="forward_step"),
          serialization.Import(opts["forward_callback"], import_as="forward_callback"),
          serialization.PythonEnlargeStackWorkaroundNonhashedCode,
          serialization.PythonCacheManagerFunctionNonhashedCode,
          serialization.PythonModelineNonhashedCode
        ]
      )
    )

    returnn_analyze_gradients_config = ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
    )

    # serialize remaining functions, e.g. dynamic learning rate
    return get_serializable_config(returnn_analyze_gradients_config, serialize_dim_tags=False)

  def get_dump_gradients_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.get("dataset_opts", {})
    config_dict.update(dict(
      task="forward",
      batching=opts.get("batching", "random")
    ))

    if opts.get("input_layer_name", "encoder_input") != "encoder_input":
      config_dict["input_layer_name"] = opts["input_layer_name"]

    if isinstance(self, CtcConfigBuilderRF):
      dataset_opts["seq_postfix"] = None

    config_dict.update(
      self.get_search_dataset(
        dataset_opts=dataset_opts
      ))
    extern_data_raw = self.get_extern_data_dict(dataset_opts, config_dict)
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
          serialization.Import(
            opts["dump_gradients_def"], import_as="_dump_gradients_def", ignore_import_as_for_hash=True),
          serialization.Import(opts["forward_step_func"], import_as="forward_step"),
          serialization.Import(opts["forward_callback"], import_as="forward_callback"),
          serialization.PythonEnlargeStackWorkaroundNonhashedCode,
          serialization.PythonCacheManagerFunctionNonhashedCode,
          serialization.PythonModelineNonhashedCode
        ]
      )
    )

    returnn_dump_gradients_config = ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
    )

    # serialize remaining functions, e.g. dynamic learning rate
    return get_serializable_config(returnn_dump_gradients_config, serialize_dim_tags=False)

  def get_dump_self_att_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.get("dataset_opts", {})
    config_dict.update(dict(
      task="forward",
      batching=opts.get("batching", "random")
    ))

    if isinstance(self, CtcConfigBuilderRF):
      dataset_opts["seq_postfix"] = None

    config_dict.update(
      self.get_search_dataset(
        dataset_opts=dataset_opts
      ))
    extern_data_raw = self.get_extern_data_dict(dataset_opts, config_dict)
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
          serialization.Import(
            opts["dump_self_att_def"], import_as="_dump_self_att_def", ignore_import_as_for_hash=True),
          serialization.Import(opts["forward_step_func"], import_as="forward_step"),
          serialization.Import(opts["forward_callback"], import_as="forward_callback"),
          serialization.PythonEnlargeStackWorkaroundNonhashedCode,
          serialization.PythonCacheManagerFunctionNonhashedCode,
          serialization.PythonModelineNonhashedCode
        ]
      )
    )

    returnn_self_att_config = ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
    )

    # serialize remaining functions, e.g. dynamic learning rate
    return get_serializable_config(returnn_self_att_config, serialize_dim_tags=False)

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

  def get_lrlin_oclr_steps_by_bs_nep(self):
    # By batch size (in k) and num (sub)epochs.
    # 500 subepochs is usually for multi-GPU with 4 GPUs,
    # i.e. the same as single-GPU 2000 subepochs.
    # If the dict is missing some entry,
    # unfortunately there is currently no good automatic way to get the number.
    # I need to log at the stats of some setup with this batch size.
    # I just run some setup with some arbitrary LR scheduling (calling it "wrongLr"),
    # or maybe with sqrt-decay, and then look at the stats (steps/ep, or total num steps),
    # and give some estimates for the steps here, i.e. 45%, 90%, almost 100%,
    # making sure the last number is slightly below the real total number of steps.
    return {
      (3, 125): [485_156, 970_312, 1_078_000],  # ~8625steps/ep, 125 eps -> 1,078,125 steps in total
      (3, 500): [1_940_625, 3_881_250, 4_312_000],  # ~8625steps/ep, 500 eps -> 4,312,500 steps in total
      (5, 500): [887_000, 1_774_000, 1_972_000],  # ~8625steps/ep, 500 eps -> 4,312,500 steps in total
      (6, 500): [970_000, 1_940_000, 2_156_000],  # ~8625steps/ep, 500 eps -> 4,312,500 steps in total
      (8, 125): [139_000, 279_000, 310_000],  # ~2485steps/ep, 125 eps -> 310k steps in total
      (8, 250): [279_000, 558_000, 621_000],  # ~2485steps/ep, 250 eps -> 621k steps in total
      (8, 500): [558_000, 1_117_000, 1_242_000],  # ~2485steps/ep, 500 eps -> 1.242k steps in total
      (10, 500): [443_000, 887_000, 986_000],  # ~1973 steps/epoch, total steps after 500 epochs: ~986k
      (15, 150): [88_000, 176_000, 196_000],  # ~1304 steps/epoch, total steps after 150 epochs: ~196k
      (15, 500): [295_000, 590_000, 652_000],  # total steps after 500 epochs: ~652k
      (15, 600): [352_000, 705_000, 783_000],  # total steps after 500 epochs: ~783k
      (20, 1000): [438_000, 877_000, 974_000],  # total steps after 1000 epochs: 974.953
      (20, 2000): [878_000, 1_757_000, 1_952_000],  # total steps after 2000 epochs: 1.952.394
      (30, 2000): [587_000, 1_174_000, 1_305_000],  # total steps after 2000 epochs: 1.305.182
      (40, 2000): [450_000, 900_000, 982_000],  # total steps after 2000 epochs: 982.312
    }

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
      _lrlin_oclr_steps_by_bs_nep = self.get_lrlin_oclr_steps_by_bs_nep()
      peak_lr = lr_opts.get("peak_lr", 1e-3)
      return dict(
        dynamic_learning_rate=dynamic_lr.dyn_lr_piecewise_linear,
        learning_rate=1.0,
        learning_rate_piecewise_steps=_lrlin_oclr_steps_by_bs_nep[(lr_opts["batch_size"] // 1000, lr_opts["num_epochs"])],
        learning_rate_piecewise_values=[peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3],
      )
    elif lr_opts["type"] == "dyn_lr_piecewise_linear_epoch-wise_v1":
      peak_lr = lr_opts.get("peak_lr", 1e-3)
      initial_lr = peak_lr / 10
      cyc_ep = int(0.45 * lr_opts["num_epochs"])
      return dict(
        learning_rates=list(
          np.linspace(initial_lr, peak_lr, cyc_ep)  # go up
        ) + list(
            np.linspace(peak_lr, initial_lr, cyc_ep)  # go down
        ) + list(
          np.linspace(initial_lr, 1e-6, lr_opts["num_epochs"] - 2 * cyc_ep)  # cool down
        )
      )
    elif lr_opts["type"] == "dyn_lr_piecewise_linear_epoch-wise_v2":
      peak_lr = lr_opts.get("peak_lr", 1e-3)
      initial_lr = peak_lr * 1e-2
      final_lr = peak_lr * 1e-3
      cyc_ep = int(0.45 * lr_opts["num_epochs"])
      return dict(
        learning_rates=list(
          np.linspace(initial_lr, peak_lr, cyc_ep)  # go up
        ) + list(
            np.linspace(peak_lr, initial_lr, cyc_ep)  # go down
        ) + list(
          np.linspace(initial_lr, final_lr, lr_opts["num_epochs"] - 2 * cyc_ep)  # cool down
        )
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
    elif lr_opts["type"] == "list":
      lr_settings.update({
        "learning_rates": lr_opts["learning_rates"],
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
      opts = {
        "oggzip_path_list": oggzip_paths[corpus_key],
        "segment_file": segment_paths.get(corpus_key, None),
        "hdf_targets": hdf_targets.get(corpus_key, None),
        "peak_normalization": dataset_opts.get("peak_normalization", True),
      }

      if self.variant_params["dependencies"].bpe_codes_path is None:
        opts.update({
          "model_file": self.variant_params["dependencies"].model_path,
          "bpe_file": None,
          "vocab_file": None,
        })
      else:
        opts.update({
          "bpe_file": self.variant_params["dependencies"].bpe_codes_path,
          "vocab_file": self.variant_params["dependencies"].vocab_path,
          "model_file": None,
        })

      return opts
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
        post_process=CodeWrapper("cutoff_initial_silence") if dataset_opts.get("cutoff_initial_silence") else None,
        seq_ordering=self.variant_params["config"]["train_seq_ordering"],
        epoch_wise_filter=dataset_opts.get("epoch_wise_filter", None),
        seq_postfix=dataset_opts.get("seq_postfix", self.variant_params["dependencies"].model_hyperparameters.sos_idx),
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
        seq_postfix=dataset_opts.get("seq_postfix", self.variant_params["dependencies"].model_hyperparameters.sos_idx),
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
        seq_postfix=dataset_opts.get("seq_postfix", self.variant_params["dependencies"].model_hyperparameters.sos_idx),
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
        partition_epoch=dataset_opts.get("partition_epoch", 1),
        pre_process=None,
        seq_ordering="sorted_reverse",
        epoch_wise_filter=None,
        seq_postfix=dataset_opts.get("seq_postfix", self.variant_params["dependencies"].model_hyperparameters.sos_idx),
        **self.get_default_dataset_opts(corpus_key, dataset_opts)
      )

      concat_num = dataset_opts.get("concat_num")  # type: Optional[int]
      if concat_num:
        if dataset_opts.get("concat_segment_paths"):
          seq_list_file = dataset_opts["concat_segment_paths"].get(corpus_key)
        else:
          seq_list_file = self.variant_params["dataset"]["corpus"].segment_paths[corpus_key + "_concat-%d" % concat_num]
        dataset_dict = get_concat_dataset_dict(
          original_dataset_dict=dataset_dict,
          seq_len_file=self.variant_params["dataset"]["corpus"].seq_len_files[corpus_key],
          seq_list_file=seq_list_file,
          remove_in_between_postfix=dataset_opts.get("remove_in_between_postfix", None),
          repeat_in_between_last_frame_up_to_multiple_of=dataset_opts.get("repeat_in_between_last_frame_up_to_multiple_of", None),
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

  def get_extern_data_dict(self, dataset_opts: Dict, config_dict: Dict):
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

    if self.use_correct_dim_tags:
      extern_data_dict["targets"] = {"dim_tags": [batch_dim, out_spatial_dim]}

      if isinstance(self, GlobalAttConfigBuilderRF):
        non_blank_target_dimension = self.variant_params["dependencies"].model_hyperparameters.target_num_labels_wo_blank
      else:
        if isinstance(self, SegmentalAttConfigBuilderRF) and self.use_joint_model:
          align_target_dimension = self.variant_params["dependencies"].model_hyperparameters.target_num_labels_wo_blank
          non_blank_target_dimension = self.variant_params["dependencies"].model_hyperparameters.target_num_labels_wo_blank
        else:
          align_target_dimension = self.variant_params["dependencies"].model_hyperparameters.target_num_labels
          non_blank_target_dimension = self.variant_params["dependencies"].model_hyperparameters.target_num_labels_wo_blank

        align_target_dim = Dim(description="align_target_dim", dimension=align_target_dimension, kind=Dim.Types.Spatial)

      non_blank_target_dim = Dim(description="non_blank_target_dim", dimension=non_blank_target_dimension, kind=Dim.Types.Spatial)
      if isinstance(self, GlobalAttConfigBuilderRF):
        extern_data_dict["targets"]["sparse_dim"] = non_blank_target_dim
      else:
        if dataset_opts["target_is_alignment"]:
          extern_data_dict["targets"]["sparse_dim"] = align_target_dim
        else:
          extern_data_dict["targets"]["sparse_dim"] = non_blank_target_dim
    else:
      if isinstance(self, SegmentalAttConfigBuilderRF) and self.use_joint_model:
        vocab_dimension = self.variant_params["dependencies"].model_hyperparameters.target_num_labels_wo_blank
      else:
        vocab_dimension = self.variant_params["dependencies"].model_hyperparameters.target_num_labels
      vocab_dim = Dim(
        description="vocab",
        dimension=vocab_dimension,
        kind=Dim.Types.Spatial
      )
      extern_data_dict["targets"] = {
        "dim_tags": [batch_dim, out_spatial_dim],
        "sparse_dim": vocab_dim,
      }

    if dataset_opts.get("add_alignment_interpolation_datasets"):
      score_dim = Dim(description="interpolation_alignment_score", dimension=1, kind=Dim.Types.Feature)
      interpolation_alignment_spatial_dim = Dim(
        description="interpolation_alignment_spatial", dimension=None, kind=Dim.Types.Spatial)
      extern_data_dict.update({
        "interpolation_alignment": {
          "dim_tags": [batch_dim, interpolation_alignment_spatial_dim], "sparse_dim": vocab_dim},
        "interpolation_alignment_scores": {"dim_tags": [batch_dim, score_dim]}
      })

    return extern_data_dict

  def get_dataset(self, dataset_opts: Dict, type_: str):
    if type_ == "train":
      dataset_dict = self.get_train_dataset_dict(dataset_opts)
    elif type_ == "cv":
      dataset_dict = self.get_cv_dataset_dict(dataset_opts)
    elif type_ == "devtrain":
      dataset_dict = self.get_devtrain_dataset_dict(dataset_opts)
    else:
      assert type_ == "search"
      dataset_dict = self.get_search_dataset_dict(dataset_opts["corpus_key"], dataset_opts)

    if dataset_opts.get("use_multi_proc", False):
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
      dev=self.get_dataset(dataset_opts, type_='cv'),
      eval_datasets={"devtrain": self.get_dataset(dataset_opts, type_='devtrain')}
    )

    if dataset_opts.get("add_alignment_interpolation_datasets"):
      for corpus_key, dataset_dict in datasets.items():
        if corpus_key == "eval_datasets":
          dataset_dict = dataset_dict["devtrain"]

        assert dataset_dict["class"] == "MetaDataset"
        assert set(dataset_dict["data_map"].keys()) == {"data", "targets"}
        dataset_dict["datasets"].update({
          "interpolation_alignment_dataset": {
            "class": "VariableDataset",
            "get_dataset": get_interpolation_alignment_dataset,
          },
          "interpolation_alignment_scores_dataset": {
            "class": "VariableDataset",
            "get_dataset": get_interpolation_alignment_scores_dataset,
          }
        })
        dataset_dict["data_map"].update({
          "interpolation_alignment": ("interpolation_alignment_dataset", "data"),
          "interpolation_alignment_scores": ("interpolation_alignment_scores_dataset", "data"),
        })

    if dataset_opts.get("add_realignment_dataset"):
      train_dataset_dict = datasets["train"]
      assert train_dataset_dict["class"] == "MetaDataset"
      assert set(train_dataset_dict["data_map"].keys()) == {"data", "targets"}
      realignment_dataset = {
        "class": "VariableDataset",
        "get_dataset": get_realignment_dataset,
      }
      train_dataset_dict["datasets"]["realignment"] = realignment_dataset
      train_dataset_dict["data_map"]["targets"] = ("realignment", "data")

    return datasets

  def get_search_dataset(self, dataset_opts: Dict):
    return dict(
      forward_data=self.get_dataset(dataset_opts=dataset_opts, type_='search')
    )

  def get_eval_dataset(self, eval_corpus_key: str, dataset_opts: Dict):
    raise NotImplementedError
    # return dict(
    #   extern_data=self.get_extern_data_dict(dataset_opts),
    #   eval=self.get_eval_dataset_dict(corpus_key=eval_corpus_key, dataset_opts=dataset_opts)
    # )

  @property
  def batch_size_factor(self):
    raise NotImplementedError

  @property
  def red_factor(self):
    raise NotImplementedError

  @property
  def red_subtrahend(self):
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


class LibrispeechConformerConfigBuilderRF(ConfigBuilderRF, ABC):
  @property
  def batch_size_factor(self):
    return 160

  @property
  def red_factor(self):
    return 960

  @property
  def red_subtrahend(self):
    return 399


class GlobalAttConfigBuilderRF(ConfigBuilderRF, ABC):
  def __init__(
          self,
          use_weight_feedback: bool = True,
          enc_ctx_layer: Optional[str] = None,
          use_feed_forward_encoder: bool = False,
          hard_att_opts: Optional[Dict] = None,
          **kwargs
  ):
    super(GlobalAttConfigBuilderRF, self).__init__(**kwargs)

    self.config_dict.update(dict(
      max_seq_length_default_target=75,
    ))

    if not use_weight_feedback:
      self.config_dict["use_weight_feedback"] = use_weight_feedback

    if enc_ctx_layer is not None:
      self.config_dict["enc_ctx_layer"] = enc_ctx_layer

    if use_feed_forward_encoder:
      self.config_dict["use_feed_forward_encoder"] = use_feed_forward_encoder

    if hard_att_opts is not None:
      self.config_dict["hard_att_opts"] = hard_att_opts

  def get_train_config(self, opts: Dict):
    train_config = super(GlobalAttConfigBuilderRF, self).get_train_config(opts)

    if opts.get("hard_att_opts", None) is not None:
      train_config.config["hard_att_opts"] = opts["hard_att_opts"]

    return train_config

  def get_extern_data_dict(self, dataset_opts: Dict, config_dict: Dict):
    extern_data_dict = super(GlobalAttConfigBuilderRF, self).get_extern_data_dict(dataset_opts, config_dict)

    vocab = self.get_vocab_dict_for_tensor()
    extern_data_dict["targets"]["vocab"] = vocab

    return extern_data_dict

  def get_dump_att_weight_config(self, opts: Dict):
    if "dataset_opts" not in opts:
      opts["dataset_opts"] = {}
    opts["dataset_opts"]["target_is_alignment"] = False

    return super(GlobalAttConfigBuilderRF, self).get_dump_att_weight_config(opts)

  def get_forward_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.get("dataset_opts", {})
    config_dict.update(dict(
      task="forward",
      batching=opts.get("batching", "random")
    ))

    config_dict.update(
      self.get_search_dataset(
        dataset_opts=dataset_opts
      ))
    extern_data_raw = self.get_extern_data_dict(dataset_opts, config_dict)
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
          serialization.Import(opts["forward_def"], import_as="_forward_def", ignore_import_as_for_hash=True),
          serialization.Import(opts["forward_step_func"], import_as="forward_step"),
          serialization.Import(opts["forward_callback"], import_as="forward_callback"),
          serialization.PythonEnlargeStackWorkaroundNonhashedCode,
          serialization.PythonCacheManagerFunctionNonhashedCode,
          serialization.PythonModelineNonhashedCode
        ]
      )
    )

    returnn_forward_config = ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
    )

    # serialize remaining functions, e.g. dynamic learning rate
    return get_serializable_config(returnn_forward_config, serialize_dim_tags=False)


class TransducerConfigBuilderRF(ConfigBuilderRF, ABC):
  def get_realign_config(self, opts: Dict):
    config_dict = copy.deepcopy(self.config_dict)
    post_config_dict = copy.deepcopy(self.post_config_dict)
    python_prolog = copy.deepcopy(self.python_prolog)
    python_epilog = copy.deepcopy(self.python_epilog)

    dataset_opts = opts.get("dataset_opts", {})
    dataset_opts["seq_postfix"] = None
    config_dict.update(dict(
      task="forward",
      batching=opts.get("batching", "random")
    ))

    if "preload_from_files" in opts:
      config_dict["preload_from_files"] = opts["preload_from_files"]

    config_dict.update(
      self.get_search_dataset(
        dataset_opts=dataset_opts
      ))
    extern_data_raw = self.get_extern_data_dict(dataset_opts, config_dict)
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
          serialization.Import(opts["realign_def"], import_as="_realign_def", ignore_import_as_for_hash=True),
          serialization.Import(opts["forward_step_func"], import_as="forward_step"),
          serialization.Import(opts["forward_callback"], import_as="forward_callback"),
          serialization.PythonEnlargeStackWorkaroundNonhashedCode,
          serialization.PythonCacheManagerFunctionNonhashedCode,
          serialization.PythonModelineNonhashedCode
        ]
      )
    )

    returnn_realign_config = ReturnnConfig(
      config=config_dict,
      post_config=post_config_dict,
      python_prolog=python_prolog,
      python_epilog=python_epilog,
    )

    # serialize remaining functions, e.g. dynamic learning rate
    return get_serializable_config(returnn_realign_config, serialize_dim_tags=False)


class SegmentalAttConfigBuilderRF(TransducerConfigBuilderRF, ABC):
  def __init__(
          self,
          center_window_size: int,
          blank_decoder_version: Optional[int] = None,
          use_joint_model: bool = False,
          use_weight_feedback: bool = True,
          gaussian_att_weight_opts: Optional[Dict] = None,
          separate_blank_from_softmax: bool = False,
          blank_decoder_opts: Optional[Dict] = None,
          **kwargs
  ):
    if use_joint_model:
      assert not blank_decoder_version, "Either use joint model or separate label and blank model"

    if blank_decoder_opts is not None:
      assert blank_decoder_version is not None
      assert blank_decoder_opts["version"] == blank_decoder_version

    super(SegmentalAttConfigBuilderRF, self).__init__(**kwargs)

    self.center_window_size = center_window_size
    self.config_dict.update(dict(
      center_window_size=center_window_size,
    ))
    self.use_joint_model = use_joint_model
    self.use_weight_feedback = use_weight_feedback

    if blank_decoder_version is not None and blank_decoder_version != 1:
      self.config_dict["blank_decoder_version"] = blank_decoder_version
    if use_joint_model:
      self.config_dict["use_joint_model"] = use_joint_model
    if not use_weight_feedback:
      self.config_dict["use_weight_feedback"] = use_weight_feedback
    if gaussian_att_weight_opts is not None:
      self.config_dict["gaussian_att_weight_opts"] = gaussian_att_weight_opts
    if separate_blank_from_softmax:
      self.config_dict["separate_blank_from_softmax"] = separate_blank_from_softmax

    if blank_decoder_opts is not None:
      self.config_dict["blank_decoder_opts"] = blank_decoder_opts

    self.reset_eos_params = False

  def get_train_config(self, opts: Dict):
    train_config = super(SegmentalAttConfigBuilderRF, self).get_train_config(opts)

    if opts.get("alignment_augmentation_opts"):
      train_config.config["alignment_augmentation_opts"] = opts["alignment_augmentation_opts"]

    remaining_opt_keys = [
      "full_sum_training_opts",
      "chunking",
      "min_chunk_size"
    ]
    train_config.config.update(
      {k: opts.pop(k) for k in remaining_opt_keys if k in opts}
    )

    nb_loss_scale = opts.pop("nb_loss_scale", 1.0)
    if nb_loss_scale and nb_loss_scale != 1.0:
      train_config.config["nb_loss_scale"] = nb_loss_scale
    b_loss_scale = opts.pop("b_loss_scale", 1.0)
    if b_loss_scale and b_loss_scale != 1.0:
      train_config.config["b_loss_scale"] = b_loss_scale

    if opts.pop("reset_eos_params", False):
      assert "preload_from_files" in train_config.config
      assert "pretrained_global_att_params" in train_config.config["preload_from_files"]
      preload_dict = train_config.config["preload_from_files"]["pretrained_global_att_params"]
      if "custom_missing_load_func" not in preload_dict:
        preload_dict["custom_missing_load_func"] = load_missing_params
      train_config.config["reset_eos_params"] = True

    return train_config

  def get_recog_config(self, opts: Dict):
    recog_config = super(SegmentalAttConfigBuilderRF, self).get_recog_config(opts)

    recog_config.config["non_blank_vocab"] = self.get_vocab_dict_for_tensor()

    use_recombination = opts.get("use_recombination")
    if use_recombination is not None:
      recog_config.config["beam_search_opts"]["use_recombination"] = use_recombination

    ilm_correction_opts = opts.get("ilm_correction_opts")  # type: Dict
    if ilm_correction_opts:
      recog_config.config["beam_search_opts"].update({
        "subtract_ilm_eos_score": True,
      })

    if opts.pop("reset_eos_params", False):
      recog_config.config["reset_eos_params"] = True
      # recog_config.config["preload_from_files"].update({
      #   "prefix": "do_not_load_",
      #   "var_name_mapping": {
      #     "label_decoder.target_embed": "do_not_load_label_decoder.target_embed_reset_eos",
      #     "label_decoder.output_prob.weight": "do_not_load_label_decoder.output_prob_reset_eos.weight",
      #     "label_decoder.output_prob.bias": "do_not_load_label_decoder.output_prob_reset_eos.bias",
      #   }
      #     layer: f"do_not_load_{layer}" for layer in (
      #       "label_decoder.target_embed_reset_eos",
      #       "label_decoder.output_prob_reset_eos",
      #     )
      #   }
      # })

    return recog_config

  def get_dump_att_weight_config(self, opts: Dict):
    if "dataset_opts" not in opts:
      opts["dataset_opts"] = {}
    opts["dataset_opts"]["target_is_alignment"] = True

    return super(SegmentalAttConfigBuilderRF, self).get_dump_att_weight_config(opts)


class CtcConfigBuilderRF(TransducerConfigBuilderRF, ABC):
  pass


class LibrispeechGlobalAttConformerConfigBuilderRF(LibrispeechConformerConfigBuilderRF, GlobalAttConfigBuilderRF, ABC):
  pass


class LibrispeechSegmentalAttConformerConfigBuilderRF(LibrispeechConformerConfigBuilderRF, SegmentalAttConfigBuilderRF, ABC):
  pass


class LibrispeechCtcAttConformerConfigBuilderRF(LibrispeechConformerConfigBuilderRF, CtcConfigBuilderRF, ABC):
  pass

