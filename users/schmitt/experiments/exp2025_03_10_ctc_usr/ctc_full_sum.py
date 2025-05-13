"""
CTC experiments.
"""

from __future__ import annotations

import copy
import functools
import torch
import sys
import time
import warnings
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, Callable, Dict, Any, List
import numpy as np
import dataclasses
from dataclasses import dataclass

import returnn.frontend as rf
import returnn.torch.frontend as rtf
from returnn.tensor import Tensor, Dim, batch_dim, single_step_dim
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.tensor_array import TensorArray
from returnn.datasets.util.vocabulary import Vocabulary

from sisyphus import tk

from i6_experiments.users.schmitt.model_interfaces import ModelDef, ModelDefWithCfg, TrainDef, RecogDef
from i6_experiments.users.schmitt.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.schmitt.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.train import ExtendedTrainDef
from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.lm.n_gram import get_count_based_n_gram, \
  get_prior_from_unigram
from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.lm.ffnn import FeedForwardLm, get_ffnn_lm
from i6_experiments.users.schmitt.nn.util import DelayedCodeWrapper
from i6_experiments.common.setups import serialization

from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.ctc import (
  model_recog_lm,
  model_recog_flashlight,
  model_recog_lm_albert,
  ctc_model_def,
  model_recog,
  Model,
  ctc_training,
  _remove_eos_label_v2,
  ctc_sum_training,
  enc_conformer_layer_default
)

from i6_core.returnn.training import PtCheckpoint

from i6_core.util import uopen

from .configs import (
  _get_cfg_lrlin_oclr_by_bs_nep_v4,
  _batch_size_factor,
  config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
  dict_update_deep,
  post_config,
)

if TYPE_CHECKING:
  from i6_experiments.common.setups import serialization
  from i6_experiments.users.schmitt.model_with_checkpoints import ModelWithCheckpoints
  from i6_experiments.users.schmitt.datasets.task import Task
  from i6_experiments.users.schmitt.datasets.score_results import RecogOutput

OUT_BLANK_LABEL = "<blank>"
CHECK_DECODER_CONSISTENCY = False
_raw_sample_rate = _batch_size_factor * 100  # bs factor is from 10ms frames to raw samples

# Some params for the jobs influencing the hash
num_shards_recog = 4  # None, 4, 16
num_shards_recog_init = 4
num_shards_pseudo = 64  # 32, 64
num_shards_prior = 64
num_shards_prior_init = None
calculate_pseudo_label_scores = True
calculate_pseudo_label_scores_init = True
# TODO: confirm that i do not need this explicit "cf" cache manager
cache_manager = False  # True
exclude_epochs = True

# encoder modifications
model_opts = {"enc_conformer_layer": enc_conformer_layer_default, "feature_batch_norm": True}

# config for supervised pre-training
supervised_init_config_v1 = dict(
  vocab="bpe128",
  decoding_imp="albert-greedy",  # "albert-lm",
  batch_size=15_000 * _batch_size_factor,
  epochs=50,
  init_small=True,
  with_prior=True,
  empirical_prior=True,
  prior_from_max=False,
  aux_loss=True,
  # decoder_lm_config={"class": "FeedForwardLm", "context_size": 8},
  model_config=model_opts,
  # decoder_hyperparameters={"prior_weight": 0.3, "lm_weight": 0.3, "lm_order": "ffnn8"},
  decoder_hyperparameters=dict()
)
supervised_init_config_v2 = dict_update_deep(
  supervised_init_config_v1,
  {
    "init_w_w2v": True,
    "aux_loss": False,
    "model_config": {}
  },
)
supervised_init_configs = {
  # "v1": supervised_init_config_v1,
  "v2": supervised_init_config_v2,
}
supervised_checkpoints = {}

# config for full-sum training
full_sum_config_v1 = dict_update_deep(
  supervised_init_config_v1,
  {
    "epochs": 500,
    "train_language_model_opts": {"class": "ngram", "order": 2},
    "full_sum_opts": {
      "am_scale": 1.0,
      "lm_scale": 1.0,
      "prior_scale": 1.0,
      "horizontal_prior": True,
      "blank_prior": True,
      "prior_gradient": True,
      "max_prior": False,
      "top_k": 0,
      "alignment_topk": True,
      "blank_correction_version": 0,
      "correction_in_final_score": False,
      "print_gradients": False,
      "version": 2,
    }
  }
)
full_sum_config_v1["model_config"] = {"model_opts": model_opts}

full_sum_configs = [
  # # trainings with different am, lm, prior scales
  # # all scales 1.0: ctc loss goes up, full-sum down, number of greedy non-blanks towards 0
  # dict_update_deep(
  #   full_sum_config_v1,
  #   {
  #     "full_sum_opts.am_scale": am_scale,
  #     "full_sum_opts.lm_scale": lm_scale,
  #     "full_sum_opts.prior_scale": prior_scale,
  #   }
  # ) for am_scale, lm_scale, prior_scale in [
  #   (1.0, 1.0, 1.0)
  # ]
]


def py():
  """Sisyphus entry point"""

  for config_alias, config in supervised_init_configs.items():
    config_updates = {
      **_get_cfg_lrlin_oclr_by_bs_nep_v4(config["epochs"]),
      "batch_size": config["batch_size"],
      "optimizer.weight_decay": 1e-2,
      "__train_audio_preprocess": speed_pert_librosa_config,
      "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
      "max_seq_length_default_target": None,
      "max_seq_length_default_input": 19.5 * _raw_sample_rate,
      "init_w_w2v": config.get("init_w_w2v", False),
    }
    post_config_updates = {
      "cleanup_old_models": {
        "keep": [config["epochs"]],
        # "keep_last_n": 5,
        # "keep_best_n": 4
      }
    }

    # TODO: change alias
    alias_name = f"altLRedge1e-4-ctc-baseline_{config['epochs']}-ep"  # + \

    if not config["aux_loss"]:
      config_updates["aux_loss_layers"] = None

    if config.get("init_w_w2v", True):
      # preload_from_files = {
      #   "wav2vec2_base": {
      #     "filename": "/work/asr4/schmitt/sisyphus_work_dirs/2025_03_10_ctc_usr/i6_core/tools/download/DownloadJob.3y3hrN6raDKF/output/wav2vec2_base_no_finetune.pt",
      #     "ignore_missing": True,
      #     "init_for_train": True,
      #     "checkpoint_key": None,
      #   }
      # }

      alias_name += "_init_wv2ec"

      from i6_core.tools.download import DownloadJob
      wav2vec2_base_unsup_chkpt = DownloadJob(
        # "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
        "https://huggingface.co/facebook/wav2vec2-base/resolve/main/pytorch_model.bin?download=true",
        target_filename="wav2vec2_base_no_finetune.bin",
      ).out_file
      config_updates["preload_from_files"] = {
        "wav2vec2_base": {
          "filename": wav2vec2_base_unsup_chkpt,
          "ignore_missing": True,
        }
      }

      wav2vec_base_fine_tune_config = DownloadJob(
        "https://huggingface.co/facebook/wav2vec2-base/resolve/main/config.json?download=true",
        target_filename="wav2vec2_base_no_finetune_config.json",
      ).out_file
      config_updates["w2v_opts"] = {
        "config": wav2vec_base_fine_tune_config,
      }

    if config['decoding_imp'] in ["flashlight", "marten-greedy"]:
      decoder_def = model_recog_lm
    elif config['decoding_imp'] == "albert-greedy":
      decoder_def = model_recog
    elif config['decoding_imp'] == "albert-flashlight":
      decoder_def = model_recog_flashlight
    elif config['decoding_imp'] == "albert-lm":
      decoder_def = model_recog_lm_albert
    else:
      raise ValueError(f"Unknown decoder selection: {config['decoding_imp']}")

    epilog = [
      # resampy is needed for speed perturbation
      # sys.path.insert(...)
      serialization.ExternalImport(tk.Path("/work/asr3/zeyer/schmitt/venvs/resampy_package"))
    ]

    supervised_checkpoints[config_alias] = train_supervised_baseline(
      name=alias_name,
      config=config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
      decoder_def=decoder_def,
      decoder_hyperparameters=config['decoder_hyperparameters'],
      model_config=config['model_config'],
      config_updates=config_updates,
      vocab=config['vocab'],
      init_small=config['init_small'],
      with_prior=config['with_prior'],
      empirical_prior=config['empirical_prior'],
      prior_from_max=config['prior_from_max'],
      epilog=epilog,
      post_config_updates=post_config_updates
    )


  for config in full_sum_configs:
    config_updates = {
      **_get_cfg_lrlin_oclr_by_bs_nep_v4(config["epochs"]),
      "batch_size": config["batch_size"],
      "optimizer.weight_decay": 1e-2,
      "__train_audio_preprocess": speed_pert_librosa_config,
      "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
      "max_seq_length_default_target": None,
      "max_seq_length_default_input": 19.5 * _raw_sample_rate,
      "full_sum_opts": config["full_sum_opts"],
    }
    post_config_updates = {
      "cleanup_old_models": {
        "keep": [config["epochs"]],
        # "keep_last_n": 5,
        # "keep_best_n": 4
      }
    }

    # TODO: change alias
    alias_name = f"full-sum_altLRedge1e-4-ctc-baseline_{config['epochs']}-ep"  # + \

    if config['decoding_imp'] in ["flashlight", "marten-greedy"]:
      decoder_def = model_recog_lm
    elif config['decoding_imp'] == "albert-greedy":
      decoder_def = model_recog
    elif config['decoding_imp'] == "albert-flashlight":
      decoder_def = model_recog_flashlight
    elif config['decoding_imp'] == "albert-lm":
      decoder_def = model_recog_lm_albert
    else:
      raise ValueError(f"Unknown decoder selection: {config['decoding_imp']}")

    epilog = [
      # resampy is needed for speed perturbation
      # sys.path.insert(...)
      serialization.ExternalImport(tk.Path("/work/asr3/zeyer/schmitt/venvs/resampy_package"))
    ]

    train_full_sum(
      name=alias_name,
      config=config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
      model_config=config['model_config'],
      train_language_model_opts=config["train_language_model_opts"],
      config_updates=config_updates,
      vocab=config['vocab'],
      epilog=epilog,
      post_config_updates=post_config_updates,
      init_checkpoint=supervised_checkpoints["v1"].get_last_fixed_epoch().checkpoint
    )


_train_experiments: Dict[str, ModelWithCheckpoints] = {}


# noinspection PyShadowingNames
def train_supervised_baseline(
        name: str,
        config: Dict[str, Any],
        decoder_def: Callable,
        *,
        decoder_hyperparameters: dict = None,
        model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
        vocab: str = "bpe10k",
        train_vocab_opts: Optional[Dict[str, Any]] = None,
        train_def: Optional[TrainDef[Model]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        config_updates: Optional[Dict[str, Any]] = None,
        config_deletes: Optional[Sequence[str]] = None,
        post_config_updates: Optional[Dict[str, Any]] = None,
        epilog: Sequence[serialization.SerializerObject] = (),
        num_epochs: int = 2000,
        gpu_mem: Optional[int] = 24,
        num_processes: Optional[int] = None,
        time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
        env_updates: Optional[Dict[str, str]] = None,
        enabled: bool = True,
        self_training_rounds: int = 0,
        init_small: bool = False,
        with_prior: bool = False,
        empirical_prior: bool = False,
        prior_from_max: bool = False,
) -> ModelWithCheckpoints:
  """
  Train experiment
  """
  from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.train import train
  from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.recog import recog_training_exp
  from i6_experiments.users.schmitt.datasets.librispeech import get_librispeech_task_raw_v2, TrainDatasetSel

  if not enabled:
    return None

  if _sis_prefix is None:
    _sis_setup_global_prefix()

  prefix = _sis_prefix + "/" + name

  # get standard LS task, pseudo label dataset (e.g. 860h), and train_100h dataset (for supervised init)
  task, pseudo_labels_ds, train_100_ds = get_librispeech_task_raw_v2(
    vocab=vocab,
    train_vocab_opts=train_vocab_opts,
    save_pseudo_labels=None,
    ds_sel=TrainDatasetSel.train_100h if init_small else TrainDatasetSel.train_960h,
    init_small=init_small,
    with_prior=with_prior,
    empirical_prior=empirical_prior,
  )

  if with_prior and empirical_prior:
    emp_prior = get_prior_from_unigram(task.prior_dataset.vocab, task.prior_dataset, vocab)

  config = copy.deepcopy(config)
  config = dict_update_deep(config, config_updates, config_deletes)
  # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
  if "__train_audio_preprocess" in config:
    task: Task = copy.copy(task)
    task.train_dataset = copy.copy(task.train_dataset)
    task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

  if not model_def:
    model_def = ctc_model_def
  if model_config:
    mc = model_config.copy()
    model_def = ModelDefWithCfg(model_def, mc)
  if not train_def:
    train_def = ctc_training

  # Get recog ffnn LM
  search_config = {}
  if model_config and "recog_language_model" in model_config:
    recog_language_model = model_config["recog_language_model"].copy()
    cls_name = recog_language_model.pop("class")
    assert cls_name == "FeedForwardLm"
    lm_checkpoint = get_ffnn_lm(task.train_dataset.vocab, **recog_language_model)
    if cache_manager:
      lm_checkpoint_path = DelayedCodeWrapper("cf('{}')", lm_checkpoint.checkpoint)
    else:
      lm_checkpoint_path = lm_checkpoint.checkpoint

    search_config.update({
      "preload_from_files": {
        "recog_lm": {
          "prefix": "recog_language_model.",
          "filename": lm_checkpoint_path,
        },
      },
    })

  model_with_checkpoint = []
  model_with_checkpoint.append(train(
    prefix,
    task=task,
    config=config,
    post_config=dict_update_deep(post_config, post_config_updates),
    epilog=epilog,
    model_def=model_def,
    train_def=train_def,
    num_epochs=num_epochs,
    gpu_mem=gpu_mem,
    num_processes=num_processes,
    time_rqmt=time_rqmt if time_rqmt else (36 if init_small else 132),
  ))

  train_job = model_with_checkpoint[0].get_training_job()
  if env_updates:
    for k, v in env_updates.items():
      train_job.set_env(k, v)

  recog_post_proc_funcs = []
  if config.get("use_eos_postfix", False):
    recog_post_proc_funcs.append(_remove_eos_label_v2)
  recog_training_exp(
    prefix,
    task,
    model_with_checkpoint[0],
    recog_def=decoder_def,
    decoder_hyperparameters=decoder_hyperparameters,
    search_config=search_config,
    recog_post_proc_funcs=recog_post_proc_funcs,
    num_shards_recog=num_shards_recog_init,  # NOTE: breaks hash
    num_shards_pseudo=num_shards_pseudo,
    num_shards_prior=num_shards_prior_init,
    is_last=self_training_rounds == 0,
    prior_from_max=prior_from_max,
    empirical_prior=emp_prior if with_prior and empirical_prior else None,
    cache_manager=cache_manager,
  )

  return model_with_checkpoint[0]


def train_full_sum(
        name: str,
        config: Dict[str, Any],
        *,
        vocab: str = "bpe10k",
        train_language_model_opts: Dict[str, Any],
        train_vocab_opts: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        config_updates: Optional[Dict[str, Any]] = None,
        post_config_updates: Optional[Dict[str, Any]] = None,
        epilog: Sequence[serialization.SerializerObject] = (),
        num_epochs: int = 2000,
        gpu_mem: Optional[int] = 24,
        num_processes: Optional[int] = None,
        env_updates: Optional[Dict[str, str]] = None,
        init_checkpoint: Optional[PtCheckpoint] = None,
) -> Optional[ModelWithCheckpoints]:
  """
  Train experiment
  """
  from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.train import train
  from i6_experiments.users.schmitt.datasets.librispeech import get_librispeech_task_raw_v2, TrainDatasetSel

  if _sis_prefix is None:
    _sis_setup_global_prefix()
  prefix = _sis_prefix + "/" + name + "_full-sum-from-scratch"

  task, _, _ = get_librispeech_task_raw_v2(
    vocab=vocab,
    train_vocab_opts=train_vocab_opts,
    ds_sel=TrainDatasetSel.train_960h,
    init_small=False,
    with_prior=True,
    empirical_prior=True,
    keep_small_labels=False,
    train_subset=None,
    eval_subset=3000,
  )

  emp_prior = get_prior_from_unigram(task.prior_dataset.vocab, task.prior_dataset, vocab)

  config = copy.deepcopy(config)
  config = dict_update_deep(config, config_updates)
  # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
  if "__train_audio_preprocess" in config:
    task: Task = copy.copy(task)
    task.train_dataset = copy.copy(task.train_dataset)
    task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

  model_def = ctc_model_def
  if model_config:
    mc = model_config.copy()
    model_def = ModelDefWithCfg(model_def, mc)

  assert "order" in train_language_model_opts and train_language_model_opts["class"] == "ngram"
  train_lm = get_count_based_n_gram(task.train_dataset.vocab, train_language_model_opts["order"])

  train_def = ctc_sum_training

  assert "full_sum_opts" in config
  config["full_sum_opts"].update({
    "lm_path": train_lm,
    "empirical_prior": emp_prior,
  })

  config.pop("aux_loss_layers", None)

  # use single gpu
  del config["__num_processes"]
  del config["torch_distributed"]

  model_with_checkpoint = train(
    prefix,
    task=task,
    config=config,
    post_config=dict_update_deep(post_config, post_config_updates),
    epilog=epilog,
    model_def=model_def,
    train_def=train_def,
    init_params=init_checkpoint,
    reset_steps=False,
    num_epochs=num_epochs,
    gpu_mem=gpu_mem,
    num_processes=num_processes,
    time_rqmt=80,
  )
  train_job = model_with_checkpoint.get_training_job()
  if env_updates:
    for k, v in env_updates.items():
      train_job.set_env(k, v)


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
  if not prefix_name:
    from i6_experiments.users.schmitt.util.sis_setup import get_setup_prefix_for_module

    prefix_name = get_setup_prefix_for_module(__name__)
  global _sis_prefix
  _sis_prefix = prefix_name
