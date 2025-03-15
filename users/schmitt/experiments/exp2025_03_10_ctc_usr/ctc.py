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
from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.lm.n_gram import get_count_based_n_gram, get_prior_from_unigram
from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.lm.ffnn import FeedForwardLm, get_ffnn_lm
from i6_experiments.users.schmitt.nn.util import DelayedCodeWrapper
from i6_experiments.common.setups import serialization

from i6_core.util import uopen

from .configs import (
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
    config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    dict_update_deep,
    post_config,
    config_params_v1,
    config_params_v2,
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
num_shards_recog = 4 # None, 4, 16
num_shards_recog_init = 4
num_shards_pseudo = 64 # 32, 64
num_shards_prior = 64
num_shards_prior_init = None
calculate_pseudo_label_scores = True
calculate_pseudo_label_scores_init = True
cache_manager = True
exclude_epochs = True


@dataclass
class ConfigParams:
    """
    Config parameters for the experiments.

    Args:
        vocab: Vocabulary name, e.g. "bpe128, bpe10k", "char", "spm20k"
        decoding_imp: Decoding implementation, e.g. "flashlight", "albert-flashlight", "albert-lm", "albert-greedy", "marten-greedy"
        epochs: Number of training epochs
        self_training_rounds: Self-supevised training rounds. number of epochs per round is definded in epoch_dict below
        reset_steps: # Whether to reset steps after the first self-training round (for correct LR scheduling)
        init_small: whether to use 100h (True) or 960h (False) supervised initialization
        pseudo_label_small: 860h (True) vs 960h (False) pseudo-labels
        keep_small_labels: Keep true labels of 100h data during self-training (overwrites pseudo_label_small)
        pseudo_nbest: Number of top-k pseudo seqs to use for training (1=max-approx)
        with_prior: Use prior in decoding
        empirical_prior: Use empirical prior (True) vs model prior (False)
        prior_from_max: normalized over number of frames
            True: count for all labels, in how many frames they are the max label
            False: sum over model probs of each frame
        aux_loss: Use auxiliary loss
        alt_decoder: Use alternative decoder settings for self-training rounds (basically always the case)
        calc_last_pseudo_labels: Calculate pseudo-labels also after last self-training round (e.g. for eval)
        tune_hyperparameters:
            Tune LM and prior weights. currently using very basic tuning, want to switch to Albert's tuning
        from_scratch: both use pseudo labels of 100h supervised model but:
            False: init with weights from 100h supervised model
            True: random init
        decode_every_step:
            do decoding (for pseudo labels) in after every mini-batch. we can still use gradient accumulation
            to only update the model after a certain number of mini-batches.
        accum_grad_multiple_step: Accumulate gradients over multiple steps
        decoder_lm_config: Decoder LM config
        use_norm_st_loss:
            Use normalized self-training loss. for CTC loss, the loss is normalized over the number of labels.
            for full-sum loss, it is normalized over the number of frames.
        use_sum_criterion: Use full-sum criterion
        horizontal_prior: prior for repetitions
        blank_prior: Blank prior
        prior_gradient: Prior gradient
        empirical_prior_full_sum: Empirical prior full sum
        prior_from_max_full_sum: same as prior_from_max but for full sum criterion
        train_lm_config: Train LM config
        top_k:
        version:
        print_gradients:
        alignment_topk:
        blank_correction_version:
        correction_in_final_score:
        am_lm_prior: scales for AM, LM, prior
        use_sgd: SGD or Adam
        adamw_betas:
        self_train_subset: small train subset for testing
    """
    vocab: str
    decoding_imp: str
    epochs: int
    self_training_rounds: int
    reset_steps: bool
    init_small: bool
    pseudo_label_small: bool
    keep_small_labels: bool
    pseudo_nbest: int
    with_prior: bool
    empirical_prior: bool
    prior_from_max: bool
    aux_loss: bool
    alt_decoder: bool
    calc_last_pseudo_labels: bool
    tune_hyperparameters: bool
    from_scratch: bool
    decode_every_step: bool
    accum_grad_multiple_step: int
    decoder_lm_config: Optional[Dict[str, Any]]
    use_norm_st_loss: bool
    use_sum_criterion: bool
    horizontal_prior: bool
    blank_prior: bool
    prior_gradient: bool
    empirical_prior_full_sum: bool
    prior_from_max_full_sum: bool
    train_lm_config: Optional[Dict[str, Any]]
    top_k: int
    version: int
    print_gradients: bool
    alignment_topk: bool
    blank_correction_version: int
    correction_in_final_score: bool
    am_lm_prior: List[Tuple[float, float, float]]
    use_sgd: bool
    adamw_betas: Optional[Tuple[float, float]]
    self_train_subset: Optional[int]
    model_config: Optional[Dict[str, Any]]
    batch_size: int
    # the following params are set after __init__ by __post_init__
    self_epochs: Optional[int] = dataclasses.field(init=False)
    decoder_hyperparameters: Optional[Dict[str, Any]] = dataclasses.field(init=False)
    alt_decoder_hyperparameters: Optional[Dict[str, Any]] = dataclasses.field(init=False)
    decoding_str: Optional[str] = dataclasses.field(init=False)
    every_step_hyperparameters: Optional[Dict[str, Any]] = dataclasses.field(init=False)

    # after __init__, modify/set some more parameters
    def __post_init__(self):
        self.set_epochs()
        self.set_decoder_hyperparams()

    def set_epochs(self):
        if self.init_small:
            self.epochs = self.epochs // 10
        if self.self_training_rounds > 0:
            if self.pseudo_label_small:  # 860h
                epoch_dict = {
                    1: 450,
                    2: 225,
                    4: 113,  # 113 * 4 (gpus) / 18 (sub-epochs) = 25 epochs
                    6: 75,
                    8: 56,
                    10: 45
                }
            else:
                epoch_dict = {1: 500, 2: 250, 4: 125, 6: 83, 8: 63, 10: 50}
            self.self_epochs = epoch_dict[self.self_training_rounds]
            if self.self_train_subset:
                self.self_epochs = 56
        else:
            self.self_epochs = None

    def set_decoder_hyperparams(self):
        decoder_hyperparameters = None
        alt_decoder_hyperparameters = None
        every_step_hyperparameters = None

        if self.decoding_imp == "marten-greedy":
            decoder_hyperparameters = {
                "greedy": True
            }
            decoding_str = "-recog_greedy"
            if self.with_prior:
                decoder_hyperparameters["prior_weight"] = 0.2
                decoding_str += f"_p{str(decoder_hyperparameters['prior_weight']).replace('.', '')}" + ("-emp" if config.empirical_prior else "")
        elif self.decoding_imp == "albert-greedy":
            decoding_str = "-recog_albert"
        elif self.decoding_imp.endswith("flashlight") or self.decoding_imp == "albert-lm":
            decoder_hyperparameters = {
                "log_add": False,
                "nbest": 1,
                "beam_size": 10,
                "lm_weight": 0.8, # NOTE: weights are exponentials of the probs
                "use_logsoftmax": True,
                "use_lm": True,
                "use_lexicon": True,
            }
            if self.with_prior:
                decoder_hyperparameters["prior_weight"] = 0.3 # 0.2 if not using emprirical prior
            if self.decoder_lm_config:
                decoder_hyperparameters["lm_order"] = self.decoder_lm_config["order"] if self.decoder_lm_config["class"] == "ngram" else f"ffnn{self.decoder_lm_config['context_size']}"
                decoder_hyperparameters["use_lexicon"] = False
                if self.decoder_lm_config["class"] == "FeedForwardLm":
                    self.model_config["recog_language_model"] = self.decoder_lm_config
                    if self.decode_every_step:
                        self.model_config["train_language_model"] = self.decoder_lm_config

            p0 = f"_p{str(decoder_hyperparameters['prior_weight']).replace('.', '')}" + ("-emp" if self.empirical_prior else ("-from_max" if self.prior_from_max else "")) if self.with_prior else ""
            p1 = "sum" if decoder_hyperparameters['log_add'] else "max"
            p2 = f"n{self.pseudo_nbest}"
            p3 = f"b{decoder_hyperparameters['beam_size']}"
            p4 = f"w{str(decoder_hyperparameters['lm_weight']).replace('.', '')}" + ((f"o{self.decoder_lm_config['order']}" if self.decoder_lm_config["class"] == "ngram" else f"ffnn{self.decoder_lm_config['context_size']}") if self.decoder_lm_config else "")
            p5 = "_logsoftmax" if decoder_hyperparameters['use_logsoftmax'] else ""
            p6 = "_noLM" if not decoder_hyperparameters['use_lm'] else ""
            p7 = "_noLEX" if not decoder_hyperparameters['use_lexicon'] else ""
            decoding_str = f"{p0}_{p1}_{p2}_{p3}_{p4}{p5}{p6}{p7}"

            if self.decoding_imp == "albert-flashlight":
                decoding_str = "-recog_albert_lm" + decoding_str
            elif self.decoding_imp == "albert-lm":
                decoding_str = "-recog_alberts_own_lm" + decoding_str
            else:
                decoding_str = "-recog_lm" + decoding_str

            if self.alt_decoder:
                alt_decoder_hyperparameters = decoder_hyperparameters.copy()
                alt_decoder_hyperparameters["lm_weight"] = 0.4
                alt_decoder_hyperparameters["beam_size"] = 10
                if self.with_prior:
                    alt_decoder_hyperparameters["prior_weight"] = 0.3

                if self.decode_every_step:
                    every_step_hyperparameters = alt_decoder_hyperparameters.copy()
                    every_step_str = ""

                if self.use_sum_criterion or self.decode_every_step:
                    alt_decoder_hyperparameters["lm_weight"] = 0.0
                    alt_decoder_hyperparameters["prior_weight"] = 0.0
                    alt_decoder_hyperparameters["use_lm"] = False
                    alt_decoder_hyperparameters["use_lexicon"] = False
                    str_add = "_no-lexicon"
                    if self.decode_every_step:
                        a0 = f"p{str(every_step_hyperparameters['prior_weight']).replace('.', '')}" if self.with_prior else ""
                        a1 = f"b{every_step_hyperparameters['beam_size']}"
                        a2 = f"w{str(every_step_hyperparameters['lm_weight']).replace('.', '')}"
                        every_step_str = f"_{a0}_{a1}_{a2}"
                else:
                    str_add = ""

                a0 = f"_p{str(alt_decoder_hyperparameters['prior_weight']).replace('.', '')}" + ("-emp" if self.empirical_prior else ("-from_max" if self.prior_from_max else "")) if self.with_prior else ""
                a1 = f"b{alt_decoder_hyperparameters['beam_size']}"
                a2 = f"w{str(alt_decoder_hyperparameters['lm_weight']).replace('.', '')}"
                a3 = ("_every-step" + (f"-accum{self.accum_grad_multiple_step}" if self.accum_grad_multiple_step > 1 else "") + every_step_str if self.decode_every_step else "") + ("_tune" if self.tune_hyperparameters else "")
                decoding_str += f"_ALT{a3}{a0}_{a1}_{a2}{str_add}"
        else:
            raise ValueError(f"Unknown decoder selection: {self.decoding_imp}")

        self.decoder_hyperparameters = decoder_hyperparameters
        self.alt_decoder_hyperparameters = alt_decoder_hyperparameters
        self.every_step_hyperparameters = every_step_hyperparameters
        self.decoding_str = decoding_str


def py():
    """Sisyphus entry point"""

    # Config
    configs = [
      ConfigParams(**config_params) for config_params in [
            config_params_v1,
            # config_params_v2
        ]
    ]
    # configs.append(
    #     ConfigParams(
    #         **config_params_v1
    #     )
    # )

    for config in configs:
        assert not config.decode_every_step or (config.decode_every_step and config.decoder_lm_config["class"] == "FeedForwardLm" and config.empirical_prior)
        assert (config.empirical_prior_full_sum and config.empirical_prior) or not config.empirical_prior_full_sum

        config_updates = {
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(config.epochs),
            "batch_size": config.batch_size,
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        }
        config_updates_self_training = {
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(config.self_epochs),
            "batch_size": config.batch_size,
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        } if config.self_training_rounds > 0 else None

        if config_updates_self_training:
            if config.adamw_betas:
                config_updates_self_training["optimizer.betas"] = config.adamw_betas
            if not config.reset_steps:
                if config.pseudo_label_small:
                    config_updates_self_training["learning_rate_piecewise_steps"] = [253_000, 506_000, 562_000]
                else:
                    config_updates_self_training["learning_rate_piecewise_steps"] = [279_000, 558_000, 620_000]
            if config.decode_every_step:
                config_updates_self_training["decode_every_step"] = config.decode_every_step
                assert config.every_step_hyperparameters
                config_updates_self_training["hyperparameters_decoder"] = config.every_step_hyperparameters
                if config.accum_grad_multiple_step > 1:
                    config_updates_self_training["accum_grad_multiple_step"] = config.accum_grad_multiple_step
            if not config.use_norm_st_loss:
                config_updates_self_training["use_normalized_loss"] = config.use_norm_st_loss

        # Start Train Pipeline
        for am, lm, prior in config.am_lm_prior:
            if config.use_sum_criterion:
                if am != 1.0 or lm != 1.0 or prior != 1.0:
                    scales_not_std = True
                    config_full_sum = {
                        "am_scale": am,
                        "lm_scale": lm,
                        "prior_scale": prior
                    }
                else:
                    scales_not_std = False
                    config_full_sum = {}

                if not config.horizontal_prior:
                    config_full_sum["horizontal_prior"] = config.horizontal_prior
                if not config.blank_prior:
                    config_full_sum["blank_prior"] = config.blank_prior
                if not config.prior_gradient:
                    config_full_sum["prior_gradient"] = config.prior_gradient
                if config.top_k > 0:
                    config_full_sum["top_k"] = config.top_k
                if config.empirical_prior_full_sum:
                    config_full_sum["empirical_prior"] = True
                if config.prior_from_max_full_sum:
                    config_full_sum["max_prior"] = True
                if not config.alignment_topk:
                    config_full_sum["alignment_topk"] = False
                if config.blank_correction_version > 0:
                    config_full_sum["blank_correction_version"] = config.blank_correction_version
                if config.correction_in_final_score:
                    config_full_sum["correction_in_final_score"] = True
                if config.print_gradients:
                    config_full_sum["print_gradients"] = True

                # This is to change the hash when we made chnages in the loss function
                config_full_sum["version"] = config.version

                sum_str = f"-full_sum" + \
                    (f"_p{str(config_full_sum['prior_scale']).replace('.', '')}_l{str(config_full_sum['lm_scale']).replace('.', '')}_a{str(config_full_sum['am_scale']).replace('.', '')}" if scales_not_std else "") + \
                    (f"_LMorder{config.train_lm_config['order']}" if config.train_lm_config["class"] == "ngram" and config.train_lm_config["order"] > 2 else (f"_ffnn{config.train_lm_config['context_size']}" if config.train_lm_config["class"] == "FeedForwardLm" else "")) + \
                    (f"_topK{config.top_k}" + ("_align" if config.alignment_topk else "") + (f"_bc{config.blank_correction_version}" + ("sc" if config.correction_in_final_score else "") if config.blank_correction_version > 0 else "") if config.top_k > 0 else "") + \
                    ("_emp" if config.empirical_prior_full_sum else "") + \
                    ("_max_pr" if not config.empirical_prior_full_sum and config.prior_from_max_full_sum else "") + \
                    ("_wo_hor_pr" if not config.horizontal_prior else "") + \
                    ("_wo_blank_pr" if not config.blank_prior else "") + \
                    ("_wo_pr_grad" if not config.prior_gradient else "")

                if config.train_lm_config:
                    config.model_config["train_language_model"] = config.train_lm_config

            alias_name = f"altLRedge1e-4-ctc-baseline" + \
                (sum_str if config.use_sum_criterion else "") + \
                (f"-self_training_{config.self_training_rounds}" + ("_no_norm" if not config.use_norm_st_loss else "") + ("_keep_LR" if not config.reset_steps else "") + ("_SGD" if config.use_sgd else (f"_b1-{str(config.adamw_betas[0]).replace('.', '')}_b2-{str(config.adamw_betas[1]).replace('.', '')}" if config.adamw_betas else "")) + ("_from_scratch" if config.from_scratch else "") + (f"_s{config.self_train_subset}" if config.self_train_subset is not None else "") + (f"_e{config.self_epochs}" if config.self_epochs != 450 else "") if config.self_training_rounds > 0 else "") + \
                (f"-wo_aux_loss" if not config.aux_loss else "") + \
                (f"-ds100h" if config.init_small else "") + \
                (f"-pl960h" + ("_keep100h" if config.keep_small_labels else "") if not config.pseudo_label_small else "") + \
                f"-{config.vocab}" + \
                f"{config.decoding_str}"

            if config.decoding_imp in ["flashlight", "marten-greedy"]:
                decoder_def = model_recog_lm
            elif config.decoding_imp == "albert-greedy":
                decoder_def = model_recog
            elif config.decoding_imp == "albert-flashlight":
                decoder_def = model_recog_flashlight
            elif config.decoding_imp == "albert-lm":
                decoder_def = model_recog_lm_albert
            else:
                raise ValueError(f"Unknown decoder selection: {config.decoding_imp}")

            epilog = [
                # resampy is needed for speed perturbation
                # sys.path.insert(...)
                serialization.ExternalImport(tk.Path("/work/asr3/zeyer/schmitt/venvs/resampy_package"))
            ]

            train_exp(
                name=alias_name,
                config=config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
                decoder_def=decoder_def,
                decoder_hyperparameters=config.decoder_hyperparameters,
                hyperparamters_self_training=config.alt_decoder_hyperparameters if config.alt_decoder else None,
                pseudo_nbest=config.pseudo_nbest,
                model_config=config.model_config,
                config_updates=config_updates,
                config_updates_self_training=config_updates_self_training,
                config_full_sum=config_full_sum if config.use_sum_criterion else None,
                vocab=config.vocab,
                self_training_rounds=config.self_training_rounds,
                init_small=config.init_small,
                pseudo_label_small=config.pseudo_label_small,
                keep_small_labels=config.keep_small_labels,
                with_prior=config.with_prior,
                empirical_prior=config.empirical_prior,
                prior_from_max=config.prior_from_max,
                use_sum_criterion=config.use_sum_criterion,
                aux_loss=config.aux_loss,
                self_train_subset=config.self_train_subset,
                calc_last_pseudo_labels=config.calc_last_pseudo_labels,
                tune_hyperparameters=config.tune_hyperparameters,
                from_scratch=config.from_scratch,
                use_sgd=config.use_sgd,
                reset_steps=config.reset_steps,
                epilog=epilog,
            )
    

_train_experiments: Dict[str, ModelWithCheckpoints] = {}


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    decoder_def: Callable,
    *,
    decoder_hyperparameters: dict = None,
    hyperparamters_self_training: dict = None,
    pseudo_nbest: int = 1,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    vocab: str = "bpe10k",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    config_updates_self_training: Optional[Dict[str, Any]] = None,
    config_full_sum: Optional[Dict[str, Any]] = None,
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
    pseudo_label_small: bool = True,
    keep_small_labels: bool = False,
    with_prior: bool = False,
    empirical_prior: bool = False,
    prior_from_max: bool = False,
    use_sum_criterion: bool = False,
    aux_loss: bool = False,
    self_train_subset: Optional[int] = None,
    calc_last_pseudo_labels: bool = False,
    tune_hyperparameters: bool = False,
    from_scratch: bool = False,
    use_sgd: bool = False,
    reset_steps: bool = True,
) -> Optional[ModelWithCheckpoints]:
    """
    Train experiment
    """
    from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.train import train
    from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.recog import recog_training_exp, GetBestTuneValue
    from i6_experiments.users.schmitt.datasets.librispeech import get_librispeech_task_raw_v2, TrainDatasetSel

    print("Job Name:", name)
    if not enabled:
        return None

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name

    # get standard LS task, pseudo label dataset (e.g. 860h), and train_100h dataset (for supervised init)
    task, pseudo_labels_ds, train_100_ds = get_librispeech_task_raw_v2(
        vocab=vocab,
        train_vocab_opts=train_vocab_opts,
        save_pseudo_labels = (TrainDatasetSel.train_860h if pseudo_label_small else TrainDatasetSel.train_960h) if self_training_rounds > 0 or calc_last_pseudo_labels else None,
        ds_sel = TrainDatasetSel.train_100h if init_small else TrainDatasetSel.train_960h,
        init_small=init_small,
        with_prior=with_prior,
        empirical_prior=empirical_prior,
    )
    
    if with_prior and empirical_prior:
        emp_prior = get_prior_from_unigram(task.prior_dataset.vocab, task.prior_dataset, vocab)
    
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
    if "__train_audio_preprocess" in config:
        task: Task = copy.copy(task)
        task.train_dataset = copy.copy(task.train_dataset)
        task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

    if not model_def:
        model_def = ctc_model_def
        model_def_self = ctc_model_def
    if model_config:
        mc = model_config.copy()
        if "train_language_model" in mc:
            mc.pop("train_language_model", None)
        mc_self = model_config.copy()
        if "train_language_model" in mc and mc["train_language_model"]["class"] == "ngram":
            mc_self.pop("train_language_model", None)
        model_def = ModelDefWithCfg(model_def, mc)
        model_def_self = ModelDefWithCfg(model_def_self, mc_self)
    if not train_def:
        train_def = ctc_training
        
    # Create LM for full-sum criterion
    if use_sum_criterion:
        if model_config and "train_language_model" in model_config:
            train_language_model = model_config["train_language_model"].copy()
            cls_name = train_language_model.pop("class")
            if cls_name == "FeedForwardLm":
                lm_checkpoint = get_ffnn_lm(task.train_dataset.vocab, **train_language_model)
                if cache_manager:
                    lm_checkpoint_path = DelayedCodeWrapper("cf('{}')", lm_checkpoint.checkpoint)
                else:
                    lm_checkpoint_path = lm_checkpoint.checkpoint
                config_updates_self_training.update({
                    "preload_from_files": {
                        "train_lm": {
                            "init_for_train": True,
                            "prefix": "train_language_model.",
                            "filename": lm_checkpoint_path,
                        },
                    },
                })
                train_lm = None
            elif cls_name == "ngram":
                train_lm = get_count_based_n_gram(task.train_dataset.vocab, train_language_model["order"])
            else:
                raise NotImplementedError("This LM does not exist")
        else:
            raise NotImplementedError("No LM for full-sum criterion selected")
        
    # Get recog ffnn LM
    search_config = None
    if model_config and "recog_language_model" in model_config:
        recog_language_model = model_config["recog_language_model"].copy()
        cls_name = recog_language_model.pop("class")
        assert cls_name == "FeedForwardLm"
        lm_checkpoint = get_ffnn_lm(task.train_dataset.vocab, **recog_language_model)
        if cache_manager:
            lm_checkpoint_path = DelayedCodeWrapper("cf('{}')", lm_checkpoint.checkpoint)
        else:
            lm_checkpoint_path = lm_checkpoint.checkpoint
            
        search_config = {
            "preload_from_files": {
                "recog_lm": {
                    "prefix": "recog_language_model.",
                    "filename": lm_checkpoint_path,
                },
            },
        }
        if config_updates_self_training and config_updates_self_training.get("decode_every_step", False):
            config_updates_self_training.update({
                "preload_from_files": {
                    "train_lm": {
                        "init_for_train": True,
                        "prefix": "train_language_model.",
                        "filename": lm_checkpoint.checkpoint,
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
    return
    train_job = model_with_checkpoint[0].get_training_job()
    if env_updates:
        for k, v in env_updates.items():
            train_job.set_env(k, v)

    recog_post_proc_funcs = []
    if config.get("use_eos_postfix", False):
        recog_post_proc_funcs.append(_remove_eos_label_v2)
    pseudo_label_path_dict = recog_training_exp(
        prefix,
        task,
        model_with_checkpoint[0],
        recog_def=decoder_def,
        decoder_hyperparameters=decoder_hyperparameters,
        save_pseudo_labels=(pseudo_labels_ds, train_100_ds) if calc_last_pseudo_labels or self_training_rounds > 0 else None,
        pseudo_nbest=pseudo_nbest,
        calculate_pseudo_label_scores=calculate_pseudo_label_scores_init, # NOTE: breaks hash
        search_config=search_config,
        recog_post_proc_funcs=recog_post_proc_funcs,
        num_shards_recog=num_shards_recog_init, # NOTE: breaks hash
        num_shards_pseudo=num_shards_pseudo,
        num_shards_prior=num_shards_prior_init,
        is_last=self_training_rounds == 0,
        prior_from_max=prior_from_max,
        empirical_prior=emp_prior if with_prior and empirical_prior else None,
        cache_manager=cache_manager,
        # check_train_scores_nbest=0,
        # exclude_epochs=sorted(list(model_with_checkpoint[0].fixed_epochs))[:-1]
    )
    
    # Do self training on pseudo labels
    for i in range(self_training_rounds):
        assert pseudo_label_path_dict is not None, "Pseudo label path is not set"
        prefix_self_training = prefix + f"/self-training-{i+1}"
        task, _, _ = get_librispeech_task_raw_v2(
            vocab=vocab,
            train_vocab_opts=train_vocab_opts,
            ds_sel=TrainDatasetSel.train_860h if pseudo_label_small else TrainDatasetSel.train_960h,
            init_small=init_small,
            with_prior=with_prior,
            empirical_prior=empirical_prior,
            pseudo_label_path=pseudo_label_path_dict,
            keep_small_labels=keep_small_labels,
            train_subset=self_train_subset,
            eval_subset=300 if self_train_subset else 3000,
            pseudo_nbest=pseudo_nbest,
        )
        
        config_self = config.copy()
        config_self = dict_update_deep(config_self, config_updates_self_training)
        # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
        if "__train_audio_preprocess" in config_self:
            task: Task = copy.copy(task)
            task.train_dataset = copy.copy(task.train_dataset)
            task.train_dataset.train_audio_preprocess = config_self.pop("__train_audio_preprocess")
        
        if use_sum_criterion:
            train_def = ctc_sum_training
            config_self = dict_update_deep(config_self, config_full_sum)
            if train_lm:
                config_self["lm_path"] = train_lm
            else:
                config_self["lm_path"] = "ffnn" + str(model_config["train_language_model"]["context_size"])
        elif pseudo_nbest > 1:
            config_self["ps_nbest"] = pseudo_nbest
            
        if config_self.get("empirical_prior", False) or config_self.get("decode_every_step", False):
            config_self["empirical_prior"] = emp_prior
                
        if use_sgd:
            config_self["optimizer"] = {
                "class": "sgd"
            }
            
        # When testing on a smaller subset we only want one gpu
        if self_train_subset is not None:
            config_self["__num_processes"] = 1
            # config_self["learning_rate_piecewise_steps"] = [4_500, 9_000, 10_000]
            config_self["learning_rate_piecewise_steps"] = [2_250, 4_500, 5_000]
            if not use_sgd:
                # peak_lr = 1e-4
                # config_self["learning_rate_piecewise_values"] = [peak_lr * 1.001e-1, peak_lr, peak_lr * 3e-2, peak_lr * 3e-3]
                # peak_lr = 3e-5
                # config_self["learning_rate_piecewise_values"] = [peak_lr * 1e-1, peak_lr, peak_lr * 1e-1, peak_lr * 1e-2]
                peak_lr = 1e-4
                # config_self["learning_rate_piecewise_values"] = [peak_lr, peak_lr, peak_lr, peak_lr]
                config_self["learning_rate_piecewise_values"] = [peak_lr, peak_lr, peak_lr * 0.27, peak_lr * 0.1]
            else:
                peak_lr = 1e-2
                config_self["learning_rate_piecewise_values"] = [peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3]
        if not aux_loss:
            config_self.pop("aux_loss_layers")

        # Use different LR if second iteration, NOTE: this is very specific to 860h training
        if i > 0 and reset_steps:
            # if i > 2:
            #     peak_lr = 4e-4
            #     config_self["learning_rate_piecewise_values"] = [peak_lr, peak_lr, peak_lr * 3e-2, peak_lr * 3e-3]
            #     config_self["learning_rate_piecewise_steps"] = [20_000] + config_self["learning_rate_piecewise_steps"][1:]
            # else:
            peak_lr = 4e-4
            config_self["learning_rate_piecewise_values"] = [peak_lr * 1e-1, peak_lr, peak_lr * 3e-2, peak_lr * 3e-3]
            config_self["learning_rate_piecewise_steps"] = [20_000] + config_self["learning_rate_piecewise_steps"][1:]
        
        if i == 0 and from_scratch:
            init_checkpoint = None
        else:
            init_checkpoint = model_with_checkpoint[i].get_last_fixed_epoch().checkpoint

        model_with_checkpoint.append(train(
            prefix_self_training,
            task=task,
            config=config_self,
            post_config=dict_update_deep(post_config, post_config_updates),
            epilog=epilog,
            model_def=model_def_self,
            train_def=train_def,
            init_params=init_checkpoint,
            reset_steps=True if reset_steps or i == 0 else False,
            num_epochs=num_epochs,
            gpu_mem=gpu_mem,
            num_processes=num_processes,
            time_rqmt=time_rqmt if time_rqmt else ((8 if self_train_subset else 156) if use_sum_criterion else 156),
        ))
        train_job = model_with_checkpoint[i + 1].get_training_job()
        if env_updates:
            for k, v in env_updates.items():
                train_job.set_env(k, v)
        
        if tune_hyperparameters:
            original_params = hyperparamters_self_training if hyperparamters_self_training else decoder_hyperparameters
            params = copy.copy(original_params)
            params.pop("lm_weight_tune", None)
            params.pop("prior_weight_tune", None)
            default_lm = original_params.get("lm_weight")
            default_prior = original_params.get("prior_weight")
            lm_scores = []
            prior_scores = []
            lm_tune_ls = [0.0, 0.05, 0.1, -0.05, -0.1]
            prior_tune_ls = [0.0, 0.05, 0.1, -0.05, -0.1]
            tune_exclude_epochs = []
            if exclude_epochs:
                tune_exclude_epochs = sorted(list(model_with_checkpoint[i + 1].fixed_epochs))[:-1]
            for dc_lm in lm_tune_ls:
                params["lm_weight"] = default_lm + dc_lm
                score = recog_training_exp(
                    prefix_self_training + f"/tune/lm/{str(dc_lm).replace('.', '').replace('-', 'm')}",
                    task,
                    model_with_checkpoint[i + 1],
                    recog_def=decoder_def,
                    decoder_hyperparameters=params,
                    search_config=search_config,
                    recog_post_proc_funcs=recog_post_proc_funcs,
                    exclude_epochs=tune_exclude_epochs,
                    num_shards_recog=num_shards_recog, # NOTE: breaks hash
                    num_shards_prior=num_shards_prior,
                    prior_from_max=prior_from_max,
                    empirical_prior=emp_prior if with_prior and empirical_prior else None,
                    return_summary = True,
                    cache_manager=cache_manager,
                    check_train_scores_nbest=0 if exclude_epochs else 2,
                )
                lm_scores.append(score)
            best_lm_tune = GetBestTuneValue(lm_scores, lm_tune_ls).out_best_tune
            tk.register_output(prefix_self_training + "/tune/lm_best", best_lm_tune)
            params["lm_weight"] = default_lm
            params["lm_weight_tune"] = best_lm_tune
            for dc_prior in prior_tune_ls:
                params["prior_weight"] = default_prior + dc_prior
                score = recog_training_exp(
                    prefix_self_training + f"/tune/prior/{str(dc_prior).replace('.', '').replace('-', 'm')}",
                    task,
                    model_with_checkpoint[i + 1],
                    recog_def=decoder_def,
                    decoder_hyperparameters=params,
                    search_config=search_config,
                    recog_post_proc_funcs=recog_post_proc_funcs,
                    exclude_epochs=tune_exclude_epochs,
                    num_shards_recog=num_shards_recog, # NOTE: breaks hash
                    num_shards_prior=num_shards_prior,
                    prior_from_max=prior_from_max,
                    empirical_prior=emp_prior if with_prior and empirical_prior else None,
                    return_summary = True,
                    cache_manager=cache_manager,
                    check_train_scores_nbest=0 if exclude_epochs else 2,
                )
                prior_scores.append(score)
            best_prior_tune = GetBestTuneValue(prior_scores, prior_tune_ls).out_best_tune
            tk.register_output(prefix_self_training + "/tune/prior_best", best_prior_tune)
            
            original_params["lm_weight_tune"] = best_lm_tune
            original_params["prior_weight_tune"] = best_prior_tune
        
        pseudo_label_path_dict = recog_training_exp(
            prefix_self_training,
            task,
            model_with_checkpoint[i + 1],
            recog_def=decoder_def,
            decoder_hyperparameters=hyperparamters_self_training if hyperparamters_self_training else decoder_hyperparameters,
            save_pseudo_labels=None if not calc_last_pseudo_labels and i+1 == self_training_rounds else (pseudo_labels_ds, train_100_ds),
            pseudo_nbest=pseudo_nbest,
            calculate_pseudo_label_scores=calculate_pseudo_label_scores,
            search_config=search_config,
            recog_post_proc_funcs=recog_post_proc_funcs,
            num_shards_recog=num_shards_recog, # NOTE: breaks hash
            num_shards_pseudo=num_shards_pseudo,
            num_shards_prior=num_shards_prior,
            is_last=i+1 == self_training_rounds,
            prior_from_max=prior_from_max,
            empirical_prior=emp_prior if with_prior and empirical_prior else None,
            cache_manager=cache_manager,
        )

    _train_experiments[name] = model_with_checkpoint[-1]
    return model_with_checkpoint[-1]


def _remove_eos_label_v2(res: RecogOutput) -> RecogOutput:
    from i6_experiments.users.schmitt.datasets.score_results import RecogOutput
    from i6_core.returnn.search import SearchRemoveLabelJob

    return RecogOutput(SearchRemoveLabelJob(res.output, remove_label="</s>", output_gzip=True).out_search_results)


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.schmitt.util.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


#---------------------------------------------------------------------------------------------------------------------------------------
# MODEL DEFINITION

# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    num_enc_layers = config.int("num_enc_layers", 12)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)

    conv_norm = config.typed_value("conv_norm", None)
    enc_conformer_layer = config.typed_value("enc_conformer_layer", None)
    if enc_conformer_layer:
        assert not conv_norm, "set only enc_conformer_layer or conv_norm, not both"
        assert isinstance(enc_conformer_layer, dict) and "class" in enc_conformer_layer
    else:
        enc_conformer_layer = rf.build_dict(
            rf.encoder.conformer.ConformerEncoderLayer,
            conv_norm=conv_norm or {"class": "rf.BatchNorm", "use_mask": True},
            self_att=rf.build_dict(
                rf.RelPosSelfAttention,
                # Shawn et al 2018 style, old RETURNN way.
                with_bias=False,
                with_linear_pos=False,
                with_pos_bias=False,
                learnable_pos_emb=True,
                separate_pos_emb_per_head=False,
            ),
            ff_activation=rf.build_dict(rf.relu_square),
            num_heads=8,
        )
    enc_other_opts = config.typed_value("enc_other_opts", None)
    
    train_language_model = config.typed_value("train_language_model", None)
    train_lm = None
    if train_language_model:
        assert isinstance(train_language_model, dict)
        train_language_model = train_language_model.copy()
        cls_name = train_language_model.pop("class")
        assert cls_name == "FeedForwardLm"
        train_lm = FeedForwardLm(vocab_dim=target_dim, **train_language_model)
    recog_language_model = config.typed_value("recog_language_model", None)
    recog_lm = None
    if recog_language_model:
        assert isinstance(recog_language_model, dict)
        recog_language_model = recog_language_model.copy()
        cls_name = recog_language_model.pop("class")
        assert cls_name == "FeedForwardLm"
        recog_lm = FeedForwardLm(vocab_dim=target_dim, **recog_language_model)

    return Model(
        in_dim,
        num_enc_layers=num_enc_layers,
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_conformer_layer=enc_conformer_layer,
        enc_other_opts=enc_other_opts,
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_aux_logits=enc_aux_logits or (),
        train_language_model=train_lm,
        recog_language_model=recog_lm,
    )


ctc_model_def: ModelDef[Model]
ctc_model_def.behavior_version = 21
ctc_model_def.backend = "torch"
ctc_model_def.batch_size_factor = _batch_size_factor


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_conformer_layer: Optional[Dict[str, Any]] = None,
        enc_other_opts: Optional[Dict[str, Any]] = None,
        train_language_model: Optional[FeedForwardLm] = None,
        recog_language_model: Optional[FeedForwardLm] = None
    ):
        super(Model, self).__init__()

        import numpy
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        enc_layer_drop = config.float("enc_layer_drop", 0.0)
        if enc_layer_drop:
            enc_sequential = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)
        else:
            enc_sequential = rf.Sequential

        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer=enc_conformer_layer,
            num_layers=num_enc_layers,
            sequential=enc_sequential,
            **(enc_other_opts or {}),
        )

        # Experiments without final layer norm. (We might clean this up when this is not successful.)
        # Just patch the encoder here.
        enc_conformer_final_layer_norm = config.typed_value("enc_conformer_final_layer_norm", None)
        if enc_conformer_final_layer_norm is None:
            pass
        elif enc_conformer_final_layer_norm == "last":  # only in the last, i.e. remove everywhere else
            for layer in self.encoder.layers[:-1]:
                layer: ConformerEncoderLayer
                layer.final_layer_norm = rf.identity
        else:
            raise ValueError(f"invalid enc_conformer_final_layer_norm {enc_conformer_final_layer_norm!r}")

        disable_encoder_self_attention = config.typed_value("disable_encoder_self_attention", None)
        if disable_encoder_self_attention is not None:
            # Disable self-attention in encoder.
            from i6_experiments.users.schmitt.nn_rf.disable_self_att import apply_disable_self_attention_

            apply_disable_self_attention_(self.encoder, disable_encoder_self_attention)

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if not wb_target_dim:
            wb_target_dim = target_dim + 1
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))
        self.enc_logits = rf.Linear(self.encoder.out_dim, wb_target_dim)
        self.wb_target_dim = wb_target_dim
        self.out_blank_separated = config.bool("out_blank_separated", False)

        if target_dim.vocab and not wb_target_dim.vocab:

            # Just assumption for code now, might extend this later.
            assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
            vocab_labels = list(target_dim.vocab.labels) + [OUT_BLANK_LABEL]
            wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                vocab_labels, user_defined_symbols={OUT_BLANK_LABEL: blank_idx}
            )

        ctc_label_smoothing = config.float("ctc_label_smoothing", 0.0)
        ctc_label_smoothing_exclude_blank = config.bool("ctc_label_smoothing_exclude_blank", self.out_blank_separated)
        self.ctc_label_smoothing_exclude_blank = ctc_label_smoothing_exclude_blank
        if not self.out_blank_separated:
            self.ctc_label_smoothing_opts = {
                "smoothing": ctc_label_smoothing,
                "axis": self.wb_target_dim,
                "exclude_labels": [self.blank_idx] if ctc_label_smoothing_exclude_blank else None,
            }
        else:  # separate blank
            self.ctc_label_smoothing_opts = {
                "smoothing": ctc_label_smoothing,
                "axis": self.target_dim if ctc_label_smoothing_exclude_blank else self.wb_target_dim,
            }
        self.log_prob_normed_grad_opts = config.typed_value("log_prob_normed_grad", None)
        self.log_prob_normed_grad_exclude_blank = config.bool(
            "log_prob_normed_grad_exclude_blank", self.out_blank_separated
        )

        self.feature_batch_norm = None
        if config.bool("feature_batch_norm", False):
            self.feature_batch_norm = rf.BatchNorm(self.in_dim, affine=False, use_mask=True)
        self.feature_norm = config.bool("feature_norm", False)
        self.feature_stats = None
        feature_stats = config.typed_value("feature_stats")
        if feature_stats:
            assert isinstance(feature_stats, dict)
            self.feature_stats = rf.ParameterList(
                {
                    k: rf.Parameter(
                        rf.convert_to_tensor(numpy.loadtxt(v), dims=[self.in_dim], dtype=rf.get_default_float_dtype()),
                        auxiliary=True,
                    )
                    for k, v in feature_stats.items()
                }
            )

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.schmitt.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

        self.decoder = None
        aux_attention_decoder = config.typed_value("aux_attention_decoder", None)
        if aux_attention_decoder:
            assert isinstance(aux_attention_decoder, dict)
            aux_attention_decoder = aux_attention_decoder.copy()
            aux_attention_decoder.setdefault("class", "returnn.frontend.decoder.transformer.TransformerDecoder")
            if isinstance(aux_attention_decoder.get("model_dim", None), int):
                aux_attention_decoder["model_dim"] = Dim(aux_attention_decoder["model_dim"], name="dec_model")
            self.decoder = rf.build_from_dict(aux_attention_decoder, encoder_dim=enc_model_dim, vocab_dim=target_dim)

        vn = config.typed_value("variational_noise", None)
        if vn:
            # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
            # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
            blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
            blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
            for mod in self.modules():
                if isinstance(mod, blacklist):
                    continue
                for param_name, param in mod.named_parameters(recurse=False):
                    if param_name.endswith("bias"):  # no bias
                        continue
                    if param.auxiliary:
                        continue
                    rf.weight_noise(mod, param_name, std=vn)

        weight_dropout = config.typed_value("weight_dropout", None)
        if weight_dropout:
            # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
            # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
            blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
            blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
            for mod in self.modules():
                if isinstance(mod, blacklist):
                    continue
                for param_name, param in mod.named_parameters(recurse=False):
                    if param_name.endswith("bias"):  # no bias
                        continue
                    if param.auxiliary:
                        continue
                    rf.weight_dropout(mod, param_name, drop_prob=weight_dropout)
        
        self.train_language_model = train_language_model
        self.recog_language_model = recog_language_model

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Dim]:
        """encode, get logits"""
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
        )
        if self.feature_batch_norm:
            source = self.feature_batch_norm(source)
        if self.feature_norm:
            source = rf.normalize(source, axis=in_spatial_dim)
        if self.feature_stats:
            source = (source - self.feature_stats.mean) / self.feature_stats.std_dev
        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)
        # SpecAugment
        source = rf.audio.specaugment(
            source,
            spatial_dim=in_spatial_dim,
            feature_dim=self.in_dim,
            **self._specaugment_opts,
        )
        # Encoder including convolutional frontend
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        logits = self.enc_logits(enc)
        return logits, enc, enc_spatial_dim

    def log_probs_wb_from_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: incl blank
        :return: log probs with blank from logits (wb_target_dim)
            If out_blank_separated, we use a separate sigmoid for the blank.
        """
        if not self.out_blank_separated:  # standard case, joint distrib incl blank
            log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)
        else:  # separate blank
            assert self.blank_idx == self.target_dim.dimension  # not implemented otherwise
            dummy_blank_feat_dim = Dim(1, name="blank_feat")
            logits_wo_blank, logits_blank = rf.split(
                logits, axis=self.wb_target_dim, out_dims=[self.target_dim, dummy_blank_feat_dim]
            )
            log_probs_wo_blank = rf.log_softmax(logits_wo_blank, axis=self.target_dim)
            log_probs_wo_blank = self._maybe_apply_on_log_probs(log_probs_wo_blank)
            log_probs_blank = rf.log_sigmoid(logits_blank)
            log_probs_emit = rf.squeeze(rf.log_sigmoid(-logits_blank), axis=dummy_blank_feat_dim)
            log_probs, _ = rf.concat(
                (log_probs_wo_blank + log_probs_emit, self.target_dim),
                (log_probs_blank, dummy_blank_feat_dim),
                out_dim=self.wb_target_dim,
            )
            log_probs.feature_dim = self.wb_target_dim
        log_probs = self._maybe_apply_on_log_probs(log_probs)
        return log_probs

    def _maybe_apply_on_log_probs(self, log_probs: Tensor) -> Tensor:
        """
        :param log_probs: either with blank or without blank
        :return: log probs, maybe some smoothing applied (all on gradients so far, not on log probs itself)
        """
        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim

        log_probs = self._maybe_apply_log_probs_normed_grad(log_probs)

        if self.ctc_label_smoothing_exclude_blank:
            if self.out_blank_separated:
                if log_probs.feature_dim == self.target_dim:
                    log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)
            else:
                assert log_probs.feature_dim == self.wb_target_dim
                assert self.ctc_label_smoothing_opts["exclude_labels"] == [self.blank_idx]
                log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)
        else:
            if log_probs.feature_dim == self.wb_target_dim:
                log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)

        return log_probs

    def _maybe_apply_log_probs_normed_grad(self, log_probs: Tensor) -> Tensor:
        if not self.log_prob_normed_grad_opts:
            return log_probs

        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim
        if self.log_prob_normed_grad_exclude_blank:
            assert self.out_blank_separated
            if log_probs.feature_dim == self.wb_target_dim:
                return log_probs
        else:  # not excluded blank
            if log_probs.feature_dim == self.target_dim:
                return log_probs

        from alignments.util import normed_gradient, NormedGradientFuncInvPrior

        opts: Dict[str, Any] = self.log_prob_normed_grad_opts.copy()
        func_opts = opts.pop("func")
        assert isinstance(func_opts, dict)
        func_opts = func_opts.copy()
        assert func_opts.get("class", "inv_prior") == "inv_prior"  # only case for now
        func_opts.pop("class", None)
        func = NormedGradientFuncInvPrior(**func_opts)

        assert log_probs.batch_dim_axis is not None and log_probs.feature_dim_axis is not None
        log_probs_ = log_probs.copy_template()
        log_probs_.raw_tensor = normed_gradient(
            log_probs.raw_tensor,
            batch_axis=log_probs.batch_dim_axis,
            feat_axis=log_probs.feature_dim_axis,
            **opts,
            func=func,
        )
        return log_probs_
    
    
#---------------------------------------------------------------------------------------------------------------------------------------
# TRAINING DEFINITION

def is_separator(tensor: torch.Tensor, vocab: Vocabulary, nbest: int) -> list[list, list]:
    with torch.no_grad():
        batch_size = tensor.size(0)
        start_sep = vocab.label_to_id("Z@@")
        end_sep = vocab.label_to_id("Z")
        idxs = torch.where(tensor == start_sep)
        n = len(idxs[1])
        m = tensor.size(1)
        final_idxs = [[], []]
        idxs_cnt = dict.fromkeys(list(range(batch_size)), 0)
        i = 0
        for b in range(batch_size):
            for _ in range(nbest - 1):
                found_all = 0
                first_idx = None
                while found_all == 0:
                    for j in range(4):
                        if i >= n or idxs[0][i].item() != b:
                            idxs_cnt[b] += 1
                            final_idxs[0].append(torch.tensor(b, device=tensor.device))
                            final_idxs[1].append(torch.tensor(-1, device=tensor.device))
                            found_all = 2
                            break
                        else:
                            if j > 0 and idxs[1][i - 1] + 1 != idxs[1][i]:
                                break
                            elif j == 3:
                                found_all = 1
                                break
                            elif j == 0:
                                first_idx = i
                            i += 1
                    if found_all == 1:
                        if tensor[idxs[0][i], idxs[1][i] + 1] == end_sep:
                            idxs_cnt[b] += 1
                            final_idxs[0].append(idxs[0][first_idx])
                            final_idxs[1].append(idxs[1][first_idx])
                            i += 1
                        else:
                            found_all = 0
                            i = first_idx + 1
        for b in range(batch_size):
            assert idxs_cnt[b] == nbest - 1, f"Batch {b} has {idxs_cnt[b]} separators, should have {nbest - 1}"
        return final_idxs
    
def split_on_sep(tensor: torch.Tensor, sizes: torch.Tensor, vocab_dim: Dim, nbest: int, weights: torch.Tensor) -> tuple[list[rf.Tensor], list[Dim]]:
    idxs = is_separator(tensor, vocab_dim.vocab, nbest)
    batch_size = tensor.size(0)
    assert len(idxs[0]) == batch_size * (nbest - 1), f"Not enough separators found: {len(idxs[0])}, should be {batch_size * (nbest - 1)}"
    ret = []
    new_sizes = []
    old_lengths = [-5] * batch_size
    combined_size = torch.ones(batch_size, dtype=torch.int32, device=tensor.device)
    for n in range(nbest):
        if n < nbest - 1:
            lengths = [(idxs[1][i] if idxs[1][i].item() != -1 else sizes[int((i - n) / (nbest - 1))]) for i in range(len(idxs[0])) if (i - n) % (nbest - 1) == 0]
        else:
            lengths = [sizes[i] for i in range(batch_size)]
        assert len(lengths) == batch_size, f"Lengths: {len(lengths)}, should be {batch_size}"
        new_list = []
        for i in range(batch_size):
            if lengths[i].item() == -1:
                new_list.append(torch.tensor([], dtype=tensor.dtype, device=tensor.device))
            else:
                t_slice = tensor[i, (old_lengths[i] + 5):lengths[i]]
                new_list.append(t_slice)
        new_s = [l.size(0) for l in new_list]
        max_length = max(new_s)
        new_list = [torch.cat([t_slice, torch.tensor([0] * (max_length - t_slice.size(0)), device=tensor.device)]) for t_slice in new_list]
        new_tensor = torch.stack(new_list, dim=0)
        new_tensor = new_tensor.to(torch.int32)
        new_s = torch.tensor(new_s, dtype=torch.int32, device=tensor.device)
        combined_size += ((new_s * weights[:, n]).to(torch.int32))
        new_s = rf.convert_to_tensor(new_s, dims=(batch_dim,))
        new_s = Dim(new_s, name="out_spatial", dyn_size_ext=new_s)
        new_tensor = rf.convert_to_tensor(new_tensor, dims=(batch_dim, new_s), sparse_dim=vocab_dim)
        ret.append(new_tensor)
        new_sizes.append(new_s)
        old_lengths = lengths
    combined_size = rf.convert_to_tensor(combined_size, dims=(batch_dim,))
    combined_size = Dim(combined_size, name="out_spatial_comb", dyn_size_ext=combined_size)
    return ret, new_sizes, combined_size

def ctc_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim, weights: rf.Tensor = None):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    nbest = config.int("ps_nbest", 1)
    decode_every_step = config.bool("decode_every_step", False)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    log_probs = model.log_probs_wb_from_logits(logits)
    
    if decode_every_step:
        def _output_hyps(hyp: list) -> list:
            prev = None
            ls = []
            for h in hyp:
                if h != prev:
                    ls.append(h)
                    prev = h
            ls = [h for h in ls if h != model.blank_idx]
            return ls
        
        if nbest > 1:
            raise NotImplementedError("nbest > 1 with decode_every_step not implemented")
        hyperparameters = config.typed_value("hyperparameters_decoder").copy()
        # hyperparameters["beam_size"] = 1
        prior_file = config.typed_value("empirical_prior")
        assert hyperparameters and prior_file
        with torch.no_grad():
            batch_dims = data.remaining_dims(data_spatial_dim)
            hyps = decode_albert(model=model, label_log_prob=log_probs, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, batch_dims=batch_dims, prior_file=prior_file, train_lm=True)
        assert len(hyps[0]) == 1
        hyps = [_output_hyps(hyps_batch[0]) for hyps_batch in hyps]
        if len(hyps[0]) < 2:
            print("SHORT HYP:", hyps[0])
        lengths = [len(h) for h in hyps]
        lengths2 = [l + 1 for l in lengths]
        max_length = max(lengths)
        targets_spatial_dim = torch.tensor(lengths, dtype=torch.int32, device=data.raw_tensor.device)
        targets_spatial_dim = rf.convert_to_tensor(targets_spatial_dim, dims=(batch_dim,))
        targets_spatial_dim = Dim(targets_spatial_dim, name="out_spatial", dyn_size_ext=targets_spatial_dim)
        targets_spatial_dim2 = torch.tensor(lengths2, dtype=torch.int32, device=data.raw_tensor.device)
        targets_spatial_dim2 = rf.convert_to_tensor(targets_spatial_dim2, dims=(batch_dim,))
        targets_spatial_dim2 = Dim(targets_spatial_dim2, name="out_spatial2", dyn_size_ext=targets_spatial_dim2)
        hyps = [h + [0] * (max_length - len(h)) for h in hyps]
        hyps = torch.tensor(hyps, dtype=torch.int32, device=data.raw_tensor.device)
        targets = rf.convert_to_tensor(hyps, dims=(batch_dim, targets_spatial_dim), sparse_dim=model.target_dim)
    
    if nbest > 1 and weights is not None:
        weights = rf.log_softmax(weights, axis=weights.feature_dim)
        
        from .sum_criterion import safe_logaddexp
        tensor_ls, sizes_ls, combined_size = split_on_sep(targets.raw_tensor, targets_spatial_dim.dyn_size_ext.raw_tensor, model.target_dim, nbest, torch.exp(weights.raw_tensor))
        n = len(tensor_ls)
        
        loss_sum = None
        if aux_loss_layers:
            aux_probs = {}
            for i, layer_idx in enumerate(aux_loss_layers):
                aux_loss_sum = {}
                linear = getattr(model, f"enc_aux_logits_{layer_idx}")
                aux_logits = linear(collected_outputs[str(layer_idx - 1)])
                aux_probs[i] = model.log_probs_wb_from_logits(aux_logits)
        
        for j in range(n):
            targets_s = tensor_ls[j]
            targets_spatial_dim_s = sizes_ls[j]
            
            if config.bool("use_eos_postfix", False):
                targets_s, (targets_spatial_dim_s,) = rf.pad(
                    targets_s, axes=[targets_spatial_dim_s], padding=[(0, 1)], value=model.eos_idx
                )

            if aux_loss_layers:
                for i, layer_idx in enumerate(aux_loss_layers):
                    if layer_idx > len(model.encoder.layers):
                        continue
                    aux_loss = rf.ctc_loss(
                        logits=aux_probs[i],
                        logits_normalized=True,
                        targets=targets_s,
                        input_spatial_dim=enc_spatial_dim,
                        targets_spatial_dim=targets_spatial_dim_s,
                        blank_index=model.blank_idx,
                    )
                    if i in aux_loss_sum:
                        aux_loss_sum[i] = safe_logaddexp(aux_loss_sum[i], (-aux_loss).raw_tensor + weights.raw_tensor[:, j])
                    else:
                        aux_loss_sum[i] = (-aux_loss).raw_tensor + weights.raw_tensor[:, j]
                    

            loss = rf.ctc_loss(
                logits=log_probs,
                logits_normalized=True,
                targets=targets_s,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim_s,
                blank_index=model.blank_idx,
            )
            if loss_sum is not None:
                loss_sum = safe_logaddexp(loss_sum, (-loss).raw_tensor + weights.raw_tensor[:, j])
            else:
                loss_sum = (-loss).raw_tensor + weights.raw_tensor[:, j]
        if aux_loss_layers:
            for i, layer_idx in enumerate(aux_loss_layers):
                aux_loss_sum_i = rtf.TorchBackend.convert_to_tensor(-aux_loss_sum[i], dims = [batch_dim], dtype = "float32", name=f"ctc_aux_loss_{layer_idx}")
                aux_loss_sum_i.mark_as_loss(
                    f"ctc_{layer_idx}",
                    scale=aux_loss_scales[i],
                    custom_inv_norm_factor=combined_size.get_size_tensor(),
                    use_normalized_loss=use_normalized_loss,
                )
        loss_sum = rtf.TorchBackend.convert_to_tensor(-loss_sum, dims = [batch_dim], dtype = "float32", name=f"ctc_loss")
        loss_sum.mark_as_loss(
            "ctc",
            custom_inv_norm_factor=combined_size.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )
        return
        
    if config.bool("use_eos_postfix", False):
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_loss = rf.ctc_loss(
                logits=aux_log_probs,
                logits_normalized=True,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor() if not decode_every_step else targets_spatial_dim2.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )

    loss = rf.ctc_loss(
        logits=log_probs,
        logits_normalized=True,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    print(loss.raw_tensor[0])
    loss.mark_as_loss(
        "ctc",
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor() if not decode_every_step else targets_spatial_dim2.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )

    if model.decoder:
        # potentially also other types but just assume
        # noinspection PyTypeChecker
        decoder: TransformerDecoder = model.decoder

        input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
        )
        targets_w_eos, _ = rf.pad(
            targets,
            axes=[targets_spatial_dim],
            padding=[(0, 1)],
            value=model.eos_idx,
            out_dims=[targets_w_eos_spatial_dim],
        )

        batch_dims = data.remaining_dims(data_spatial_dim)
        logits, _ = model.decoder(
            input_labels,
            spatial_dim=targets_w_eos_spatial_dim,
            encoder=decoder.transform_encoder(enc, axis=enc_spatial_dim),
            state=model.decoder.default_initial_state(batch_dims=batch_dims),
        )

        logits_packed, pack_dim = rf.pack_padded(
            logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
        )
        targets_packed, _ = rf.pack_padded(
            targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )

        log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        loss.mark_as_loss("aed_ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

        best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name="aed_fer", as_error=True)


ctc_training: TrainDef[Model]
ctc_training.learning_rate_control_error_measure = "ctc"

def ctc_sum_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, lm_path: tk.Path, seq_tags: rf.Tensor = None, targets: rf.Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    from .sum_criterion import sum_loss, sum_loss2, safe_logsumexp, PrintGradients
    
    # torch.autograd.set_detect_anomaly(True)
    pg = PrintGradients.apply
    
    def _calc_log_prior(log_probs: torch.Tensor, lengths: torch.Tensor, use_max: bool = False, separate_eos: bool = False) -> torch.Tensor:
        lengths = lengths.to(log_probs.device)
        assert lengths.size(0) == log_probs.size(0), "Prior calculation batch lengths are not the same (full_sum)!"
        
        mask_bool = torch.arange(log_probs.size(1), device=log_probs.device).expand(log_probs.size(0), -1) < lengths.unsqueeze(1)
        mask = torch.where(mask_bool, 0.0, float("-inf"))
        mask = mask.unsqueeze(-1).expand(-1, -1, log_probs.size(2))
        log_probs = log_probs + mask
        
        sum_frames = lengths.sum()
        if use_max:
            if separate_eos:
                raise NotImplementedError("Separate EOS not implemented for max prior")
            else:
                argmaxs = log_probs.argmax(dim=2)
                argmaxs = argmaxs.flatten()
                argmaxs = argmaxs[mask_bool.flatten()]
                assert argmaxs.size(0) == sum_frames, f"Prior calculation frame count does not match (max) ({argmaxs.size(0)} != {sum_frames})"
                sum_probs = argmaxs.bincount(minlength=log_probs.size(2))
                sum_frames += (sum_probs == 0).sum()
                sum_probs = torch.where(sum_probs == 0, 1, sum_probs)
                log_sum_probs = sum_probs.log()
        else:
            if separate_eos:
                log_sum_probs = torch.full((log_probs.size(2) + 1,), float("-inf"), device=log_probs.device)
                log_sum_probs[1:-1] = safe_logsumexp(safe_logsumexp(log_probs[:,:,1:], dim=0), dim=0) # Sum over batch and time
                log_sum_probs[0] = safe_logsumexp(log_probs[:,0,0], dim=0) # BOS prob
                log_sum_probs[-1] = safe_logsumexp(safe_logsumexp(log_probs[:,1:,0], dim=0), dim=0) # EOS prob
            else:
                log_sum_probs = safe_logsumexp(safe_logsumexp(log_probs, dim=0), dim=0)
            
        log_mean_probs = log_sum_probs - sum_frames.log()
        
        with torch.no_grad():
            assert log_mean_probs.exp().sum().allclose(torch.tensor(1.0, device=log_mean_probs.device)), f"Prior probs do not sum to 1.0, but to {log_mean_probs.exp().sum()}"
            if log_mean_probs.isclose(torch.tensor([0.0], device=log_probs.device)).any() or log_mean_probs.isinf().any() or log_mean_probs.isnan().any():
                print("Prior probs contain inf or nan or 0 values!", log_mean_probs, log_mean_probs.exp())
        
        return log_mean_probs

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    
    am_scale = config.float("am_scale", 1.0)
    lm_scale = config.float("lm_scale", 1.0)
    prior_scale = config.float("prior_scale", 1.0)
    
    horizontal_prior = config.bool("horizontal_prior", True)
    blank_prior = config.bool("blank_prior", True)
    prior_gradient = config.bool("prior_gradient", True)
    empirical_prior = config.typed_value("empirical_prior", None)
    max_prior = config.bool("max_prior", False)
    top_k = config.int("top_k", 0)
    alignment_topk = config.bool("alignment_topk", True)
    blank_correction_version = config.int("blank_correction_version", 0)
    correction_in_final_score = config.bool("correction_in_final_score", False)
    use_prior = prior_scale > 0.0
    
    print_gradients = config.bool("print_gradients", False)
    version = config.int("version", 2)
    if version == 4:
        am_scale = 1.0
        lm_scale = 1.0
        prior_scale = 0.0
        use_prior = prior_scale > 0.0
        blank_correction_version = 16
        correction_in_final_score = True
        top_k = 1
        print_gradients = True
    
    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    if not lm_path.startswith("ffnn"):
        with uopen(lm_path, "rb") as f:
            lm = torch.load(f, map_location=data.device)
            assert isinstance(lm, torch.Tensor), "Loaded LM is not a tensor"
        lm_order = len(lm.size())
    else:
        assert model.train_language_model
        assert model.train_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.train_language_model
        lm_order = int(lm_path[len("ffnn"):])
        raise NotImplementedError("FFNN LM not implemented")

    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_log_probs = aux_log_probs.raw_tensor
            if use_prior:
                if empirical_prior is not None:
                    aux_log_prior = np.loadtxt(empirical_prior, dtype="float32")
                    aux_log_prior = torch.tensor(aux_log_prior, device=log_probs.device)
                    assert aux_log_prior.size(0) == log_probs.size(2), "Empirical prior size does not match (full_sum)!"
                else:
                    aux_log_prior = _calc_log_prior(aux_log_probs, enc_spatial_dim.dyn_size_ext.raw_tensor, use_max=max_prior)
                    if not prior_gradient:
                        aux_log_prior = aux_log_prior.detach()
            else:
                aux_log_prior = None
            # (B, T, F) -> (T, B, F)
            aux_log_probs = aux_log_probs.permute(1, 0, 2)
            aux_loss = sum_loss(
                log_probs=aux_log_probs,
                log_lm_probs=lm,
                log_prior=aux_log_prior,
                input_lengths=enc_spatial_dim.dyn_size_ext.raw_tensor,
                top_k=top_k,
                LM_order=lm_order,
                am_scale=am_scale,
                lm_scale=lm_scale,
                prior_scale=prior_scale,
                horizontal_prior=horizontal_prior,
                blank_prior=blank_prior,
                blank_idx=model.blank_idx,
                eos_idx=model.eos_idx,
                unk_idx=1,
                device=aux_log_probs.device,
                alignment_topk=alignment_topk
            )
            aux_loss = rtf.TorchBackend.convert_to_tensor(aux_loss, dims = [batch_dim], dtype = "float32", name=f"aux_full_sum_{layer_idx}")
            aux_loss.mark_as_loss(
                f"aux_full_sum_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
    
    fixed_seqs = ["train-other-500/5756-305214-0041/5756-305214-0041", "train-clean-360/2498-134786-0003/2498-134786-0003"] # MONICA DREW FRESH HOPE FROM HER SON'S WRITINGS THEY WERE FULL OF NOBLE THOUGHTS AND HIGH ASPIRATIONS, HERE IT IS
    print_for_idx = []
    
    # seq = seq_tags[0]
    # idx = np.where(seq_tags == seq)[0]
    # print_for_idx.append(idx[0])
    
    seq_tags = seq_tags.raw_tensor
    for seq in fixed_seqs:
        if seq in seq_tags:
            idx = np.where(seq_tags == seq)[0]
            print("Found seq", seq, enc_spatial_dim.dyn_size_ext.raw_tensor[idx])
            print_for_idx.append(idx[0])
            
    if print_gradients and fixed_seqs[1] in seq_tags:
        alias_name = config.typed_value("alias")
        idx_t = np.where(seq_tags == fixed_seqs[1])[0]
        print("Target:", targets.raw_tensor[idx_t].detach().cpu().numpy())
        logits_raw = logits.raw_tensor
        logits_raw = pg(logits_raw, "logits", alias_name, False, 1, idx_t)
        logits_raw = pg(logits_raw, "logits", alias_name, False, None, idx_t, [8, 9, 10, 11], ["<blank>", "H", "<blank>", "ERE"])
        logits.raw_tensor = logits_raw
        log_probs = model.log_probs_wb_from_logits(logits)
        log_probs = log_probs.raw_tensor
        log_probs = pg(log_probs, "log_probs", alias_name, False, 1, idx_t)
        log_probs = pg(log_probs, "log_probs", alias_name, False, None, idx_t, [8, 9, 10, 11], ["<blank>", "H", "<blank>", "ERE"])
    else:
        log_probs = model.log_probs_wb_from_logits(logits)
        log_probs = log_probs.raw_tensor

    if use_prior:
        if empirical_prior is not None:
            log_prior = np.loadtxt(empirical_prior, dtype="float32")
            log_prior = torch.tensor(log_prior, device=log_probs.device)
            assert log_prior.size(0) == log_probs.size(2), "Empirical prior size does not match (full_sum)!"
        else:
            log_prior = _calc_log_prior(log_probs, enc_spatial_dim.dyn_size_ext.raw_tensor, use_max=max_prior)
            if not prior_gradient:
                log_prior = log_prior.detach()
    else:
        log_prior = None
    # (B, T, V) -> (T, B, V)
    log_probs = log_probs.permute(1, 0, 2)
    
    if version == 5:
        loss = torch.ctc_loss(
            log_probs,
            targets.raw_tensor,
            enc_spatial_dim.dyn_size_ext.raw_tensor,
            targets_spatial_dim.dyn_size_ext.raw_tensor,
            blank=model.blank_idx,
            reduction=0,
            zero_infinity=False
        )
        if print_gradients and fixed_seqs[1] in seq_tags:
            print("Loss:", loss[np.where(seq_tags == fixed_seqs[1])[0]].detach().cpu().numpy()) # 0: [6.9210505], 0.0009867928 , 1: [0.00390251], 0.99610509
        loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"ctc")
        loss.mark_as_loss(
            "ctc",
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )
    elif version in [10, 11]:
        if version == 10:
            loss = torch.logsumexp(log_probs[0], dim=-1)
        else:
            loss = safe_logsumexp(log_probs[0], -1)
        for t in range(1, log_probs.size(0)):
            if version == -1:
                A = loss.unsqueeze(1)
                B = log_probs[t].unsqueeze(1).expand(-1, log_probs.size(-1), log_probs.size(-1))
                new_loss = A.matmul(B).squeeze(1)
                time_mask = (t < enc_spatial_dim.dyn_size_ext.raw_tensor.to(log_probs.device)).unsqueeze(-1)
                loss = torch.where(time_mask.expand_as(new_loss), new_loss, loss)
            else:
                # A = loss.unsqueeze(1).expand(-1, log_probs.size(-1), log_probs.size(-1))
                # B = log_probs[t].unsqueeze(-1).expand(-1, log_probs.size(-1), log_probs.size(-1))
                # new_loss = safe_logsumexp(A + B, dim=-1)
                # new_loss = safe_logsumexp(torch.stack([loss, safe_logsumexp(log_probs[t], -1)], dim=-1), dim=-1)
                if version == 10:
                    new_loss = loss + torch.logsumexp(log_probs[t], -1)
                else:
                    new_loss = loss + safe_logsumexp(log_probs[t], -1)
                time_mask = (t < enc_spatial_dim.dyn_size_ext.raw_tensor.to(log_probs.device))#.unsqueeze(-1)
                loss = torch.where(time_mask.expand_as(new_loss), new_loss, loss)
        if version == -1:
            loss = loss.sum(-1)
        # else:
        #     loss = safe_logsumexp(loss, dim=-1)
        loss = -loss
        # print(loss[0].detach().cpu().numpy())
        loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"sum")
        loss.mark_as_loss(
            f"sum",
            custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )
    else:
        loss = sum_loss2(
            log_probs=log_probs,
            log_lm_probs=lm,
            log_prior=log_prior,
            input_lengths=enc_spatial_dim.dyn_size_ext.raw_tensor,
            top_k=top_k,
            LM_order=lm_order,
            am_scale=am_scale,
            lm_scale=lm_scale,
            prior_scale=prior_scale,
            horizontal_prior=horizontal_prior,
            blank_prior=blank_prior,
            blank_idx=model.blank_idx,
            eos_idx=model.eos_idx,
            unk_idx=1,
            device=log_probs.device,
            print_best_path_for_idx=print_for_idx,
            alignment_topk=alignment_topk,
            blank_correction_version=blank_correction_version,
            correction_in_final_score = correction_in_final_score
        )
        if print_gradients and fixed_seqs[1] in seq_tags:
            print("Loss:", loss[np.where(seq_tags == fixed_seqs[1])[0]].detach().cpu().numpy()) # 0: [6.9392214] 0.0009690238, 1: [0.01604532], 0.984082720
        loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"full_sum")
        loss.mark_as_loss(
            f"full_sum",
            custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )

ctc_sum_training: ExtendedTrainDef[Model]
ctc_sum_training.learning_rate_control_error_measure = "full_sum"


#---------------------------------------------------------------------------------------------------------------------------------------
# RECOG DEFINITION

def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, Vocab
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
        label_log_prob, k_dim=Dim(beam_size, name=f"pre-filter-beam"), axis=[model.wb_target_dim]
    )  # seq_log_prob, backrefs_global: Batch, Spatial, PreFilterBeam. backrefs_pre_filter -> Vocab
    label_log_prob_pre_filter_ta = TensorArray.unstack(
        label_log_prob_pre_filter, axis=enc_spatial_dim
    )  # t -> Batch, PreFilterBeam
    backrefs_pre_filter_ta = TensorArray.unstack(backrefs_pre_filter, axis=enc_spatial_dim)  # t -> Batch, PreFilterBeam

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets = []
    seq_backrefs = []
    for t in range(max_seq_len):
        # Filter out finished beams
        seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]  # Batch, InBeam, PreFilterBeam
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, pre_filter_beam_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> PreFilterBeam.
        target = rf.gather(backrefs_pre_filter_ta[t], indices=target)  # Batch, Beam -> Vocab
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = enc_spatial_dim
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = OUT_BLANK_LABEL
model_recog.batch_size_dependent = False  # not totally correct, but we treat it as such...

def model_recog_lm(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    arpa_4gram_lm: Optional[str],
    lexicon: str,
    hyperparameters: dict,
    prior_file: tk.Path = None
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.
    
    Uses a 4gram LM and beam search.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from torchaudio.models.decoder import ctc_decoder
    import torch
    import json
    from returnn.util.basic import cf
    from i6_experiments.users.schmitt.experiments.exp2025_03_10_ctc_usr.lm.ffnn import FFNN_LM_flashlight
    
    # Get the logits from the model
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    
    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
    greedy = hyp_params.pop("greedy", False)
    prior_weight = hyp_params.pop("prior_weight", 0.0)
    prior_weight_tune = hyp_params.pop("prior_weight_tune", None)
    lm_weight_tune = hyp_params.pop("lm_weight_tune", None)
    use_logsoftmax = hyp_params.pop("use_logsoftmax", False)
    
    if prior_weight_tune:
        prior_weight_tune = json.load(open(prior_weight_tune))
        prior_weight_tune = prior_weight_tune["best_tune"]
        assert type(prior_weight_tune) == float, "Prior weight tune is not a float!"
        print(f"Prior weight with tune: {prior_weight} + {prior_weight_tune} = {prior_weight + prior_weight_tune}")
        prior_weight += prior_weight_tune
    if lm_weight_tune:
        lm_weight_tune = json.load(open(lm_weight_tune))
        lm_weight_tune = lm_weight_tune["best_tune"]
        assert type(lm_weight_tune) == float, "LM weight tune is not a float!"
        old_lm_weight = hyp_params.get("lm_weight", 0.0)
        print(f"LM weight with tune: {old_lm_weight} + {lm_weight_tune} = {old_lm_weight + lm_weight_tune}")
        hyp_params["lm_weight"] = old_lm_weight + lm_weight_tune
    
    if greedy:
        use_logsoftmax = True
    
    if use_logsoftmax:
        label_log_prob = model.log_probs_wb_from_logits(logits)
        label_log_prob = label_log_prob.raw_tensor.cpu()
    
        # Subtract prior of labels if available
        if prior_file and prior_weight > 0.0:
            prior = np.loadtxt(prior_file, dtype="float32")
            label_log_prob -= prior_weight * prior
            print("We subtracted the prior!")
    elif prior_file and prior_weight > 0.0:
        print("Cannot subtract prior without running log softmax")
        return None
    
    if greedy:
        probs, greedy_res = torch.max(label_log_prob, dim=-1)
        greedy_res = greedy_res.unsqueeze(1)
        
        scores = torch.sum(probs, dim=-1)
        scores = scores.unsqueeze(1)
        
        beam_dim = rtf.TorchBackend.get_new_dim_raw(greedy_res, 1, name="beam_dim")
        dims = [batch_dim, beam_dim, enc_spatial_dim]
        hyps = rtf.TorchBackend.convert_to_tensor(greedy_res, dims = dims, sparse_dim=model.wb_target_dim, dtype = "int64", name="hyps")
        
        dims = [batch_dim, beam_dim]
        scores = Tensor("scores", dims = dims, dtype = "float32", raw_tensor = scores)
        
        return hyps, scores, enc_spatial_dim, beam_dim
    
    if arpa_4gram_lm:
        arpa_4gram_lm = str(cf(arpa_4gram_lm))
    else:
        assert lm_name.startswith("ffnn")
        assert model.recog_language_model
        assert model.recog_language_model.vocab_dim == model.target_dim
        context_size = int(lm_name[len("ffnn"):])
        arpa_4gram_lm = FFNN_LM_flashlight(model.recog_language_model, model.recog_language_model.vocab_dim, context_size)
    
    use_lm = hyp_params.pop("use_lm", True)
    use_lexicon = hyp_params.pop("use_lexicon", True)
    
    configs = {
        "tokens": list(model.wb_target_dim.vocab.labels),
        "blank_token": OUT_BLANK_LABEL,
        "sil_token": OUT_BLANK_LABEL,
        "unk_word": "<unk>",
        "beam_size_token": None, # 16
        "beam_threshold": 1000000, # 14
    }
    configs["lexicon"] = lexicon if use_lexicon else None
    configs["lm"] = arpa_4gram_lm if use_lm else None
    
    configs.update(hyp_params)
    
    assert "ps_nbest" not in configs, "We only support nbest == 1"
    
    decoder = ctc_decoder(**configs)
    enc_spatial_dim_torch = enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()
    if use_logsoftmax:
        decoder_results = decoder(label_log_prob, enc_spatial_dim_torch)
    else:
        decoder_results = decoder(logits.raw_tensor.cpu(), enc_spatial_dim_torch)
    
    if use_lexicon:
        print("Use words directly!")
        if CHECK_DECODER_CONSISTENCY:
            for l1 in decoder_results:
                for l2 in l1:
                    lexicon_words = " ".join(l2.words)
                    token_words = " ".join([configs["tokens"][t] for t in l2.tokens])
                    assert not token_words.endswith("@@"), f"Token words ends with @@: {token_words}, Lexicon words: {lexicon_words}"
                    token_words = token_words.replace("@@ ", "")
                    assert lexicon_words == token_words, f"Words don't match: Lexicon words: {lexicon_words}, Token words: {token_words}"
        
        words = [[" ".join(l2.words) for l2 in l1] for l1 in decoder_results]
        words = np.array(words)
        words = np.expand_dims(words, axis=2)
        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        scores = torch.tensor(scores)
        
        beam_dim = Dim(words.shape[1], name="beam_dim")
        enc_spatial_dim = Dim(1, name="spatial_dim")
        words = rf._numpy_backend.NumpyBackend.convert_to_tensor(words, dims = [batch_dim, beam_dim, enc_spatial_dim], dtype = "string", name="hyps")
        scores = Tensor("scores", dims = [batch_dim, beam_dim], dtype = "float32", raw_tensor = scores)
        
        return words, scores, enc_spatial_dim, beam_dim
    else:
        def _pad_blanks(tokens, max_len):
            tokens = tokens[1:-1]
            if len(tokens) < max_len:
                # print("We had to pad blanks")
                tokens = torch.cat([tokens, torch.tensor([model.blank_idx] * (max_len - len(tokens)))])
            return tokens
        
        def _pad_lists(t, max_len, max_len2):
            if t.shape[0] < max_len2:
                print("We had to pad the list")
                t = torch.cat([t, torch.tensor([[model.blank_idx] * max_len] * (max_len2 - t.shape[0]))])
            return t
        
        def _pad_scores(l, max_len):
            l = torch.tensor(l)
            if len(l) < max_len:
                print("We had to pad scores")
                l = torch.cat([l, torch.tensor([-1000000.0] * (max_len - len(l)))])
            return l
            
        max_length = int(enc_spatial_dim_torch.max())
        hyps = [torch.stack([_pad_blanks(l2.tokens, max_length) for l2 in l1]) for l1 in decoder_results]
        max_length_2 = max([l.shape[0] for l in hyps])
        hyps = [_pad_lists(t, max_length, max_length_2) for t in hyps]
        hyps = torch.stack(hyps)
        beam_dim = rtf.TorchBackend.get_new_dim_raw(hyps, 1, name="beam_dim")
        dims = [batch_dim, beam_dim, enc_spatial_dim]
        hyps = rtf.TorchBackend.convert_to_tensor(hyps, dims = dims, sparse_dim=model.wb_target_dim, dtype = "int64", name="hyps")
        
        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        max_length_3 = max([len(l) for l in scores])
        scores = torch.stack([_pad_scores(l, max_length_3) for l in scores])
        dims = [batch_dim, beam_dim]
        scores = Tensor("scores", dims = dims, dtype = "float32", raw_tensor = scores)
        
        # print(f"CUSTOM seq_targets: {hyps} \n{hyps.raw_tensor.cpu()},\nscores: {scores} \n{scores.raw_tensor.cpu()}n {scores.raw_tensor.cpu()[0][0]},\nspatial_dim: {enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()},\n beam_size: {beam_dim}")
        
        return hyps, scores, enc_spatial_dim, beam_dim

# RecogDef API
model_recog_lm: RecogDef[Model]
model_recog_lm.output_with_beam = True
model_recog_lm.output_blank_label = OUT_BLANK_LABEL
model_recog_lm.batch_size_dependent = False  # not totally correct, but we treat it as such...


def model_recog_flashlight(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    hyperparameters: dict,
    prior_file: tk.Path = None,
    # version: Optional[int] = None,
    seq_tags: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    assert data.dims_set == {batch_dim, data_spatial_dim, data.feature_dim}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert logits.dims_set == {batch_dim, enc_spatial_dim, model.wb_target_dim}

    # The label log probs include the AM
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    
    print_idx = []
    
    return decode_flashlight(model=model, label_log_prob=label_log_prob, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, prior_file=prior_file, print_idx=print_idx)


# RecogDef API
model_recog_flashlight: RecogDef[Model]
model_recog_flashlight.output_with_beam = True
model_recog_flashlight.output_blank_label = OUT_BLANK_LABEL
model_recog_flashlight.batch_size_dependent = True  # our models currently just are batch-size-dependent...

def _output_hyps(hyp: list, model: Model) -> str:
    prev = None
    ls = []
    for h in hyp:
        if h != prev:
            ls.append(h)
            prev = h
    ls = [model.target_dim.vocab.id_to_label(h) for h in ls if h != model.blank_idx]
    s = " ".join(ls).replace("@@ ", "")
    if s.endswith("@@"):
        s = s[:-2]
    return s

def decode_flashlight(
    *,
    model: Model,
    label_log_prob: Tensor,
    enc_spatial_dim: Dim,
    hyperparameters: dict,
    prior_file: tk.Path = None,
    train_lm = False,
    print_idx = []
) -> Tuple[Tensor, Tensor, Dim, Dim] | list:
    from dataclasses import dataclass
    import torch
    import json
    from flashlight.lib.text.decoder import LM, LMState
    from i6_experiments.users.schmitt.util.lru_cache import lru_cache
    from returnn.util import basic as util

    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
    prior_weight = hyp_params.pop("prior_weight", 0.0)
    prior_weight_tune = hyp_params.pop("prior_weight_tune", None)
    lm_weight_tune = hyp_params.pop("lm_weight_tune", None)
    
    if prior_weight_tune:
        prior_weight_tune = json.load(open(prior_weight_tune))
        prior_weight_tune = prior_weight_tune["best_tune"]
        assert type(prior_weight_tune) == float, "Prior weight tune is not a float!"
        print(f"Prior weight with tune: {prior_weight} + {prior_weight_tune} = {prior_weight + prior_weight_tune}")
        prior_weight += prior_weight_tune
    if lm_weight_tune:
        lm_weight_tune = json.load(open(lm_weight_tune))
        lm_weight_tune = lm_weight_tune["best_tune"]
        assert type(lm_weight_tune) == float, "LM weight tune is not a float!"
        old_lm_weight = hyp_params.get("lm_weight", 0.0)
        print(f"LM weight with tune: {old_lm_weight} + {lm_weight_tune} = {old_lm_weight + lm_weight_tune}")
        hyp_params["lm_weight"] = old_lm_weight + lm_weight_tune

    n_best = hyp_params.pop("ps_nbest", 1)
    beam_size = hyp_params.pop("beam_size", 1)
    beam_size_token = hyp_params.pop("beam_size_token", model.wb_target_dim.vocab.num_labels)
    beam_threshold = hyp_params.pop("beam_threshold", 1000000)
    log_add = hyp_params.pop("log_add", False)

    # Eager-mode implementation of beam search using Flashlight.

    # noinspection PyUnresolvedReferences
    assert lm_name.startswith("ffnn")
    context_size = int(lm_name[len("ffnn"):])
    if train_lm:
        assert model.train_language_model
        assert model.train_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.train_language_model
    else:
        assert model.recog_language_model
        assert model.recog_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.recog_language_model
    # noinspection PyUnresolvedReferences
    lm_scale: float = hyp_params["lm_weight"]

    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)

    total_mem = None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        _, total_mem = torch.cuda.mem_get_info(dev if dev.index is not None else None)

    def _collect_mem_stats():
        if dev.type == "cuda":
            return [
                f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
            ]
        return ["(unknown)"]

    # print(
    #     f"Memory usage {dev_s} before encoder forward:",
    #     " ".join(_collect_mem_stats()),
    #     "total:",
    #     util.human_bytes_size(total_mem) if total_mem else "(unknown)",
    # )

    lm_initial_state = lm.default_initial_state(batch_dims=[])

    # https://github.com/flashlight/text/tree/main/bindings/python#decoding-with-your-own-language-model
    # https://github.com/facebookresearch/fairseq/blob/main/examples/speech_recognition/new/decoders/flashlight_decoder.py
    # https://github.com/pytorch/audio/blob/main/src/torchaudio/models/decoder/_ctc_decoder.py

    # The current implementation of FlashlightLM below assumes we can just use the token_idx as-is for the LM.
    assert model.blank_idx == model.target_dim.dimension

    @dataclass
    class FlashlightLMState:
        def __init__(self, label_seq: List[int], prev_state: LMState):
            if len(label_seq) > context_size:
                self.label_seq = label_seq[-context_size:]
            else:
                self.label_seq = label_seq
            assert len(self.label_seq) == context_size
            self.prev_state = prev_state

    # Use LRU cache for the LM states (on GPU) and log probs.
    # Note that additionally to the cache size limit here,
    # we free more when we run out of CUDA memory.
    start_lru_cache_size = 1024
    max_used_mem_fraction = 0.9

    class FlashlightLM(LM):
        def __init__(self):
            super().__init__()
            # Cannot use weakrefs because the LMState object will always be recreated on-the-fly,
            # i.e. the Python object does not persist.
            self.mapping_states: Dict[LMState, FlashlightLMState] = {}
            self._count_recalc_whole_seq = 0
            self._recent_debug_log_time = -sys.maxsize
            self._max_used_mem_fraction = max_used_mem_fraction

        def reset(self):
            self.mapping_states.clear()
            self._count_recalc_whole_seq = 0
            self._recent_debug_log_time = -sys.maxsize
            self._max_used_mem_fraction = max_used_mem_fraction
            self._calc_next_lm_state.cache_clear()
            self._calc_next_lm_state.cache_set_maxsize(start_lru_cache_size)

        @lru_cache(maxsize=start_lru_cache_size)
        def _calc_next_lm_state(self, state: LMState) -> Tuple[Any, torch.Tensor]:
            """
            :return: LM state, log probs [Vocab]
            """
            state_ = self.mapping_states[state]
            
            lm_logits, lm_state = None, None
            while True:
                self._cache_maybe_free_memory()
                try:
                    self._count_recalc_whole_seq += 1
                    spatial_dim = Dim(len(state_.label_seq), name="seq")
                    out_spatial_dim = Dim(context_size + 1, name="seq_out")
                    lm_logits, lm_state = lm(
                        rf.convert_to_tensor(state_.label_seq, dims=[spatial_dim], sparse_dim=model.target_dim),
                        spatial_dim=spatial_dim,
                        out_spatial_dim=out_spatial_dim,
                        state=lm_initial_state,
                    )  # Vocab / ...
                    lm_logits = rf.gather(lm_logits, axis=out_spatial_dim, indices=rf.last_frame_position_of_dim(out_spatial_dim))
                except torch.cuda.OutOfMemoryError as exc:
                    if self._calc_next_lm_state.cache_len() == 0:
                        raise  # cannot free more
                    print(f"{type(exc).__name__}: {exc}")
                    new_max_used_mem_fraction = max(0.2, self._max_used_mem_fraction - 0.1)
                    if new_max_used_mem_fraction != self._max_used_mem_fraction:
                        print(f"Reduce max used mem fraction to {new_max_used_mem_fraction:.0%}")
                    continue  # try again
                break
            assert lm_logits.dims == (model.target_dim,)
            lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Vocab
            log_probs_raw = lm_log_probs.raw_tensor.cpu()
            return lm_state, log_probs_raw

        def _cache_maybe_free_memory(self):
            if dev.type == "cuda":
                # Maybe check if we should free some more memory.
                count_pop = 0
                used_mem = 0
                while self._calc_next_lm_state.cache_len() > 0:
                    used_mem = torch.cuda.memory_reserved(dev)
                    if used_mem / total_mem < self._max_used_mem_fraction:
                        break
                    # Check again after trying to empty the cache.
                    # Note: gc.collect() is problematic here because of how Flashlight handles the states:
                    # We have millions of Python objects in the mapping_states dict,
                    # which takes a very long time to go through.
                    torch.cuda.empty_cache()
                    used_mem = torch.cuda.memory_reserved(dev)
                    if used_mem / total_mem < self._max_used_mem_fraction:
                        break
                    self._calc_next_lm_state.cache_pop_oldest()
                    count_pop += 1
                if count_pop > 0:
                    print(
                        f"Pop {count_pop} states from cache,"
                        f" cache size {self._calc_next_lm_state.cache_len()},"
                        f" reached {used_mem / total_mem:.1%} of total mem,"
                        f" mem usage {dev_s}: {' '.join(_collect_mem_stats())}"
                    )
                    self._calc_next_lm_state.cache_set_maxsize(self._calc_next_lm_state.cache_len())

        def start(self, start_with_nothing: bool):
            """
            Parameters:
                start_with_nothing (bool): whether or not to start sentence with sil token.
            """
            start_with_nothing  # noqa  # not sure how to handle this?
            self.reset()
            state = LMState()
            self.mapping_states[state] = FlashlightLMState(label_seq=[model.bos_idx]*context_size, prev_state=state)
            return state

        def score(self, state: LMState, token_index: int):
            """
            Evaluate language model based on the current lm state and new word

            Parameters:
                state: current lm state
                token_index: index of the word
                            (can be lexicon index then you should store inside LM the
                            mapping between indices of lexicon and lm, or lm index of a word)

            Returns:
                (LMState, float): pair of (new state, score for the current word)
            """
            state_ = self.mapping_states[state]
            # if time.monotonic() - self._recent_debug_log_time > 1:
            #     print(
            #         "LM prefix",
            #         [model.target_dim.vocab.id_to_label(label_idx) for label_idx in state_.label_seq],
            #         f"score {model.target_dim.vocab.id_to_label(token_index)!r}",
            #         f"({len(self.mapping_states)} states seen)",
            #         f"(cache info {self._calc_next_lm_state.cache_info()})",
            #         f"(mem usage {dev_s}: {' '.join(_collect_mem_stats())})",
            #     )
            #     self._recent_debug_log_time = time.monotonic()
            outstate = state.child(token_index)
            if outstate not in self.mapping_states:
                self.mapping_states[outstate] = FlashlightLMState(
                    label_seq=state_.label_seq + [token_index], prev_state=state
                )

            _, log_probs_raw = self._calc_next_lm_state(state)
            return outstate, log_probs_raw[token_index]

        def finish(self, state: LMState):
            """
            Evaluate eos for language model based on the current lm state

            Returns:
                (LMState, float): pair of (new state, score for the current word)
            """
            return self.score(state, model.eos_idx)

    fl_lm = FlashlightLM()

    from flashlight.lib.text.decoder import LexiconFreeDecoderOptions, LexiconFreeDecoder, CriterionType

    fl_decoder_opts = LexiconFreeDecoderOptions(
        beam_size=beam_size,
        beam_size_token=beam_size_token,
        beam_threshold=beam_threshold,
        lm_weight=lm_scale,
        sil_score=0.0,
        log_add=log_add,
        criterion_type=CriterionType.CTC,
    )
    fl_decoder = LexiconFreeDecoder(fl_decoder_opts, fl_lm, -1, model.blank_idx, [])

    # Subtract prior of labels if available
    if prior_file and prior_weight > 0.0:
        prior = np.loadtxt(prior_file, dtype="float32")
        prior *= prior_weight
        prior = torch.tensor(prior, dtype=torch.float32, device=dev)
        prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
        label_log_prob = label_log_prob - prior
        # print("We subtracted the prior!")
    
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob = label_log_prob.copy_transpose((batch_dim, enc_spatial_dim, model.wb_target_dim))
    batch_size, max_seq_len = label_log_prob.raw_tensor.shape[:2]
    assert enc_spatial_dim.dyn_size_ext.dims == (batch_dim,)

    label_log_prob = rf.cast(label_log_prob, "float32")
    label_log_prob = rf.copy_to_device(label_log_prob, "cpu")
    label_log_prob_raw = label_log_prob.raw_tensor.contiguous()
    float_bytes = 4

    # print(f"Memory usage {dev_s} after encoder forward:", " ".join(_collect_mem_stats()))

    hyps = []
    scores = []
    for batch_idx in range(batch_size):
        emissions_ptr = label_log_prob_raw.data_ptr() + float_bytes * batch_idx * label_log_prob_raw.stride(0)
        seq_len = enc_spatial_dim.dyn_size[batch_idx]
        assert seq_len <= max_seq_len
        results = fl_decoder.decode(emissions_ptr, seq_len, model.wb_target_dim.dimension)
        # I get -1 (silence label?) at the beginning and end in the tokens? Filter those away.
        # These are also additional frames which don't correspond to the input frames?
        # When removing those two frames, the len of tokens (align labels) matches the emission frames
        # (as it should be).
        hyps_per_batch = [[label for label in result.tokens if label >= 0] for result in results]
        scores_per_batch = [result.score for result in results]
        # print(
        #     f"batch {batch_idx + 1}/{batch_size}: {len(results)} hyps,"
        #     f" best score: {scores_per_batch[0]},"
        #     f" best seq {_format_align_label_seq(results[0].tokens, model.wb_target_dim)},"
        #     f" worst score: {scores_per_batch[-1]},"
        #     f" LM cache info {fl_lm._calc_next_lm_state.cache_info()},"
        #     f" LM recalc whole seq count {fl_lm._count_recalc_whole_seq},"
        #     f" mem usage {dev_s}: {' '.join(_collect_mem_stats())}"
        # )
        assert all(
            len(hyp) == seq_len for hyp in hyps_per_batch
        ), f"seq_len {seq_len}, hyps lens {[len(hyp) for hyp in hyps_per_batch]}"
        if print_idx:
            for idx in print_idx:
                if idx == batch_idx:
                    print(f"RES: {hyps_per_batch}, {scores_per_batch}")
        
        if len(results) >= n_best:
            if n_best > 1:
                # We have to select the n_best on output level
                hyps_shortened = [_output_hyps(hyp) for hyp in hyps_per_batch]
                nbest_hyps = []
                nbest_hyps_ids = []
                k = 0
                i = 0
                while k < n_best:
                    if i >= len(hyps_shortened):
                        break
                    if hyps_shortened[i] not in nbest_hyps:
                        nbest_hyps.append(hyps_shortened[i])
                        nbest_hyps_ids.append(i)
                        k += 1
                    i += 1
                hyps_per_batch = [hyps_per_batch[id] for id in nbest_hyps_ids]
                scores_per_batch = [scores_per_batch[id] for id in nbest_hyps_ids]
                
                if len(hyps_per_batch) < n_best:
                    print("Not enough n-best")
                    hyps_per_batch += [[]] * (n_best - len(hyps_per_batch))
                    scores_per_batch += [-1e30] * (n_best - len(scores_per_batch))
            else:
                hyps_per_batch = hyps_per_batch[:n_best]
                scores_per_batch = scores_per_batch[:n_best]
        else:
            hyps_per_batch += [[]] * (n_best - len(results))
            scores_per_batch += [-1e30] * (n_best - len(results))
        assert len(hyps_per_batch) == len(scores_per_batch) == n_best
        hyps_per_batch = [hyp + [model.blank_idx] * (max_seq_len - len(hyp)) for hyp in hyps_per_batch]
        assert all(len(hyp) == max_seq_len for hyp in hyps_per_batch)
        hyps.append(hyps_per_batch)
        scores.append(scores_per_batch)
    fl_lm.reset()
    hyps_pt = torch.tensor(hyps, dtype=torch.int32)
    assert hyps_pt.shape == (batch_size, n_best, max_seq_len)
    scores_pt = torch.tensor(scores, dtype=torch.float32)
    assert scores_pt.shape == (batch_size, n_best)

    beam_dim = Dim(n_best, name="beam")
    out_spatial_dim = enc_spatial_dim
    hyps_r = rf.convert_to_tensor(hyps_pt, dims=(batch_dim, beam_dim, out_spatial_dim), sparse_dim=model.wb_target_dim)
    scores_r = rf.convert_to_tensor(scores_pt, dims=(batch_dim, beam_dim))
    print(f"Memory usage ({dev_s}) after batch:", " ".join(_collect_mem_stats()))
    if train_lm:
        return hyps
    else:
        return hyps_r, scores_r, out_spatial_dim, beam_dim
    
def model_recog_lm_albert(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    hyperparameters: dict,
    prior_file: tk.Path = None,
    version: Optional[int] = None,
    seq_tags: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Note, for debugging, see :func:`model_recog_debug` below.

    Note, some potential further improvements:
    There are many align label seqs which correspond to the same label seq,
    but the LM score is calculated for each of them.
    We could make this somehow unique depending on the label seq.
    (But unclear how exactly to do this in a GPU friendly, batched way.)

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    assert data.dims_set == {batch_dim, data_spatial_dim, data.feature_dim}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert logits.dims_set == {batch_dim, enc_spatial_dim, model.wb_target_dim}
    
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))

    # The label log probs include the AM
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    
    seq_tags = seq_tags.raw_tensor
    print_idx = []
    if version == 7:
        for seq in ["dev-other/1630-96099-0024/1630-96099-0024"]:
            if seq in seq_tags:
                idx = np.where(seq_tags == seq)[0]
                print_idx.append(idx)
    
    return decode_albert(model=model, label_log_prob=label_log_prob, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, batch_dims=batch_dims, prior_file=prior_file, version=version, print_idx=print_idx)

# RecogDef API
model_recog_lm_albert: RecogDef[Model]
model_recog_lm_albert.output_with_beam = True
model_recog_lm_albert.output_blank_label = OUT_BLANK_LABEL
model_recog_lm_albert.batch_size_dependent = True  # our models currently just are batch-size-dependent...

def decode_albert(
    *,
    model: Model,
    label_log_prob: Tensor,
    enc_spatial_dim: Dim,
    hyperparameters: dict,
    batch_dims: List[Dim],
    prior_file: tk.Path = None,
    train_lm = False,
    version: int = 1,
    print_idx = []
):
    import json
    
    def _update_context(context: Tensor, new_label: Tensor, context_dim: Dim) -> Tensor:
        new_dim = Dim(1, name="new_label")
        new_label = rf.expand_dim(new_label, dim=new_dim)
        old_context, old_context_dim = rf.slice(context, axis=context_dim, start=1)
        new_context, new_context_dim = rf.concat((old_context, old_context_dim), (new_label, new_dim), out_dim=context_dim)
        assert new_context_dim == context_dim
        return new_context
    
    def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
        assert target.sparse_dim == wb_target_dim
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        return rf.set_sparse_dim(target, target_dim)


    def _target_dense_extend_blank(
        target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
    ) -> Tensor:
        assert target_dim in target.dims
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
        return res

    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
    prior_weight = hyp_params.pop("prior_weight", 0.0)
    prior_weight_tune = hyp_params.pop("prior_weight_tune", None)
    lm_weight_tune = hyp_params.pop("lm_weight_tune", None)
    
    if prior_weight_tune:
        prior_weight_tune = json.load(open(prior_weight_tune))
        prior_weight_tune = prior_weight_tune["best_tune"]
        assert type(prior_weight_tune) == float, "Prior weight tune is not a float!"
        print(f"Prior weight with tune: {prior_weight} + {prior_weight_tune} = {prior_weight + prior_weight_tune}")
        prior_weight += prior_weight_tune
    if lm_weight_tune:
        lm_weight_tune = json.load(open(lm_weight_tune))
        lm_weight_tune = lm_weight_tune["best_tune"]
        assert type(lm_weight_tune) == float, "LM weight tune is not a float!"
        old_lm_weight = hyp_params.get("lm_weight", 0.0)
        print(f"LM weight with tune: {old_lm_weight} + {lm_weight_tune} = {old_lm_weight + lm_weight_tune}")
        hyp_params["lm_weight"] = old_lm_weight + lm_weight_tune

    n_best = hyp_params.pop("ps_nbest", 1)
    assert n_best == 1, "n-best not implemented yet"
    beam_size = hyp_params.pop("beam_size", 1)
    
    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    import returnn
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__
    
    # Subtract prior of labels if available
    if prior_file and prior_weight > 0.0:
        prior = np.loadtxt(prior_file, dtype="float32")
        prior *= prior_weight
        prior = torch.tensor(prior, dtype=torch.float32, device=dev)
        prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
        label_log_prob = label_log_prob - prior
        # print("We subtracted the prior!")
        
    assert lm_name.startswith("ffnn")
    context_size = int(lm_name[len("ffnn"):])
    if train_lm:
        assert model.train_language_model
        assert model.train_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.train_language_model
    else:
        assert model.recog_language_model
        assert model.recog_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.recog_language_model
    # noinspection PyUnresolvedReferences
    lm_scale: float = hyp_params["lm_weight"]

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    context_dim = Dim(context_size, name="context")
    lm_out_dim = Dim(context_size + 1, name="context+1")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_ta = TensorArray.unstack(label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    target = rf.constant(model.bos_idx, dims=batch_dims_ + [context_dim], sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB

    lm_state = lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
    lm_logits, lm_state = lm(
        target,
        spatial_dim=context_dim,
        out_spatial_dim=lm_out_dim,
        state=lm_state,
    )  # Batch, InBeam, Vocab / ...
    lm_logits = rf.gather(lm_logits, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
    assert lm_logits.dims == (*batch_dims_, model.target_dim)
    lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
    lm_log_probs *= lm_scale
    
    if version == 7:
        if print_idx:
            flashlight_align = [184, 184, 184, 24, 184, 9, 184, 75, 74, 184, 184, 24, 184, 7, 184, 106, 184, 18, 184, 184, 184, 42, 184, 184, 24, 184, 184, 78, 11, 184, 126, 184, 184, 108, 184, 184, 184, 184, 184, 184, 184, 130, 184, 9, 121, 184, 184, 114, 184, 184, 184, 141, 184, 184, 10, 184, 184, 184, 118, 184, 184, 184, 184, 184, 184, 27, 184, 46, 184, 156, 184, 184, 28, 17, 2, 74, 184, 184, 184, 14, 15, 29, 184, 12, 30, 184, 184, 24, 184, 3, 184, 119, 184, 184, 184, 175, 184, 184, 184, 184, 20, 184, 184, 184, 184, 12, 84, 184, 40, 184, 184, 184, 184, 30, 184, 184, 152, 14, 14, 5, 184, 184, 135, 184, 184, 184, 184, 184, 184, 184, 184, 184]
            fixed_lm_state = [model.bos_idx] * 8
            fixed_seq_prob = torch.zeros(1, dtype=torch.float32)

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = seq_log_prob + label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
        if lm_scale > 0.0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seq_log_prob += rf.where(
                    (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
                    _target_dense_extend_blank(
                        lm_log_probs,
                        target_dim=model.target_dim,
                        wb_target_dim=model.wb_target_dim,
                        blank_idx=model.blank_idx,
                        value=0.0,
                    ),
                    0.0,
                )  # Batch, InBeam, VocabWB
            
            if version == 2 and t == max_seq_len - 1:
                # Add LM EOS score at the end.
                eos_dim = Dim(model.target_dim.capacity, name="eos_dim")
                eos_target = rf.expand_dim(prev_target, dim=eos_dim)
                new_label = rf.expand_dims(rf.range_over_dim(eos_dim), dims=batch_dims + [beam_dim])
                eos_target = _update_context(
                    eos_target,
                    new_label,
                    context_dim
                )
                
                eos_logits, _ = lm(
                    eos_target,
                    spatial_dim=context_dim,
                    out_spatial_dim=lm_out_dim,
                    state=lm_state_,
                )  # Flat_Batch_Beam, Vocab / ...
                eos_logits = rf.gather(eos_logits, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
                assert eos_logits.dims == (beam_dim, *batch_dims, eos_dim, model.target_dim)
                lm_eos_score = rf.log_softmax(eos_logits, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
                lm_eos_score *= lm_scale
                lm_eos_score = rf.gather(lm_eos_score, indices=model.eos_idx, axis=model.target_dim)
                lm_eos_score = _target_dense_extend_blank(
                    lm_eos_score,
                    target_dim=eos_dim,
                    wb_target_dim=model.wb_target_dim,
                    blank_idx=model.blank_idx,
                    value=0.0,
                )
                
                seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB
        
        if version == 7:
            if print_idx:
                fixed_seq_prob += label_log_prob_ta[t].raw_tensor[print_idx[0], flashlight_align[t]]
                if flashlight_align[t] != 184:
                    lm_initial_state2 = lm.default_initial_state(batch_dims=batch_dims_)
                    lm_logits2, lm_state2 = lm(
                        rf.convert_to_tensor(fixed_lm_state, dims=[context_dim], sparse_dim=model.target_dim),
                        spatial_dim=context_dim,
                        out_spatial_dim=lm_out_dim,
                        state=lm_initial_state2,
                    )  # Vocab / ...
                    lm_logits2 = rf.gather(lm_logits2, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
                    lm_log_probs2 = rf.log_softmax(lm_logits2, axis=model.target_dim)  # Vocab
                    lm_log_probs2 *= lm_scale
                    
                    fixed_lm_state = fixed_lm_state[1:] + [flashlight_align[t]]
                    fixed_seq_prob += lm_log_probs2.raw_tensor[flashlight_align[t]]
                print(f"t={t}: {fixed_seq_prob}")
                if t == max_seq_len - 1:
                    lm_initial_state2 = lm.default_initial_state(batch_dims=batch_dims_)
                    lm_logits2, lm_state2 = lm(
                        rf.convert_to_tensor(fixed_lm_state, dims=[context_dim], sparse_dim=model.target_dim),
                        spatial_dim=context_dim,
                        out_spatial_dim=lm_out_dim,
                        state=lm_initial_state2,
                    )  # Vocab / ...
                    lm_logits2 = rf.gather(lm_logits2, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
                    lm_log_probs2 = rf.log_softmax(lm_logits2, axis=model.target_dim)  # Vocab
                    lm_log_probs2 *= lm_scale
                    
                    fixed_seq_prob += lm_log_probs2.raw_tensor[model.eos_idx]
                    print(f"FINAL SCORE: {fixed_seq_prob}")
        
        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        target_wb = rf.cast(target_wb, "int32")
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        if version != 2 or t < max_seq_len - 1:
            lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
            lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)
            prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
            prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB
            got_new_label = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
            target = rf.where(
                got_new_label,
                _update_context(
                    prev_target,
                    _target_remove_blank(
                        target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
                    ),
                    context_dim
                ),
                prev_target,
            )  # Batch, Beam -> Vocab

            got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
            if got_new_label_cpu.raw_tensor.sum().item() > 0:
                (target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = rf.nested.masked_select_nested(
                    (target, lm_state),
                    mask=got_new_label,
                    mask_cpu=got_new_label_cpu,
                    dims=batch_dims + [beam_dim],
                )
                # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
                assert packed_new_label_dim.get_dim_value() > 0
                
                lm_logits_, lm_state_ = lm(
                    target_,
                    spatial_dim=context_dim,
                    out_spatial_dim=lm_out_dim,
                    state=lm_state_,
                )  # Flat_Batch_Beam, Vocab / ...
                lm_logits_ = rf.gather(lm_logits_, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
                assert lm_logits_.dims == (packed_new_label_dim, model.target_dim)
                lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
                lm_log_probs_ *= lm_scale

                lm_log_probs, lm_state = rf.nested.masked_scatter_nested(
                    (lm_log_probs_, lm_state_),
                    (lm_log_probs, lm_state),
                    mask=got_new_label,
                    mask_cpu=got_new_label_cpu,
                    dims=batch_dims + [beam_dim],
                    in_dim=packed_new_label_dim,
                    masked_select_dim_map=packed_new_label_dim_map,
                )  # Batch, Beam, Vocab / ...
                
    if version != 2 and lm_scale > 0.0:
        # seq_log_prob, lm_log_probs: Batch, Beam
        # Add LM EOS score at the end.
        lm_eos_score = rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)
        seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB
        

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)
    
    if int(beam_dim.get_dim_value()) >= n_best:
        # if n_best > 1:
        #     for batch_idx in range(seq_targets_wb.raw_tensor.size(0)):
        #         hyps_per_batch = seq_targets_wb.raw_tensor[batch_idx].tolist()
        #         scores_per_batch = seq_log_prob.raw_tensor[batch_idx].tolist()
                
        #         # We have to select the n_best on output level
        #         hyps_shortened = [_output_hyps(hyp) for hyp in hyps_per_batch]
        #         nbest_hyps = []
        #         nbest_hyps_ids = []
        #         k = 0
        #         i = 0
        #         while k < n_best:
        #             if i >= len(hyps_shortened):
        #                 break
        #             if hyps_shortened[i] not in nbest_hyps:
        #                 nbest_hyps.append(hyps_shortened[i])
        #                 nbest_hyps_ids.append(i)
        #                 k += 1
        #             i += 1
        #         hyps_per_batch = [hyps_per_batch[id] for id in nbest_hyps_ids]
        #         scores_per_batch = [scores_per_batch[id] for id in nbest_hyps_ids]
                
        #         if len(hyps_per_batch) < n_best:
        #             print("Not enough n-best")
        #             hyps_per_batch += [[]] * (n_best - len(hyps_per_batch))
        #             scores_per_batch += [-1e30] * (n_best - len(scores_per_batch))
        # else:
        seq_log_prob, indices, beam_dim_new = rf.top_k(
            seq_log_prob, k_dim=Dim(n_best, name=f"nbest-beam"), axis=beam_dim
        )
        seq_targets_wb = rf.gather(seq_targets_wb, axis=beam_dim, indices=indices)
        beam_dim = beam_dim_new

    if train_lm:
        return seq_targets_wb.raw_tensor.transpose(0,1).transpose(1,2).tolist()
    else:
        return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim