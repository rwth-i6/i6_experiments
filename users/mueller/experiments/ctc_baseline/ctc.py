"""
CTC experiments.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, Callable, Dict, Any
import numpy as np

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim

from sisyphus import tk

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, TrainDef, RecogDef
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.mueller.train import ExtendedTrainDef
from i6_experiments.users.mueller.experiments.language_models.n_gram import get_count_based_n_gram, get_prior_from_unigram
from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm, get_ffnn_lm
from i6_experiments.users.mann.nn.util import DelayedCodeWrapper
from i6_experiments.users.mueller.utils import calc_stats
from i6_experiments.users.mueller.experiments.ctc_baseline.configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor, config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4, dict_update_deep, post_config

from i6_experiments.users.mueller.experiments.ctc_baseline.model import Model, OUT_BLANK_LABEL, _log_mel_feature_dim
from i6_experiments.users.mueller.experiments.ctc_baseline.decoding import recog_flashlight_ngram, recog_no_lm, recog_flashlight_ffnn, recog_ffnn
from i6_experiments.users.mueller.experiments.ctc_baseline.training import ctc_train, full_sum_train, ce_train

if TYPE_CHECKING:
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.mueller.datasets.task import Task
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput


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
decode_nbest_epochs = 0
decode_nbest_epochs_init = 0
decode_all_fixed_epochs = True
decode_all_fixed_epochs_init = False
exclude_epochs = True
cache_manager = True

def py():
    """Sisyphus entry point"""
    # relPosAttDef: Use the default RelPosSelfAttention instead of the Shawn et al 2018 style, old RETURNN way.
    enc_conformer_layer_default = rf.build_dict(
        rf.encoder.conformer.ConformerEncoderLayer,
        ff_activation=rf.build_dict(rf.relu_square),
        num_heads=8,
    )

    # Config
    vocab = "bpe128"                            # "spm20k", "char", "bpe10k"
    decoding_imp = "flashlight"                 # "flashlight", "albert-flashlight", "albert-lm", "albert-greedy", "marten-greedy""
    epochs = 500                                # Training epochs
    self_training_rounds = 0                    # Self-supevised training rounds
    reset_steps = False                          # Whether to reset steps after the first self-training round
    init_small = True                           # 100h supervised initialization
    pseudo_label_small = True                   # 860h pseudo-labels
    keep_small_labels = False                   # Keep true labels of 100h data during self-training
    pseudo_nbest = 1                            # Number of pseudo-labels
    with_prior = True
    empirical_prior = True
    prior_from_max = False
    aux_loss = True
    alt_decoder = False
    calc_last_pseudo_labels = False
    tune_hyperparameters = False
    from_scratch = True
    decode_every_step = False
    accum_grad_multiple_step = 1
    # decoder_lm_config = {}
    # decoder_lm_config = {"class": "FeedForwardLm", "context_size": 8}
    decoder_lm_config = {"class": "ngram", "order": 2}
    use_norm_st_loss = True
    # only relevant in alberts LM decoding
    use_recombination = True
    recombine_blank = True
    recombine_after_topk = True
    
    use_sum_criterion = False
    horizontal_prior = False
    blank_prior = False
    prior_gradient = False
    empirical_prior_full_sum = True
    prior_from_max_full_sum = False
    # train_lm_config = {"class": "FeedForwardLm", "context_size": 3}
    train_lm_config = {"class": "ngram", "order": 2}
    top_k = 0
    version = 1
    print_gradients = True
    alignment_topk = False
    blank_correction_version = 0
    correction_in_final_score = False
    am_lm_prior = [
        (1.0, 0.3, 0.12)
    ]
    
    use_sgd = False
    adamw_betas = None # (0.5, 0.98) # None
    self_train_subset = None # 18000
    # TODO gradient_cli_global_norm
    
    assert not decode_every_step or (decode_every_step and decoder_lm_config["class"] == "FeedForwardLm" and empirical_prior)
    assert pseudo_nbest == 1 or (decoder_lm_config["class"] == "FeedForwardLm" and empirical_prior)
    assert (empirical_prior_full_sum and empirical_prior) or not empirical_prior_full_sum
    model_config = {"enc_conformer_layer": enc_conformer_layer_default, "feature_batch_norm": True}
    
    if init_small:
        epochs = 50
    if self_training_rounds > 0:
        if pseudo_label_small:
            epoch_dict = {1: 450, 2: 225, 4: 113, 6: 75, 8: 56, 10: 45, 25: 18, 50: 9}
        else:
            epoch_dict = {1: 500, 2: 250, 4: 125, 6: 83, 8: 63, 10: 50, 25: 20, 50: 10}
        self_epochs = epoch_dict[self_training_rounds]
        if self_train_subset:
            self_epochs = 56
    
    decoder_hyperparameters = {}
    if decoding_imp == "marten-greedy":
        decoder_hyperparameters = {
            "greedy": True
        }
        decoding_str = "-recog_greedy"
        if with_prior:
            decoder_hyperparameters["prior_weight"] = 0.2
            decoding_str += f"_p{str(decoder_hyperparameters['prior_weight']).replace('.', '')}" + ("-emp" if empirical_prior else "")
    elif decoding_imp == "albert-greedy":
        decoding_str = "-recog_albert"
    elif decoding_imp.endswith("flashlight") or decoding_imp == "albert-lm":
        decoder_hyperparameters = {
            "log_add": False,
            "nbest": 1,
            "beam_size": 80,
            "lm_weight": 0.02, # NOTE: weights are exponentials of the probs
            "use_logsoftmax": True,
            "use_lm": True,
            "use_lexicon": True,
            # "version": 1,
        }
        if with_prior:
            decoder_hyperparameters["prior_weight"] = 0.0 # 0.2 if not using emprirical prior
        if decoder_lm_config:
            decoder_hyperparameters["lm_order"] = decoder_lm_config["order"] if decoder_lm_config["class"] == "ngram" else f"ffnn{decoder_lm_config['context_size']}"
            decoder_hyperparameters["use_lexicon"] = False
            if decoder_lm_config["class"] == "FeedForwardLm":
                model_config["recog_language_model"] = decoder_lm_config
                if decode_every_step or pseudo_nbest > 1:
                    model_config["train_language_model"] = decoder_lm_config
                if decoding_imp == "albert-lm" and use_recombination:
                    decoder_hyperparameters["use_recombination"] = True
                    if recombine_blank:
                        decoder_hyperparameters["recomb_blank"] = True
                    if recombine_after_topk:
                        decoder_hyperparameters["recomb_after_topk"] = True
            
        p0 = f"_p{str(decoder_hyperparameters['prior_weight']).replace('.', '')}" + ("-emp" if empirical_prior else ("-from_max" if prior_from_max else "")) if with_prior else ""
        p1 = "sum" if decoder_hyperparameters['log_add'] else "max"
        p2 = f"n{pseudo_nbest}"
        p3 = f"b{decoder_hyperparameters['beam_size']}"
        p4 = f"w{str(decoder_hyperparameters['lm_weight']).replace('.', '')}" + ((f"o{decoder_lm_config['order']}" if decoder_lm_config["class"] == "ngram" else f"ffnn{decoder_lm_config['context_size']}") if decoder_lm_config else "")
        p6 = "_noLM" if not decoder_hyperparameters['use_lm'] else ""
        p7 = "_noLEX" if not decoder_hyperparameters['use_lexicon'] else ""
        decoding_str = f"{p0}_{p1}_{p2}_{p3}_{p4}{p6}{p7}"
        
        if decoding_imp == "albert-flashlight":
            decoding_str = "-recog_albert_lm" + decoding_str
        elif decoding_imp == "albert-lm":
            decoding_str = "-recog_v_lm" + ("_r" + ("-b" if recombine_blank else "") + ("-a" if recombine_after_topk else "") if use_recombination else "") + decoding_str
        else:
            decoding_str = "-recog_lm" + decoding_str
        
        if alt_decoder:
            alt_decoder_hyperparameters = decoder_hyperparameters.copy()
            alt_decoder_hyperparameters["lm_weight"] = 0.7
            alt_decoder_hyperparameters["beam_size"] = 10
            if with_prior:
                alt_decoder_hyperparameters["prior_weight"] = 0.3
                
            if decode_every_step:
                every_step_hyperparameters = decoder_hyperparameters.copy()
                every_step_hyperparameters["lm_weight"] = 0.4
                # every_step_hyperparameters["decay"] = 0.9995
                # every_step_hyperparameters["decay_limit"] = 0.25
                every_step_hyperparameters["beam_size"] = 10
                if with_prior:
                    every_step_hyperparameters["prior_weight"] = 0.3
                a0 = f"p{str(every_step_hyperparameters['prior_weight']).replace('.', '')}" if with_prior else ""
                a1 = f"b{every_step_hyperparameters['beam_size']}"
                a2 = f"w{str(every_step_hyperparameters['lm_weight']).replace('.', '')}"
                a3 = (f"dec{str(every_step_hyperparameters['decay']).replace('.', '')}" if 'decay' in every_step_hyperparameters else "") + (f"-lim{str(every_step_hyperparameters['decay_limit']).replace('.', '')}" if 'decay_limit' in every_step_hyperparameters else "")
                every_step_str = f"_{a0}_{a1}_{a2}_{a3}"
                
            if use_sum_criterion or decode_every_step:
                alt_decoder_hyperparameters["lm_weight"] = 0.0
                alt_decoder_hyperparameters["prior_weight"] = 0.0
                alt_decoder_hyperparameters["use_lm"] = False
                alt_decoder_hyperparameters["use_lexicon"] = False
                str_add = "_no-lexicon"
            else:
                str_add = ""
                
            a0 = f"_p{str(alt_decoder_hyperparameters['prior_weight']).replace('.', '')}" + ("-emp" if empirical_prior else ("-from_max" if prior_from_max else "")) if with_prior else ""
            a1 = f"b{alt_decoder_hyperparameters['beam_size']}"
            a2 = f"w{str(alt_decoder_hyperparameters['lm_weight']).replace('.', '')}"
            a3 = (f"-accum{accum_grad_multiple_step}" if accum_grad_multiple_step > 1 else "") + ("_every-step" + every_step_str if decode_every_step else "") + ("_tune" if tune_hyperparameters else "")
            decoding_str += f"_ALT{a3}{a0}_{a1}_{a2}{str_add}"
    else:
        raise ValueError(f"Unknown decoder selection: {decoding_imp}")
    
    config_updates = {
        **_get_cfg_lrlin_oclr_by_bs_nep(15_000, epochs),
        "optimizer.weight_decay": 1e-2,
        "__train_audio_preprocess": speed_pert_librosa_config,
        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        "max_seq_length_default_target": None,
        "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    }
    config_updates_self_training = {
        **_get_cfg_lrlin_oclr_by_bs_nep(15_000, self_epochs),
        "optimizer.weight_decay": 1e-2,
        "__train_audio_preprocess": speed_pert_librosa_config,
        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        "max_seq_length_default_target": None,
        "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    } if self_training_rounds > 0 else None
    
    if config_updates_self_training:
        if adamw_betas:
            config_updates_self_training["optimizer.betas"] = adamw_betas
        if not reset_steps:
            if pseudo_label_small:
                config_updates_self_training["learning_rate_piecewise_steps"] = [253_000, 506_000, 562_000]
            else:
                config_updates_self_training["learning_rate_piecewise_steps"] = [279_000, 558_000, 620_000]
        if decode_every_step:
            config_updates_self_training["decode_every_step"] = decode_every_step
            assert every_step_hyperparameters
            config_updates_self_training["hyperparameters_decoder"] = every_step_hyperparameters
        if pseudo_nbest > 1:
            config_updates_self_training["ps_nbest"] = pseudo_nbest
            config_updates_self_training["hyperparameters_decoder"] = decoder_hyperparameters.copy()
        if accum_grad_multiple_step > 1:
            config_updates_self_training["accum_grad_multiple_step"] = accum_grad_multiple_step
        if not use_norm_st_loss:
            config_updates_self_training["use_normalized_loss"] = use_norm_st_loss

    for am, lm, prior in am_lm_prior:
        if use_sum_criterion:
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
            
            if not horizontal_prior:
                config_full_sum["horizontal_prior"] = horizontal_prior
            if not blank_prior:
                config_full_sum["blank_prior"] = blank_prior
            if not prior_gradient:
                config_full_sum["prior_gradient"] = prior_gradient
            if top_k > 0:
                config_full_sum["top_k"] = top_k
            if empirical_prior_full_sum:
                config_full_sum["empirical_prior"] = True
            if prior_from_max_full_sum:
                config_full_sum["max_prior"] = True
            if not alignment_topk:
                config_full_sum["alignment_topk"] = False
            if blank_correction_version > 0:
                config_full_sum["blank_correction_version"] = blank_correction_version
            if correction_in_final_score:
                config_full_sum["correction_in_final_score"] = True
            if print_gradients:
                config_full_sum["print_gradients"] = True
            
            # This is to change the hash when we made chnages in the loss function
            config_full_sum["version"] = version
            
            sum_str = f"-full_sum" + \
                (f"_p{str(config_full_sum['prior_scale']).replace('.', '')}_l{str(config_full_sum['lm_scale']).replace('.', '')}_a{str(config_full_sum['am_scale']).replace('.', '')}" if scales_not_std else "") + \
                (f"_LMorder{train_lm_config['order']}" if train_lm_config["class"] == "ngram" and train_lm_config["order"] > 2 else (f"_ffnn{train_lm_config['context_size']}" if train_lm_config["class"] == "FeedForwardLm" else "")) + \
                (f"_topK{top_k}" + ("_align" if alignment_topk else "") + (f"_bc{blank_correction_version}" + ("sc" if correction_in_final_score else "") if blank_correction_version > 0 else "") if top_k > 0 else "") + \
                ("_emp" if empirical_prior_full_sum else "") + \
                ("_max_pr" if not empirical_prior_full_sum and prior_from_max_full_sum else "") + \
                ("_wo_h_pr" if not horizontal_prior else "") + \
                ("_wo_b_pr" if not blank_prior else "") + \
                ("_wo_pr_grad" if not prior_gradient else "")
                
            if train_lm_config:
                model_config["train_language_model"] = train_lm_config
        
        alias_name = f"ctc" + \
            (sum_str if use_sum_criterion else "") + \
            (f"-st_{self_training_rounds}" + (f"_LRedge-e4" if False else "") + ("_no_norm" if not use_norm_st_loss else "") + ("_keep_LR" if not reset_steps else "") + ("_SGD" if use_sgd else (f"_b1-{str(adamw_betas[0]).replace('.', '')}_b2-{str(adamw_betas[1]).replace('.', '')}" if adamw_betas else "")) + ("_from_scratch" if from_scratch else "") + (f"_s{self_train_subset}" if self_train_subset is not None else "") + (f"_e{self_epochs}" if self_epochs != 450 else "") if self_training_rounds > 0 else "") + \
            (f"-wo_aux_loss" if not aux_loss else "") + \
            (f"-ds100h" if init_small else "") + \
            (f"-pl960h" + ("_keep100h" if keep_small_labels else "") if not pseudo_label_small else "") + \
            f"-{vocab}" + \
            f"{decoding_str}"
            
        if decoding_imp in ["flashlight", "marten-greedy"]:
            decoder_def = model_recog_lm
        elif decoding_imp == "albert-greedy":
            decoder_def = model_recog
        elif decoding_imp == "albert-flashlight":
            decoder_def = model_recog_flashlight
        elif decoding_imp == "albert-lm":
            decoder_def = model_recog_lm_albert
        else:
            raise ValueError(f"Unknown decoder selection: {decoding_imp}")

        train_exp(
            name = alias_name,
            config = config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
            decoder_def = decoder_def,
            decoder_hyperparameters = decoder_hyperparameters,
            hyperparamters_self_training = alt_decoder_hyperparameters if alt_decoder else None,
            pseudo_nbest=pseudo_nbest,
            model_config = model_config,
            config_updates = config_updates,
            config_updates_self_training = config_updates_self_training,
            config_full_sum=config_full_sum if use_sum_criterion else None,
            vocab = vocab,
            self_training_rounds = self_training_rounds,
            init_small = init_small,
            pseudo_label_small = pseudo_label_small,
            keep_small_labels = keep_small_labels,
            with_prior = with_prior,
            empirical_prior=empirical_prior,
            prior_from_max=prior_from_max,
            use_sum_criterion=use_sum_criterion,
            aux_loss=aux_loss,
            self_train_subset=self_train_subset,
            calc_last_pseudo_labels=calc_last_pseudo_labels,
            tune_hyperparameters=tune_hyperparameters,
            from_scratch=from_scratch,
            use_sgd=use_sgd,
            reset_steps=reset_steps,
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
    from i6_experiments.users.mueller.train import train
    from i6_experiments.users.mueller.recog import recog_training_exp, GetBestTuneValue
    from i6_experiments.users.mueller.datasets.librispeech import get_librispeech_task_raw_v2, TrainDatasetSel

    print("Job Name:", name)
    if not enabled:
        return None

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    
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
        if "train_language_model" in mc_self and mc_self["train_language_model"]["class"] == "ngram":
            mc_self.pop("train_language_model", None)
        model_def = ModelDefWithCfg(model_def, mc)
        model_def_self = ModelDefWithCfg(model_def_self, mc_self)
    if not train_def:
        train_def = ctc_training
        
    # Calculate some DataSet Stats
    # calc_stats(task.train_dataset.vocab)
        
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
        if config_updates_self_training and (config_updates_self_training.get("decode_every_step", False) or config_updates_self_training.get("ps_nbest", 1) > 1):
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
        check_train_scores_nbest=decode_nbest_epochs_init,
        exclude_epochs=sorted(list(model_with_checkpoint[0].fixed_epochs))[:-1] if not decode_all_fixed_epochs_init else (),
    )
    
    # Do self training on pseudo labels
    for i in range(self_training_rounds):
        assert pseudo_label_path_dict is not None, "Pseudo label path is not set"
        prefix_self_training = prefix + f"/self-training-{i+1}"
        task, _, _ = get_librispeech_task_raw_v2(
            vocab=vocab,
            train_vocab_opts=train_vocab_opts,
            ds_sel = TrainDatasetSel.train_860h if pseudo_label_small else TrainDatasetSel.train_960h,
            init_small=init_small,
            with_prior=with_prior,
            empirical_prior=empirical_prior,
            pseudo_label_path = pseudo_label_path_dict,
            keep_small_labels = keep_small_labels,
            train_subset = self_train_subset,
            eval_subset = 300 if self_train_subset else 3000,
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
            
        if config_self.get("empirical_prior", False) or config_self.get("decode_every_step", False) or config_self.get("ps_nbest", 1) > 1:
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
        # else:
        #     peak_lr = 1e-4
        #     config_self["learning_rate_piecewise_values"] = [peak_lr, peak_lr, peak_lr * 0.27, peak_lr * 0.1]
        if not aux_loss:
            config_self.pop("aux_loss_layers")

        if not reset_steps:
            peak_lr = 5e-4
            config_self["learning_rate_piecewise_values"] = [peak_lr * 2e-2, peak_lr, peak_lr * 2e-2, peak_lr * 2e-3]
            if config_self["learning_rate_piecewise_steps"][0] > 20_000:
                config_self["learning_rate_piecewise_steps"] = [20_000] + config_self["learning_rate_piecewise_steps"][1:]
            # add something to hash so first training is different and correct epochs are saved
            if self_training_rounds != 4:
                config_self["_st_rounds"] = self_training_rounds
        # Use different LR if second iteration, NOTE: this is very specific to 860h training
        elif i > 0:
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
            check_train_scores_nbest=decode_nbest_epochs,
            exclude_epochs=sorted(list(model_with_checkpoint[i + 1].fixed_epochs))[:-1] if not decode_all_fixed_epochs else (),
        )

    _train_experiments[name] = model_with_checkpoint[-1]
    return model_with_checkpoint[-1]


def _remove_eos_label_v2(res: RecogOutput) -> RecogOutput:
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
    from i6_core.returnn.search import SearchRemoveLabelJob

    return RecogOutput(SearchRemoveLabelJob(res.output, remove_label="</s>", output_gzip=True).out_search_results)


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


#---------------------------------------------------------------------------------------------------------------------------------------
# MODEL DEFINITION

def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    
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


#---------------------------------------------------------------------------------------------------------------------------------------
# TRAINING DEFINITIONS

def ctc_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
    return ctc_train(model=model, data=data, data_spatial_dim=data_spatial_dim, targets=targets, targets_spatial_dim=targets_spatial_dim)

ctc_training: TrainDef[Model]
ctc_training.learning_rate_control_error_measure = "ctc"

def ctc_sum_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, lm_path: tk.Path, seq_tags: rf.Tensor = None, targets: rf.Tensor, targets_spatial_dim: Dim):
    return full_sum_train(model=model, data=data, data_spatial_dim=data_spatial_dim, lm_path=lm_path, seq_tags=seq_tags, targets=targets, targets_spatial_dim=targets_spatial_dim)

ctc_sum_training: ExtendedTrainDef[Model]
ctc_sum_training.learning_rate_control_error_measure = "full_sum"

def ce_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
    return ce_train(model=model, data=data, data_spatial_dim=data_spatial_dim, targets=targets, targets_spatial_dim=targets_spatial_dim)

ce_training: TrainDef[Model]
ce_training.learning_rate_control_error_measure = "ce"


#---------------------------------------------------------------------------------------------------------------------------------------
# RECOG DEFINITIONS

def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    return recog_no_lm(model=model, data=data, data_spatial_dim=data_spatial_dim)

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
    return recog_flashlight_ngram(model=model, data=data, data_spatial_dim=data_spatial_dim, arpa_4gram_lm=arpa_4gram_lm, lexicon=lexicon, hyperparameters=hyperparameters, prior_file=prior_file)

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
    
    return recog_flashlight_ffnn(model=model, label_log_prob=label_log_prob, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, prior_file=prior_file, print_idx=print_idx)

# RecogDef API
model_recog_flashlight: RecogDef[Model]
model_recog_flashlight.output_with_beam = True
model_recog_flashlight.output_blank_label = OUT_BLANK_LABEL
model_recog_flashlight.batch_size_dependent = True  # our models currently just are batch-size-dependent...

    
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
    if version == 9:
        for seq in ["dev-other/1630-96099-0024/1630-96099-0024"]:
            if seq in seq_tags:
                idx = np.where(seq_tags == seq)[0]
                print_idx.append(idx)
    
    return recog_ffnn(model=model, label_log_prob=label_log_prob, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, batch_dims=batch_dims, prior_file=prior_file, version=version, print_idx=print_idx)

# RecogDef API
model_recog_lm_albert: RecogDef[Model]
model_recog_lm_albert.output_with_beam = True
model_recog_lm_albert.output_blank_label = OUT_BLANK_LABEL
model_recog_lm_albert.batch_size_dependent = True  # our models currently just are batch-size-dependent...