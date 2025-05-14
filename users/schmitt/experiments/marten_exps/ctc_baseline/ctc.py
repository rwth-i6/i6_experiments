"""
CTC experiments.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, Callable, Dict, Any
import numpy as np
import transformers

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim

from sisyphus import tk

from i6_experiments.users.mueller.train import SumTrainDef, CETrainDef
from i6_experiments.users.mueller.utils import calc_stats
from i6_experiments.users.mueller.experiments.language_models.n_gram import get_count_based_n_gram, get_prior_from_unigram
from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm, get_ffnn_lm
from i6_experiments.users.mueller.experiments.ctc_baseline.configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor, config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4, dict_update_deep, post_config
from i6_experiments.users.mueller.experiments.ctc_baseline.model import Model, OUT_BLANK_LABEL, _log_mel_feature_dim, Wav2VecModel
from i6_experiments.users.mueller.experiments.ctc_baseline.decoding import recog_flashlight_ngram, recog_no_lm, recog_flashlight_ffnn, recog_ffnn, recog_gradients
from i6_experiments.users.mueller.experiments.ctc_baseline.training import ctc_train, full_sum_train, ce_train, seq_gamma_ctc_train

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, TrainDef, RecogDef
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint
from i6_experiments.users.mann.nn.util import DelayedCodeWrapper

if TYPE_CHECKING:
    from i6_experiments.common.setups import serialization
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
decode_nbest_epochs_init = 2
decode_all_fixed_epochs = True
decode_all_fixed_epochs_init = True
exclude_epochs = True
cache_manager = True
tune_version = 3

def py():
    """Sisyphus entry point"""

    # General config
    vocab = "bpe128"                            # Vocab, e.g. "bpe128", "spm20k", "char", "bpe10k"
    self_training_rounds = 0                    # Self-supervised training rounds
    reset_steps = False                         # Whether to reset step count after the first self-training round (affects LR schedule)
    from_scratch = False                        # Self-training starts from scratch
    pseudo_label_small = False                  # 860h pseudo-labels if True, 960h pseudo-labels if False
    keep_small_labels = True                    # Keep true labels of 100h data during self-training
    pseudo_nbest = 1                            # Number of pseudo-label sequences
    norm_nbest_rescore = False                  # Normalize the LM and Prior values for each pseudo label sequence before adding them up
    grad_nbest = 10                             # Number of gradients we want to keep during gradient dumping
    rescore_ctc_loss_for_grad = False           # Rescore with CTC loss for the gradients we want to dump
    calc_last_pseudo_labels = False             # Calculate the pseudo labels after the last iteration of self-training
    decode_every_step = False                   # Decode every step during self-training
    keep_best_decoding = False                   # Keep the decoding of previous self-training rounds if it is better
    accum_grad_multiple_step = 1                # Accumulate gradients over multiple steps
    aux_loss = True                             # Whether to use the auxiliary loss
    use_norm_st_loss = True                     # Use normalized loss during self-training
    use_seq_gamma_loss = False                  # Use sequence gamma CE loss instead of summed CTC loss
    gamma_scaling = 1.0                         # Scaling for the sequence gammas
    use_ce_loss = False                         # Use CE loss instead of CTC loss
    speed_pert = True                           # Whether to use speed perturbation
    train_lm_config = {}                        # LM selection for decoding during training
    # train_lm_config = {"class": "FeedForwardLm", "context_size": 8} # LM selection for decoding during training
    # train_lm_config = {"class": "ngram", "order": 3} # LM selection for decoding during training
    train_version = 1                           # Version for training added to change the hash
    num_gpus = 4                                # Number of GPUs to use during training
    if self_training_rounds == 0:
        from_scratch = True
        pseudo_label_small = True
        keep_small_labels = False
        pseudo_nbest = 1
        norm_nbest_rescore = False
    
    # Decoder config (more further down)
    decoding_imp = "albert-lm"                  # Decoding implementation, e.g. "flashlight", "albert-flashlight", "albert-lm", "albert-greedy", "marten-greedy", "gradients"
    with_prior = True                           # Whether to use a prior during decoding
    label_prior = False                          # Use the label prior instead of frame prior
    empirical_prior = True                      # Whether to use an empirical prior instead of a model prior
    prior_from_max = False                      # Whether to calculate the model prior by max instead of softmax (not fully supported)
    alt_decoder = True                          # Whether to use different decoder hyperparameters for self-training
    tune_hyperparameters = True                # Tune decoder hyperparameters in between self-training rounds
    # decoder_lm_config = {}                    # LM selection for decoding, empty for word-level 4-gram
    decoder_lm_config = {"class": "FeedForwardLm", "context_size": 8} # LM selection for decoding, empty for word-level 4-gram
    # decoder_lm_config = {"class": "ngram", "order": 2} # LM selection for decoding, empty for word-level 4-gram
    use_recombination = True                    # Use recombination during decoding (only albert-lm)
    recombine_blank = True                      # Recombine sequences ending on blank with last seen same label (only albert-lm)
    recombine_after_topk = True                 # Recombine after top-k extraction instead of before (only albert-lm)
    recombine_with_sum = False                  # Use sum during recombination
    if self_training_rounds == 0:
        alt_decoder = False
    assert decoding_imp in ["flashlight", "albert-flashlight", "albert-lm", "albert-greedy", "marten-greedy", "gradients"]
    
    # Configs for init training
    init = "100h-supervised"                    # Which initialization to use, "100h-supervised", "960h-supervised", "100h-unsupervised"
    random_init = False                         # Start from random init during unsupervised init training, alteernatively use empirical prior
    decode_every_step_init = True               # Decode every step during unsupervised init training
    accum_grad_multiple_step_init = 80           # Accumulate gradients over multiple steps during unsupervised init training
    if not init.endswith("unsupervised"):
        decode_every_step_init = False
    
    # Configs for full-sum training
    use_sum_criterion = False                   # Use full-sum criterion
    horizontal_prior = False                    # Use prior for transitions with label repetitions which get collapsed to one label
    blank_prior = False                         # Use prior for blank transitions
    prior_gradient = False                      # If the prior is calculated for each batch, we can add this to the gradient
    empirical_prior_full_sum = True             # Use empirical prior for full-sum criterion
    prior_from_max_full_sum = False             # Use max instead of softmax for prior calculation used during full-sum training
    top_k = 10                                   # Whether to use top-k approximation instead of full-sum, 0 for full-sum
    alignment_topk = False                      # Apply top-k on alignment level instead of output level (the scores are recombined)
    print_gradients = True                      # Print gradients for a few sequences
    blank_correction_version = 0                # Which version of blank correction we want to use, 0 for no correction
    correction_in_final_score = False           # Apply the correction in the final score as well instad of just using the corrections duing top-k
    am_lm_prior = (1.0, 0.2, 0.05)               # Weights for the AM, LM and Prior in the full-sum criterion
    
    # Configs for the optimizer
    use_sgd = False                             # Use SGD instead of AdamW
    adamw_betas = None # (0.5, 0.98) # None     # AdamW betas
    self_train_subset = None # 18000            # Train on a subset of the data
    # TODO gradient_clip_global_norm
    
    assert not label_prior or empirical_prior, "Label prior is not supported with model prior, yet"
    assert gamma_scaling == 1.0 or use_seq_gamma_loss
    assert not (use_ce_loss and use_seq_gamma_loss)
    assert not (not label_prior and norm_nbest_rescore)
    assert not decode_every_step or keep_best_decoding
    assert (self_training_rounds > 0) == alt_decoder
    assert not use_ce_loss or not speed_pert
    assert not use_ce_loss or not keep_small_labels
    assert not decoding_imp == "gradients" or use_ce_loss
    assert not train_lm_config or ((train_lm_config["class"] == "FeedForwardLm" and top_k > 0) or train_lm_config["class"] == "ngram")
    assert not decode_every_step or (decode_every_step and decoder_lm_config["class"] == "FeedForwardLm" and empirical_prior)
    assert pseudo_nbest == 1 or (decoder_lm_config["class"] == "FeedForwardLm" and empirical_prior)
    assert (empirical_prior_full_sum and empirical_prior) or not empirical_prior_full_sum

    # model opts
    # relPosAttDef: Use the default RelPosSelfAttention instead of the Shawn et al 2018 style, old RETURNN way.
    enc_conformer_layer_default = rf.build_dict(
        rf.encoder.conformer.ConformerEncoderLayer,
        ff_activation=rf.build_dict(rf.relu_square),
        num_heads=8,
    )
    model_config = {"enc_conformer_layer": enc_conformer_layer_default, "feature_batch_norm": True}

    use_w2v_model = False
    
    # Read out correct number of epochs dependent on self-training iterations
    if init.startswith("100h"):
        epochs = 50
    else:
        epochs = 500
    if self_training_rounds > 0:
        if self_train_subset:
            self_epochs = 56
            # self_epochs = 2
        else:
            if pseudo_label_small:
                epoch_dict = {1: 450, 2: 225, 4: 113, 6: 75, 8: 56, 10: 45, 25: 18, 50: 9}
            else:
                epoch_dict = {1: 500, 2: 250, 4: 125, 6: 83, 8: 63, 10: 50, 25: 20, 50: 10}
            self_epochs = epoch_dict[self_training_rounds]
    
    # Create decoder hyperparameters
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
    elif decoding_imp.endswith("flashlight") or decoding_imp == "albert-lm" or decoding_imp == "gradients":
        decoder_hyperparameters = {
            "log_add": False,
            "nbest": 1,
            "beam_size": 10,
            "lm_weight": 0.8,
            "use_logsoftmax": True,
            "use_lm": True,
            "use_lexicon": True,
        }
        if with_prior:
            decoder_hyperparameters["prior_weight"] = 0.3 # 0.2 if not using emprirical prior
        if decoder_lm_config:
            decoder_hyperparameters["lm_order"] = decoder_lm_config["order"] if decoder_lm_config["class"] == "ngram" else f"ffnn{decoder_lm_config['context_size']}"
            decoder_hyperparameters["use_lexicon"] = False
            if decoder_lm_config["class"] == "FeedForwardLm":
                model_config["recog_language_model"] = decoder_lm_config
                if decoding_imp in ["albert-lm", "gradients"] and use_recombination:
                    decoder_hyperparameters["use_recombination"] = True
                    if recombine_blank:
                        decoder_hyperparameters["recomb_blank"] = True
                    if recombine_after_topk:
                        decoder_hyperparameters["recomb_after_topk"] = True
                    if recombine_with_sum:
                        decoder_hyperparameters["recomb_with_sum"] = True
        if decode_every_step or pseudo_nbest > 1 or decode_every_step_init or use_sum_criterion:
            assert train_lm_config
            model_config["train_language_model"] = train_lm_config
        else:
            assert not train_lm_config or decoding_imp == "gradients"
        if decoding_imp == "gradients":
            assert train_lm_config
            decoder_hyperparameters_grad = {}
            decoder_hyperparameters_grad["grad_nbest"] = grad_nbest
            decoder_hyperparameters_grad["lm_order"] = train_lm_config["order"] if train_lm_config["class"] == "ngram" else f"ffnn{train_lm_config['context_size']}"
            decoder_hyperparameters_grad["train_language_model"] = train_lm_config
            decoder_hyperparameters_grad["beam_size"] = 0
            if rescore_ctc_loss_for_grad:
                decoder_hyperparameters_grad["rescore_ctc_loss"] = rescore_ctc_loss_for_grad
            
        p0 = f"_p{str(decoder_hyperparameters['prior_weight']).replace('.', '')}" + ("-emp" if empirical_prior else ("-from_max" if prior_from_max else "")) if with_prior else ""
        p1 = "sum" if decoder_hyperparameters['log_add'] else "max"
        p2 = f"n{pseudo_nbest}" + ("-nm" if norm_nbest_rescore else "") + (f"-s{str(gamma_scaling).replace('.', '')}" if gamma_scaling != 1.0 else "")
        p3 = f"b{decoder_hyperparameters['beam_size']}"
        p4 = f"w{str(decoder_hyperparameters['lm_weight']).replace('.', '')}" + ((f"o{decoder_lm_config['order']}" if decoder_lm_config["class"] == "ngram" else f"ffnn{decoder_lm_config['context_size']}") if decoder_lm_config else "") + (((f"_To{train_lm_config['order']}" if train_lm_config["class"] == "ngram" else f"_Tf{train_lm_config['context_size']}") + (f"b{decoder_hyperparameters_grad['beam_size']}" if "beam_size" in decoder_hyperparameters_grad else "")) if train_lm_config and train_lm_config != decoder_lm_config else "")
        p6 = "_noLM" if not decoder_hyperparameters['use_lm'] else ""
        p7 = "_noLEX" if not decoder_hyperparameters['use_lexicon'] else ""
        decoding_str = f"{p0}_{p1}_{p2}_{p3}_{p4}{p6}{p7}"
        
        if decoding_imp == "albert-flashlight":
            decoding_str = "-recog_albert_lm" + decoding_str
        elif decoding_imp == "albert-lm":
            decoding_str = "-recog_v_lm" + ("_r" + ("-b" if recombine_blank else "") + ("-a" if recombine_after_topk else "") + ("-s" if recombine_with_sum else "") if use_recombination else "") + decoding_str
        elif decoding_imp == "gradients":
            decoding_str = "-recog_grad" + ("_r" + ("-b" if recombine_blank else "") + ("-a" if recombine_after_topk else "") + ("-s" if recombine_with_sum else "") if use_recombination else "") + (f"_n{grad_nbest}" if grad_nbest != 0 else "") + ("_ctcL" if rescore_ctc_loss_for_grad else "") + decoding_str
        else:
            decoding_str = "-recog_lm" + decoding_str
        
        if alt_decoder:
            alt_decoder_hyperparameters = decoder_hyperparameters.copy()
            alt_decoder_hyperparameters["lm_weight"] = 0.7
            alt_decoder_hyperparameters["beam_size"] = 10
            if with_prior:
                alt_decoder_hyperparameters["prior_weight"] = 0.3
                
            if keep_best_decoding:
                alt_decoder_hyperparameters["keep_best_decoding"] = True
                
            if decode_every_step:
                every_step_hyperparameters = decoder_hyperparameters.copy()
                assert train_lm_config
                every_step_hyperparameters["lm_order"] = train_lm_config["order"] if train_lm_config["class"] == "ngram" else f"ffnn{train_lm_config['context_size']}"
                every_step_hyperparameters["lm_weight"] = 0.7 # 0.4
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
                
            if use_sum_criterion:# or decode_every_step:
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
            a3 = (f"-accum{accum_grad_multiple_step}" if accum_grad_multiple_step > 1 else "") + ("_every-step" + every_step_str if decode_every_step else "") + (("_tuneR" if tune_version == 3 else "_tune") if tune_hyperparameters else "") + ("_best" if keep_best_decoding else "")
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
    if init.endswith("unsupervised"):
        if decode_every_step_init:
            config_updates["decode_every_step"] = decode_every_step_init
            assert decoder_hyperparameters
            assert train_lm_config
            every_step_hyperparameters_init = decoder_hyperparameters.copy()
            every_step_hyperparameters_init["decay"] = 0.99995
            every_step_hyperparameters_init["decay_limit"] = 0.8
            every_step_hyperparameters_init["prior_weight"] = 0.0
            every_step_hyperparameters_init["lm_weight"] = 2.0
            every_step_hyperparameters_init["lm_order"] = train_lm_config["order"] if train_lm_config["class"] == "ngram" else f"ffnn{train_lm_config['context_size']}"
            config_updates["hyperparameters_decoder"] = every_step_hyperparameters_init
            if accum_grad_multiple_step_init > 1:
                config_updates["accum_grad_multiple_step"] = accum_grad_multiple_step_init
        if not random_init:
            assert with_prior and empirical_prior
            model_config["output_bias_init"] = True
        if num_gpus != 4:
            config_updates["__num_processes"] = num_gpus
    
    config_updates_self_training = None
    config_deletes_self_training = None
    LR_str = ""

    if not aux_loss:
      config_updates["aux_loss_layers"] = None

    if use_w2v_model:
      decoding_str += "_init_wv2ec"

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
          "init_for_train": True,
          "checkpoint_key": None,
        }
      }

      wav2vec_base_fine_tune_config = DownloadJob(
        "https://huggingface.co/facebook/wav2vec2-base/resolve/main/config.json?download=true",
        target_filename="wav2vec2_base_no_finetune_config.json",
      ).out_file
      config_updates["w2v_opts"] = {
        "config_file": wav2vec_base_fine_tune_config,
      }

      # TODO: add the following
      # sys.path.insert(0, "/work/asr3/zeyer/schmitt/venvs/transformers_package")
      # sys.path.insert(0, "/work/asr3/zeyer/schmitt/venvs/fairseq_package")
    
    # Create self-training config
    if self_training_rounds > 0:
        config_deletes_self_training = []
        config_updates_self_training = {
            **_get_cfg_lrlin_oclr_by_bs_nep(15_000, self_epochs),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
        }
        if adamw_betas:
            config_updates_self_training["optimizer.betas"] = adamw_betas
        if not reset_steps:
            if True:
                if pseudo_label_small:
                    config_updates_self_training["learning_rate_piecewise_steps"] = [20_000, 506_000, 562_000]
                else:
                    config_updates_self_training["learning_rate_piecewise_steps"] = [20_000, 558_000, 620_000]
                peak_lr = 5e-4
                config_updates_self_training["learning_rate_piecewise_values"] = [peak_lr * 2e-2, peak_lr, peak_lr * 2e-2, peak_lr * 2e-3]
                # LR_str = "_LR2-6e4"
            else:
                if pseudo_label_small:
                    config_updates_self_training["learning_rate_piecewise_steps"] = [253_000, 506_000, 562_000]
                else:
                    config_updates_self_training["learning_rate_piecewise_steps"] = [279_000, 558_000, 620_000]
                peak_lr = 1e-3
                config_updates_self_training["learning_rate_piecewise_values"] = [peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3]
                LR_str = "_LRwu-e3"
            # add something to hash so first training is different and correct epochs are saved
            if self_training_rounds != 4:
                config_updates_self_training["_st_rounds"] = self_training_rounds
        if decode_every_step:
            config_updates_self_training["decode_every_step"] = decode_every_step
            assert every_step_hyperparameters
            config_updates_self_training["hyperparameters_decoder"] = every_step_hyperparameters
        if pseudo_nbest > 1:
            assert train_lm_config
            config_updates_self_training["ps_nbest"] = pseudo_nbest
            dh = decoder_hyperparameters.copy()
            dh["lm_order"] = train_lm_config["order"] if train_lm_config["class"] == "ngram" else f"ffnn{train_lm_config['context_size']}"
            config_updates_self_training["hyperparameters_decoder"] = dh
            if norm_nbest_rescore:
                config_updates_self_training["norm_rescore"] = norm_nbest_rescore
            if not label_prior:
                config_updates_self_training["rescore_alignment_prior"] = True
            if gamma_scaling != 1.0:
                config_updates_self_training["gamma_scaling"] = gamma_scaling
        if accum_grad_multiple_step > 1:
            config_updates_self_training["accum_grad_multiple_step"] = accum_grad_multiple_step
        if not use_norm_st_loss:
            config_updates_self_training["use_normalized_loss"] = use_norm_st_loss
        if not speed_pert:
            config_deletes_self_training.append("speed_pert_discrete_values")
            config_updates_self_training.pop("__train_audio_preprocess")
        if not aux_loss:
            config_deletes_self_training.append("aux_loss_layers")
        if use_sgd:
            config_updates_self_training["optimizer"] = {
                "class": "sgd"
            }
        if decoding_imp == "gradients":
            config_updates_self_training["grad_nbest"] = grad_nbest
        if train_version != 1:
            config_updates_self_training["version"] = train_version
        if num_gpus != 4:
            config_updates_self_training["__num_processes"] = num_gpus
        if self_train_subset is not None:
            # When testing on a smaller subset we only want one gpu
            config_updates_self_training["__num_processes"] = 1
            # config_updates_self_training["learning_rate_piecewise_steps"] = [4_500, 9_000, 10_000]
            config_updates_self_training["learning_rate_piecewise_steps"] = [2_250, 4_500, 5_000]
            
            if not use_sgd:
                # peak_lr = 3e-5
                # config_updates_self_training["learning_rate_piecewise_values"] = [peak_lr * 1e-1, peak_lr, peak_lr * 1e-1, peak_lr * 1e-2]
                # LR_str = "_LRwu-3e5"
                peak_lr = 1e-4
                # config_updates_self_training["learning_rate_piecewise_values"] = [peak_lr * 1e-1, peak_lr, peak_lr * 3e-2, peak_lr * 3e-3]
                # LR_str = "_LRwu-e4"
                # config_updates_self_training["learning_rate_piecewise_values"] = [peak_lr, peak_lr, peak_lr, peak_lr]
                # LR_str = "_LRall-e4"
                config_updates_self_training["learning_rate_piecewise_values"] = [peak_lr, peak_lr, peak_lr * 0.27, peak_lr * 0.1]
                LR_str = "_LRedge-e4"
            else:
                peak_lr = 1e-2
                config_updates_self_training["learning_rate_piecewise_values"] = [peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3]
        # else:
        #     peak_lr = 1e-4
        #     config_updates_self_training["learning_rate_piecewise_values"] = [peak_lr, peak_lr, peak_lr * 0.27, peak_lr * 0.1]
        #     LR_str = "_LRedge-e4"

        if use_sum_criterion:
            config_full_sum = {
                "am_scale": am_lm_prior[0],
                "lm_scale": am_lm_prior[1],
                "prior_scale": am_lm_prior[2]
            }
            
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
            
            config_updates_self_training.update(config_full_sum)
            
            sum_str = f"-full_sum" + \
                f"_p{str(config_full_sum['prior_scale']).replace('.', '')}_l{str(config_full_sum['lm_scale']).replace('.', '')}_a{str(config_full_sum['am_scale']).replace('.', '')}" + \
                (f"_topK{top_k}" + ("_align" if alignment_topk else "") + (f"_bc{blank_correction_version}" + ("sc" if correction_in_final_score else "") if blank_correction_version > 0 else "") if top_k > 0 else "") + \
                ("_emp" if empirical_prior_full_sum else "") + \
                ("_max_pr" if not empirical_prior_full_sum and prior_from_max_full_sum else "") + \
                ("_wo_h_pr" if not horizontal_prior else "") + \
                ("_wo_b_pr" if not blank_prior else "") + \
                ("_wo_pr_grad" if not prior_gradient else "")
    
    alias_name = (("ctc" if not use_seq_gamma_loss else "seq_ce") if not use_ce_loss else "ce") + \
        (sum_str if use_sum_criterion else "") + \
        (f"-st_{self_training_rounds}" + LR_str + ("_no_norm" if not use_norm_st_loss else "") + ("_keep_LR" if not reset_steps else "") + ("_SGD" if use_sgd else (f"_b1-{str(adamw_betas[0]).replace('.', '')}_b2-{str(adamw_betas[1]).replace('.', '')}" if adamw_betas else "")) + ("_from_scratch" if from_scratch else "") + (f"_s{self_train_subset}" if self_train_subset is not None else "") + (f"_e{self_epochs}" if self_epochs != 450 else "") + ("_nsp" if not speed_pert else "") if self_training_rounds > 0 else "") + \
        (f"-wo_aux_loss" if not aux_loss else "") + \
        (f"-ds100h" if init == "100h-supervised" else ("-ds100US" + (f"_accum{accum_grad_multiple_step_init}" if accum_grad_multiple_step_init > 1 else "") + ("_emp_init" if not random_init else "") if init == "100h-unsupervised" else "")) + \
        (f"-pl960h" + ("_keep100h" if keep_small_labels else "") if not pseudo_label_small else "") + \
        f"-{vocab}" + \
        ("-laPR" if label_prior else "-frPR") + \
        f"{decoding_str}" + \
        (f"_v{train_version}" if train_version != 1 else "")
        
    if decoding_imp in ["flashlight", "marten-greedy"]:
        decoder_def = model_recog_lm
    elif decoding_imp == "albert-greedy":
        decoder_def = model_recog
    elif decoding_imp == "albert-flashlight":
        decoder_def = model_recog_flashlight
    elif decoding_imp == "albert-lm":
        decoder_def = model_recog_lm_albert
    elif decoding_imp == "gradients":
        decoder_def = (model_recog_lm_albert, model_recog_gradients)
        decoder_hyperparameters = (decoder_hyperparameters, decoder_hyperparameters_grad)
        if alt_decoder:
            alt_decoder_hyperparameters = (alt_decoder_hyperparameters, decoder_hyperparameters_grad)
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
        config_deletes_self_training = config_deletes_self_training,
        vocab = vocab,
        self_training_rounds = self_training_rounds,
        init_small = init == "100h-supervised",
        pseudo_label_small = pseudo_label_small,
        keep_small_labels = keep_small_labels,
        with_prior = with_prior,
        label_prior = label_prior,
        empirical_prior=empirical_prior,
        prior_from_max=prior_from_max,
        use_sum_criterion=use_sum_criterion,
        use_ce_loss=use_ce_loss,
        use_seq_gamma_loss=use_seq_gamma_loss,
        self_train_subset=self_train_subset,
        calc_last_pseudo_labels=calc_last_pseudo_labels,
        tune_hyperparameters=tune_hyperparameters,
        from_scratch=from_scratch,
        reset_steps=reset_steps,
    )
    

_train_experiments: Dict[str, ModelWithCheckpoints] = {}


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    decoder_def: Callable,
    *,
    decoder_hyperparameters: dict | tuple[dict, dict] = None,
    hyperparamters_self_training: dict | tuple[dict, dict] = None,
    pseudo_nbest: int = 1,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    vocab: str = "bpe10k",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    config_updates_self_training: Optional[Dict[str, Any]] = None,
    config_deletes_self_training: Optional[Sequence[str]] = None,
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
    label_prior: bool = True,
    empirical_prior: bool = False,
    prior_from_max: bool = False,
    use_sum_criterion: bool = False,
    use_ce_loss: bool = False,
    use_seq_gamma_loss: bool = False,
    self_train_subset: Optional[int] = None,
    calc_last_pseudo_labels: bool = False,
    tune_hyperparameters: bool = False,
    from_scratch: bool = False,
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
    gradient_pseudo_labels = isinstance(decoder_def, tuple)
    
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
        emp_prior = get_prior_from_unigram(task.prior_dataset.vocab, task.prior_dataset, vocab, label_prior)
        
    if config_updates.get("decode_every_step", False):
        config_updates["empirical_prior"] = emp_prior
    
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
            if not config.get("decode_every_step", False) or mc["train_language_model"]["class"] == "ngram":
                mc.pop("train_language_model", None)
        if "output_bias_init" in mc and mc["output_bias_init"]:
            mc["output_bias_init"] = "/u/marten.mueller/dev/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ExtractPrior.Z4T3thKgQeci/output/prior.txt"
        mc_self = model_config.copy()
        if "train_language_model" in mc_self and mc_self["train_language_model"]["class"] == "ngram":
            mc_self.pop("train_language_model", None)
        if "output_bias_init" in mc_self:
            mc_self.pop("output_bias_init")
        model_def = ModelDefWithCfg(model_def, mc)
        model_def_self = ModelDefWithCfg(model_def_self, mc_self)
    if not train_def:
        train_def = ctc_training
        
    # Calculate some DataSet Stats
    # calc_stats(task.train_dataset.vocab)
        
    # Create LM for training
    train_lm = None
    if (model_config and "train_language_model" in model_config) or gradient_pseudo_labels:
        if gradient_pseudo_labels:
            train_language_model = decoder_hyperparameters[1].pop("train_language_model")
        else:
            train_language_model = model_config["train_language_model"].copy()
        cls_name = train_language_model.pop("class")
        assert cls_name == "FeedForwardLm" or cls_name == "ngram"
        is_ffnn = cls_name == "FeedForwardLm"
        if is_ffnn:
            lm_checkpoint = get_ffnn_lm(task.train_dataset.vocab, **train_language_model)
            if not gradient_pseudo_labels:
                config_updates_self_training.update({
                    "preload_from_files": {
                        "train_lm": {
                            "init_for_train": True,
                            "prefix": "train_language_model.",
                            "filename": lm_checkpoint.checkpoint,
                        },
                    },
                })
                if config and config.get("decode_every_step", False):
                    config.update({
                        "preload_from_files": {
                            "train_lm": {
                                "prefix": "train_language_model.",
                                "filename": lm_checkpoint.checkpoint,
                            },
                        },
                    })
            train_lm = lm_checkpoint
        else:
            train_lm = get_count_based_n_gram(task.train_dataset.vocab, train_language_model["order"])
        
    # Get recog ffnn LM
    search_config = None
    recog_lm = None
    if model_config and "recog_language_model" in model_config:
        recog_language_model = model_config["recog_language_model"].copy()
        cls_name = recog_language_model.pop("class")
        assert cls_name == "FeedForwardLm"
        lm_checkpoint = get_ffnn_lm(task.train_dataset.vocab, **recog_language_model)
        recog_lm = lm_checkpoint
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
    
    scales = None 
    if tune_hyperparameters and tune_version >= 2:
        assert with_prior and empirical_prior
        assert recog_lm is not None
        from i6_experiments.users.zeyer.datasets.utils.vocab import (
            ExtractVocabLabelsJob,
            ExtractVocabSpecialLabelsJob,
            ExtendVocabLabelsByNewLabelJob,
        )
        vocab_opts = task.train_dataset.vocab.get_opts()
        vocab_file = ExtractVocabLabelsJob(vocab_opts).out_vocab
        vocab_opts_file = ExtractVocabSpecialLabelsJob(vocab_opts).out_vocab_special_labels_dict
        vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
            vocab=vocab_file, new_label=OUT_BLANK_LABEL, new_label_idx=-1
        ).out_vocab
        
        # tune parameters for decoding
        dec_params = decoder_hyperparameters[0] if isinstance(decoder_hyperparameters, tuple) else decoder_hyperparameters
        prior_tune, lm_tune = _tune_prior_and_lm(label_prior, task, emp_prior, search_config, recog_lm, model_with_checkpoint[0].get_last_fixed_epoch(), vocab_file, vocab_opts_file, vocab_w_blank_file, prefix + "/tune", dec_params, 128)
        dec_params["prior_weight"] = prior_tune
        dec_params["lm_weight"] = lm_tune
        scales = {"lm_weight": lm_tune}
        # tune parameters for training
        if config_updates_self_training and "hyperparameters_decoder" in config_updates_self_training:
            assert not isinstance(decoder_hyperparameters, tuple)
            if config_updates_self_training["hyperparameters_decoder"]["lm_order"] != decoder_hyperparameters["lm_order"]:
                assert train_lm is not None
                dec_params_train = config_updates_self_training["hyperparameters_decoder"]
                prior_tune_train, lm_tune_train = _tune_prior_and_lm(label_prior, task, emp_prior, search_config, train_lm, model_with_checkpoint[0].get_last_fixed_epoch(), vocab_file, vocab_opts_file, vocab_w_blank_file, prefix + "/tune_train", dec_params_train, 128)
                config_updates_self_training["hyperparameters_decoder"]["prior_weight"] = prior_tune_train
                config_updates_self_training["hyperparameters_decoder"]["lm_weight"] = lm_tune_train
            else:
                config_updates_self_training["hyperparameters_decoder"]["prior_weight"] = prior_tune
                config_updates_self_training["hyperparameters_decoder"]["lm_weight"] = lm_tune
        elif isinstance(decoder_hyperparameters, tuple):
            if decoder_hyperparameters[1]["lm_order"] != decoder_hyperparameters[0]["lm_order"]:
                assert train_lm is not None
                dec_params_train = decoder_hyperparameters[0]
                prior_tune_train, lm_tune_train = _tune_prior_and_lm(label_prior, task, emp_prior, search_config, train_lm, model_with_checkpoint[0].get_last_fixed_epoch(), vocab_file, vocab_opts_file, vocab_w_blank_file, prefix + "/tune_train", dec_params_train, 128)
                decoder_hyperparameters[1]["prior_weight"] = prior_tune_train
                decoder_hyperparameters[1]["lm_weight"] = lm_tune_train
            else:
                decoder_hyperparameters[1]["prior_weight"] = prior_tune
                decoder_hyperparameters[1]["lm_weight"] = lm_tune
        
    pst = hyperparamters_self_training[0] if hyperparamters_self_training is not None and isinstance(hyperparamters_self_training, tuple) else hyperparamters_self_training
    pseudo_label_path_dict = recog_training_exp(
        prefix,
        task,
        model_with_checkpoint[0],
        recog_def=decoder_def,
        decoder_hyperparameters=decoder_hyperparameters,
        save_pseudo_labels=(pseudo_labels_ds, train_100_ds) if calc_last_pseudo_labels or self_training_rounds > 0 else None,
        pseudo_label_alignment=use_ce_loss,
        pseudo_nbest=pseudo_nbest,
        calculate_pseudo_label_scores=calculate_pseudo_label_scores_init and not gradient_pseudo_labels, # NOTE: breaks hash
        search_config=search_config,
        recog_post_proc_funcs=recog_post_proc_funcs,
        search_mem_rqmt = 32 if gradient_pseudo_labels else 6,
        num_shards_recog=num_shards_recog_init, # NOTE: breaks hash
        num_shards_pseudo=num_shards_pseudo,
        num_shards_prior=num_shards_prior_init,
        is_last=self_training_rounds == 0,
        get_prev=(pst is not None and (pst.get("keep_best_decoding", False)), False),
        prior_from_max=prior_from_max,
        empirical_prior=emp_prior if with_prior and empirical_prior else None,
        cache_manager=cache_manager,
        check_train_scores_nbest=decode_nbest_epochs_init,
        exclude_epochs=sorted(list(model_with_checkpoint[0].fixed_epochs))[:-1] if not decode_all_fixed_epochs_init else (),
        return_beam=(self_training_rounds > 0 and config_updates_self_training.get("decode_every_step", False)),
        scales=scales,
    )
    
    # Do self training on pseudo labels
    for i in range(self_training_rounds):
        assert pseudo_label_path_dict is not None, "Pseudo label path is not set"
        assert hyperparamters_self_training is not None, "Hyperparameters for self training are not set"
        prefix_self_training = prefix + f"/self-training-{i+1}"
        task, _, _ = get_librispeech_task_raw_v2(
            vocab=vocab,
            train_vocab_opts=train_vocab_opts,
            ds_sel = TrainDatasetSel.train_860h if pseudo_label_small else TrainDatasetSel.train_960h,
            init_small=init_small,
            with_prior=with_prior,
            empirical_prior=empirical_prior,
            pseudo_label_path = pseudo_label_path_dict,
            pseudo_label_alignment = (config_updates_self_training["grad_nbest"] if gradient_pseudo_labels else 0) if use_ce_loss else -1,
            pseudo_label_nbest = pseudo_nbest,
            pseudo_label_scores = config_updates_self_training.get("decode_every_step", False),
            keep_small_labels = keep_small_labels,
            train_subset = self_train_subset,
            eval_subset = 0 if use_ce_loss else (300 if self_train_subset else 3000),
            train_epoch_wise_filter = None
        )
        
        config_self = config.copy()
        config_self = dict_update_deep(config_self, config_updates_self_training, config_deletes_self_training)
        # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
        if "__train_audio_preprocess" in config_self:
            task: Task = copy.copy(task)
            task.train_dataset = copy.copy(task.train_dataset)
            task.train_dataset.train_audio_preprocess = config_self.pop("__train_audio_preprocess")
        else:
            task: Task = copy.copy(task)
            task.train_dataset = copy.copy(task.train_dataset)
            task.train_dataset.train_audio_preprocess = None
        
        if use_sum_criterion:
            train_def = ctc_sum_training
            if isinstance(train_lm, tk.Path):
                config_self["train_lm_model"] = train_lm
            else:
                assert isinstance(train_lm, ModelWithCheckpoint)
                config_self["train_lm_model"] = "ffnn" + str(model_config["train_language_model"]["context_size"])
        elif use_ce_loss:
            train_def = ce_training
        elif use_seq_gamma_loss:
            train_def = seq_gamma_training
            
        if config_self.get("empirical_prior", False) or config_self.get("decode_every_step", False) or config_self.get("ps_nbest", 1) > 1:
            config_self["empirical_prior"] = emp_prior
        
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
            finish_all=config_self.get("decode_every_step", False),
            num_epochs=num_epochs,
            gpu_mem=gpu_mem,
            num_processes=num_processes,
            time_rqmt=time_rqmt if time_rqmt else ((8 if self_train_subset else 156) if use_sum_criterion else 156),
        ))
        train_job = model_with_checkpoint[i + 1].get_training_job()
        if env_updates:
            for k, v in env_updates.items():
                train_job.set_env(k, v)
        
        scales = None
        if tune_hyperparameters:
            assert tune_version >= 2
            # tune parameters for decoding
            dec_params = hyperparamters_self_training[0] if isinstance(hyperparamters_self_training, tuple) else hyperparamters_self_training
            prior_tune, lm_tune = _tune_prior_and_lm(label_prior, task, emp_prior, search_config, recog_lm, model_with_checkpoint[i + 1].get_last_fixed_epoch(), vocab_file, vocab_opts_file, vocab_w_blank_file, prefix_self_training + "/tune", dec_params, 128)
            dec_params["prior_weight"] = prior_tune
            dec_params["lm_weight"] = lm_tune
            scales = {"lm_weight": lm_tune}
            # tune parameters for training
            if "hyperparameters_decoder" in config_updates_self_training:
                assert not isinstance(hyperparamters_self_training, tuple)
                if config_updates_self_training["hyperparameters_decoder"]["lm_order"] != hyperparamters_self_training["lm_order"]:
                    dec_params_train = config_updates_self_training["hyperparameters_decoder"]
                    prior_tune_train, lm_tune_train = _tune_prior_and_lm(label_prior, task, emp_prior, search_config, train_lm, model_with_checkpoint[i + 1].get_last_fixed_epoch(), vocab_file, vocab_opts_file, vocab_w_blank_file, prefix_self_training + "/tune_train", dec_params_train, 128)
                    config_updates_self_training["hyperparameters_decoder"]["prior_weight"] = prior_tune_train
                    config_updates_self_training["hyperparameters_decoder"]["lm_weight"] = lm_tune_train
                else:
                    config_updates_self_training["hyperparameters_decoder"]["prior_weight"] = prior_tune
                    config_updates_self_training["hyperparameters_decoder"]["lm_weight"] = lm_tune
            elif isinstance(hyperparamters_self_training, tuple):
                if hyperparamters_self_training[1]["lm_order"] != hyperparamters_self_training[0]["lm_order"]:
                    dec_params_train = hyperparamters_self_training[0]
                    prior_tune_train, lm_tune_train = _tune_prior_and_lm(label_prior, task, emp_prior, search_config, train_lm, model_with_checkpoint[i + 1].get_last_fixed_epoch(), vocab_file, vocab_opts_file, vocab_w_blank_file, prefix_self_training + "/tune_train", dec_params_train, 128)
                    hyperparamters_self_training[1]["prior_weight"] = prior_tune_train
                    hyperparamters_self_training[1]["lm_weight"] = lm_tune_train
                else:
                    hyperparamters_self_training[1]["prior_weight"] = prior_tune
                    hyperparamters_self_training[1]["lm_weight"] = lm_tune
        
        sc = search_config.copy()
        if isinstance(hyperparamters_self_training, tuple):
            hst = (hyperparamters_self_training[0].copy(), hyperparamters_self_training[1])
            if hst[0].get("keep_best_decoding", False):
                hst[0].pop("keep_best_decoding")
                sc["__prev_hyps"] = pseudo_label_path_dict
        else:
            hst = hyperparamters_self_training.copy()
            if hst.get("keep_best_decoding", False):
                hst.pop("keep_best_decoding")
                sc["__prev_hyps"] = pseudo_label_path_dict
        
        pseudo_label_path_dict = recog_training_exp(
            prefix_self_training,
            task,
            model_with_checkpoint[i + 1],
            recog_def=decoder_def,
            decoder_hyperparameters=hst,
            save_pseudo_labels=None if not calc_last_pseudo_labels and i+1 == self_training_rounds else (pseudo_labels_ds, train_100_ds),
            pseudo_label_alignment=use_ce_loss,
            pseudo_nbest=pseudo_nbest,
            calculate_pseudo_label_scores=calculate_pseudo_label_scores and not gradient_pseudo_labels,
            search_config=sc,
            recog_post_proc_funcs=recog_post_proc_funcs,
            search_mem_rqmt = 32 if gradient_pseudo_labels else 6,
            num_shards_recog=num_shards_recog, # NOTE: breaks hash
            num_shards_pseudo=num_shards_pseudo,
            num_shards_prior=num_shards_prior,
            is_last=i+1 == self_training_rounds,
            get_prev=(pst.get("keep_best_decoding", False), pst.get("keep_best_decoding", False)),
            prior_from_max=prior_from_max,
            empirical_prior=emp_prior if with_prior and empirical_prior else None,
            cache_manager=cache_manager,
            check_train_scores_nbest=decode_nbest_epochs,
            exclude_epochs=sorted(list(model_with_checkpoint[i + 1].fixed_epochs))[:-1] if not decode_all_fixed_epochs else (),
            return_beam = config_updates_self_training.get("decode_every_step", False),
            scales=scales,
        )

    _train_experiments[name] = model_with_checkpoint[-1]
    return model_with_checkpoint[-1]

def _tune_prior_and_lm(label_prior, task, prior_file, search_config, lm, model_with_checkpoint, vocab_file, vocab_opts_file, vocab_w_blank_file, prefix, recomb_params, beam_size):
    from i6_experiments.users.mueller.scale_tune import ctc_recog_framewise_prior_auto_scale, ctc_recog_labelwise_prior_auto_scale
    
    tune_config = search_config.copy()
    recomb_config = None
    if tune_version == 2:
        tune_config["beam_size"] = beam_size
    else:
        recomb_config = {
            "beam_size": beam_size,
            "ps_nbest": beam_size,
            "use_recombination": recomb_params.get("use_recombination", False),
            "recomb_blank": recomb_params.get("recomb_blank", False),
            "recomb_after_topk": recomb_params.get("recomb_after_topk", False),
            "recomb_with_sum": recomb_params.get("recomb_with_sum", False),
        }
                
    if label_prior:
        prior_tune, lm_tune = ctc_recog_labelwise_prior_auto_scale(
            prefix = prefix,
            task = task,
            ctc_model=model_with_checkpoint,
            prior_file=prior_file,
            lm=lm,
            vocab_file=vocab_file,
            vocab_opts_file=vocab_opts_file,
            num_shards=num_shards_recog,
            search_config=tune_config,
            recomb_config=recomb_config,
        )
    else:
        prior_tune, lm_tune = ctc_recog_framewise_prior_auto_scale(
            prefix = prefix,
            task = task,
            ctc_model=model_with_checkpoint,
            prior_file=prior_file,
            lm=lm,
            vocab_file=vocab_file,
            vocab_opts_file=vocab_opts_file,
            vocab_w_blank_file=vocab_w_blank_file,
            num_shards=num_shards_recog,
            search_config=tune_config,
            recomb_config=recomb_config,
        )
        
    return prior_tune, lm_tune


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

def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Union[Model, Wav2VecModel]:
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

    train_language_model = config.typed_value("train_language_model", None)
    train_lm = None
    if train_language_model is not None:
        assert isinstance(train_language_model, dict)
        train_language_model = train_language_model.copy()
        cls_name = train_language_model.pop("class")
        assert cls_name == "FeedForwardLm"
        train_lm = FeedForwardLm(vocab_dim=target_dim, **train_language_model)
    recog_language_model = config.typed_value("recog_language_model", None)
    recog_lm = None
    if recog_language_model is not None:
        assert isinstance(recog_language_model, dict)
        recog_language_model = recog_language_model.copy()
        cls_name = recog_language_model.pop("class")
        assert cls_name == "FeedForwardLm"
        recog_lm = FeedForwardLm(vocab_dim=target_dim, **recog_language_model)

    if config.bool("use_w2v_model", False):
        w2v_opts = config.typed_value("w2v_opts", {})
        w2v_config_file = w2v_opts["config_file"]
        wav2vec2_config = transformers.Wav2Vec2Config.from_pretrained(w2v_config_file)
        return Wav2VecModel(
            wav2vec_config=wav2vec2_config,
            target_dim=target_dim,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            train_language_model=train_lm,
            recog_language_model=recog_lm,
        )

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
    
    output_bias_init = config.typed_value("output_bias_init", None)

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
        output_bias_init=output_bias_init,
    )

ctc_model_def: ModelDef[Model]
ctc_model_def.behavior_version = 21
ctc_model_def.backend = "torch"
ctc_model_def.batch_size_factor = _batch_size_factor


#---------------------------------------------------------------------------------------------------------------------------------------
# TRAINING DEFINITIONS

def ctc_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim, nbest_lengths: rf.Tensor = None, scores: rf.Tensor = None, seq_tags: rf.Tensor = None):
    return ctc_train(model=model, data=data, data_spatial_dim=data_spatial_dim, targets=targets, targets_spatial_dim=targets_spatial_dim, nbest_lengths=nbest_lengths, scores=scores, seq_tags=seq_tags)

ctc_training: TrainDef[Model]
ctc_training.learning_rate_control_error_measure = "ctc"

def ctc_sum_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, seq_tags: rf.Tensor = None, targets: rf.Tensor, targets_spatial_dim: Dim):
    return full_sum_train(model=model, data=data, data_spatial_dim=data_spatial_dim, seq_tags=seq_tags, targets=targets, targets_spatial_dim=targets_spatial_dim)

ctc_sum_training: SumTrainDef[Model]
ctc_sum_training.learning_rate_control_error_measure = "full_sum"

def ce_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim, targets_indices: rf.Tensor = None):
    return ce_train(model=model, data=data, data_spatial_dim=data_spatial_dim, targets=targets, targets_spatial_dim=targets_spatial_dim, targets_indices=targets_indices)

ce_training: CETrainDef[Model]
ce_training.learning_rate_control_error_measure = "ce"

def seq_gamma_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim, nbest_lengths: rf.Tensor = None, scores: rf.Tensor = None, seq_tags: rf.Tensor = None):
    return seq_gamma_ctc_train(model=model, data=data, data_spatial_dim=data_spatial_dim, targets=targets, targets_spatial_dim=targets_spatial_dim, nbest_lengths=nbest_lengths, scores=scores, seq_tags=seq_tags)

seq_gamma_training: TrainDef[Model]
seq_gamma_training.learning_rate_control_error_measure = "seq_ce"


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


def model_recog_gradients(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    hyperparameters: dict,
    prior_file: tk.Path = None,
    arpa_lm: Optional[str] = None,
    version: Optional[int] = None,
    seq_tags: Optional[Tensor] = None
) -> Tuple[Tensor, Dim]:
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
    
    # The label log probs include the AM
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    
    print_idx = []
    
    return recog_gradients(model=model, label_log_prob=label_log_prob, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, prior_file=prior_file, arpa_lm=arpa_lm, version=version, print_idx=print_idx)

# RecogDef API
model_recog_gradients: RecogDef[Model]
model_recog_gradients.output_with_beam = False
model_recog_gradients.output_blank_label = OUT_BLANK_LABEL
model_recog_gradients.batch_size_dependent = True  # our models currently just are batch-size-dependent...