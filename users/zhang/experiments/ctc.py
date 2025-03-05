"""
CTC experiments.
"""

from __future__ import annotations

import copy
import pdb
import sys
import functools
from typing import TYPE_CHECKING, Optional, Callable, Union, Tuple, Sequence, Collection, List
import numpy as np
import re
import json
import os

from sisyphus import tk
import returnn.frontend as rf
import returnn.torch.frontend as rtf
from returnn.tensor import Tensor, Dim, batch_dim
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zhang.experiments.lm.ffnn import FFNN_LM_flashlight, FeedForwardLm
from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob

from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor

if TYPE_CHECKING:
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.zhang.datasets.task import Task
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

OUT_BLANK_LABEL = "<blank>"
CHECK_DECODER_CONSISTENCY = False
# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

_raw_sample_rate = _batch_size_factor * 100  # bs factor is from 10ms frames to raw samples


def py():
    """Sisyphus entry point"""

    # relPosAttDef: Use the default RelPosSelfAttention instead of the Shawn et al 2018 style, old RETURNN way.
    enc_conformer_layer_default = rf.build_dict(
        rf.encoder.conformer.ConformerEncoderLayer,
        ff_activation=rf.build_dict(rf.relu_square),
        num_heads=8,
    )

    #-------------------setting decoding config-------------------
    decoding_config = {
        "log_add": False,
        "nbest": 1,
        "beam_size": 2,
        "lm_weight": 1.4,  # NOTE: weights are exponentials of the probs. 1.9 seems make the results worse by using selftrained lm
        "use_logsoftmax": True,
        "use_lm": True,
        "use_lexicon": False, # Do open vocab search when using bpe lms.
    }




    # ---------------------------------------------------------
    # model name: f"v6-relPosAttDef-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100"
    #               f"-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2"
    #               f"-[vocab]" + f"-[sample]"
    exclude_epochs = set(range(0, 500)) - set([477]) #Reduce #ForwardJobs, 477 is the best epoch for most cases.

    for vocab, sample, alpha in [
        # ("spm20k", None, None),  # 5.96
        # ("spm20k", "spm", 0.7),  # 6.14
        # # TODO ("spm20k", "bpe", 0.005),
        # ("spm20k", "bpe", 0.01),  # 6.13
        # ("spm20k", "bpe", 0.02),  # 6.21
        # ("bpe10k", "bpe", 0.005),  # 6.48
        # ("bpe10k", "bpe", 0.01),  # 6.40
        # ("spm10k", None, None),  # 6.00
        # # TODO ("spm10k", "spm", 0.8),
        # ("spm10k", "spm", 0.7),  # 6.20
        # ("spm10k", "bpe", 0.001),  # 5.93
        # ("spm10k", "bpe", 0.005),  # 5.89 (!!)
        # ("spm10k", "bpe", 0.01),  # 5.93
        # ("spm_bpe10k", None, None),  # 6.33
        # ("spm_bpe10k", "spm", 1e-4),  # 6.26
        # # TODO ("spm_bpe10k", "bpe", 0.005),
        # ("spm_bpe10k", "bpe", 0.01),  # 6.21
        # ("spm4k", None, None),  # 6.07 (but test-other even better: 5.94?)
        # ("spm4k", "spm", 0.7),  # 6.42
        # # TODO ("spm4k", "bpe", 0.005),
        # ("spm4k", "bpe", 0.01),  # 6.05
        # ("spm1k", None, None),  # 6.07
        # ("spm1k", "spm", 1.0),  # 6.73
        # ("spm1k", "spm", 0.99),  # 6.93
        # ("spm1k", "spm", 0.9),  # 7.04
        # ("spm1k", "spm", 0.7),  # 7.33
        # ("spm1k", "bpe", 0.0),  # 6.07
        # # TODO ("spm1k", "bpe", 0.0005),
        # ("spm1k", "bpe", 0.001),  # 6.15
        # ("spm1k", "bpe", 0.005),  # 6.25
        # ("spm1k", "bpe", 0.01),  # 6.13 (but dev-clean,test-* are better than no sampling)
        # ("spm_bpe1k", None, None),  # 6.03
        # ("spm_bpe1k", "bpe", 0.01),  # 6.05
        # ("spm512", None, None),  # 6.08
        # ("spm512", "bpe", 0.001),  # 6.05
        # ("spm512", "bpe", 0.005),  # 6.01
        # ("spm512", "bpe", 0.01),  # 6.08 (but test-* is better than spm512 without sampling)
        # ("spm128", None, None),  # 6.37
        # # TODO ("spm128", "bpe", 0.001),
        # ("spm128", "bpe", 0.01),  # 6.40
        # # TODO ("spm128", "bpe", 0.005),
        ("bpe128", None, None),
        # ("bpe10k", None, None),  # 6.49  # Require much more time on recog even with lexicon
        # ("spm64", None, None),
        # ("bpe64", None, None),
        # ("utf8", None, None),
        # ("char", None, None),
        # ("bpe0", None, None),
    ]:
        # from ..datasets.librispeech_lm import get_4gram_binary_lm
        from i6_experiments.users.zhang.experiments.language_models.n_gram import get_count_based_n_gram # If move to lm it somehow breaks the hash...
        lms = dict()
        wer_ppl_results = dict()
        ppl_results = dict()
        # vocabs = {str(i) + "gram":_get_bpe_vocab(i) for i in [3,4,5,6,7,8,9,10]}
        # ----------------------Add bpe count based n-gram LMs(Only if ctc also operates on same bpe)--------------------
        if re.match("^bpe[0-9]+.*$", vocab):
            #bpe = bpe10k if vocab == "bpe10k" else _get_bpe_vocab(bpe_size=vocab[len("bpe") :])
            for n_order in [#2, 3,
                      #4, 5, 6,
                      #7, 8, # TODO: Currently KenLM only support up to 6 order
                      #9, 10
                      ]: # Assume we only do bpe for now
                for prune_thresh in [x*1e-9 for x in range(1,20,3)]:
                    lm_name = str(n_order) + "gram" + f"{prune_thresh:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
                    lm, ppl_var = get_count_based_n_gram(vocab, n_order, prune_thresh)
                    lms.update(dict([(lm_name, lm)]))
                    ppl_results.update(dict([(lm_name, ppl_var)]))
                    tk.register_output(f"datasets/LibriSpeech/lm/ppl/" + lm_name + "_" + vocab, ppl_var)
        # ----------------------Add word count based n-gram LMs--------------------
        exp_names_postfix = ""
        prune_num = 0
        for n_order in [  # 2, 3,
            4, #5
            # 6 Slow recog
        ]:
            exp_names_postfix += str(n_order) + "_"
            prune_threshs = list(set([x * 1e-9 for x in range(1, 100, 6)] +
                                     [x * 1e-7 for x in range(1, 200, 25)] +
                                     [x * 1e-7 for x in range(1, 200, 16)] +
                                     [x * 1e-7 for x in range(1, 10, 2)]
                                     ))
            prune_threshs.sort()
            if n_order == 5:
                prune_threshs = [x for x in prune_threshs if x and x < 9*1e-6]
            prune_threshs.append(None)
            for prune_thresh in prune_threshs: # + [None]: #[x * 1e-9 for x in range(1, 30, 3)]
                prune_num += 1 if prune_thresh else 0
                lm, ppl_log = get_count_based_n_gram("word", n_order, prune_thresh)
                lm_name = str(n_order) + "gram_word" + (f"{prune_thresh:.1e}" if prune_thresh else "").replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
                lms.update(dict([(lm_name, lm)]))
                ppl_results.update(dict([(lm_name, ppl_log)]))
                tk.register_output(f"datasets/LibriSpeech/lm/ppl/" + lm_name, ppl_log)
        exp_names_postfix += f"pruned_{str(prune_num)}"
        # Try to use the out of downstream job which has existing logged output. Instead of just Forward job, which seems cleaned up each time
        lms.update({"NoLM": None})
        # for prunning in [(None, 5), (None, 6), (None, 7), (None, 8)]: # (thresholds, quantization level)
        #     official_4gram, ppl_official4gram = get_4gram_binary_lm(**dict(zip(["prunning","quant_level"],prunning)))
        #     lms.update({"4gram_word_official"+f"q{prunning[1]}": official_4gram})
        #     ppl_results.update({"4gram_word_official"+f"q{prunning[1]}": ppl_official4gram})
        print(ppl_results)

        for name, lm in lms.items():
            # lm_name = lm if isinstance(lm, str) else lm.name
            decoding_config["lm"] = lm
            decoding_config["use_lm"] = True if lm else False
            if re.match(r".*word.*", name) or "NoLM" in name:
                decoding_config["use_lexicon"] = True
            else:
                decoding_config["use_lexicon"] = False
            p1 = "sum" if decoding_config['log_add'] else "max"
            p2 = f"n{decoding_config['nbest']}"
            p3 = f"b{decoding_config['beam_size']}"
            p4 = f"w{str(decoding_config['lm_weight']).replace('.', '')}"
            p5 = "_logsoftmax" if decoding_config['use_logsoftmax'] else ""
            p6 = "_lexicon" if decoding_config['use_lexicon'] else ""
            lm_hyperparamters_str = f"{p1}_{p2}_{p3}_{p4}{p5}{p6}"
            lm_hyperparamters_str = vocab + lm_hyperparamters_str  # Assume only experiment on one ASR model, so the difference of model itself is not reflected here

            alias_name = f"ctc-baseline" + "decodingWith_" + lm_hyperparamters_str + name if lm else f"ctc-baseline-" + vocab + name
            print(f"name{name}","lexicon:" + str(decoding_config["use_lexicon"]))
            _, wer_result_path, _ = train_exp(
                name=alias_name,
                config=config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
                decoder_def=model_recog_lm,
                model_config={"enc_conformer_layer": enc_conformer_layer_default, "feature_batch_norm": True},
                config_updates={
                    **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
                    "optimizer.weight_decay": 1e-2,
                    "__train_audio_preprocess": speed_pert_librosa_config,
                    "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                    "max_seq_length_default_target": None,
                    # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                    # out of 281241 seqs in train, we removed only 71 seqs.
                    # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                    "max_seq_length_default_input": 19.5 * _raw_sample_rate,
                },
                vocab=vocab,
                train_vocab_opts=(
                    {
                        "other_opts": (
                            {
                                "spm": {"enable_sampling": True, "alpha": 0.005},
                                "bpe": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.005},
                            }[sample]
                        )
                    }
                    if sample
                    else None
                ),
                decoding_config=decoding_config,
                exclude_epochs=exclude_epochs,
                search_mem_rqmt= 30 if vocab == "bpe10k" else 6
            )
            if lm:
                wer_ppl_results[name] = (ppl_results.get(name), wer_result_path)
        (names, results) = zip(*wer_ppl_results.items())
        summaryjob = WER_ppl_PlotAndSummaryJob(names, results)
        tk.register_output("wer_ppl/"+lm_hyperparamters_str + exp_names_postfix + "/summary", summaryjob.out_summary)
        tk.register_output("wer_ppl/"+ lm_hyperparamters_str + exp_names_postfix + "/plot.png", summaryjob.out_plot)

_train_experiments: Dict[str, ModelWithCheckpoints] = {}


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    decoder_def: Callable,
    *,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    vocab: str = "bpe10k",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    dataset_train_opts: Optional[Dict[str, Any]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None, # Returnn config
    config_updates: Optional[Dict[str, Any]] = None, # For iterative changing of config
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    epilog: Sequence[serialization.SerializerObject] = (),
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
    env_updates: Optional[Dict[str, str]] = None,
    enabled: bool = True,
    decoding_config: dict = None,
    exclude_epochs: Collection[int] = (),
    recog_epoch: int = None,
    with_prior: bool = False,
    empirical_prior: bool = False,
    prior_from_max: bool = False,
    tune_hyperparameters: bool = False,
    search_mem_rqmt: Union[int, float] = 6,
    search_rqmt: dict = None,
) -> Tuple[Optional[ModelWithCheckpoints], Optional[tk.path], Optional[tk.path], Optional[tk.path]]:
    """
    Train experiment
    """
    from i6_experiments.users.zeyer.train_v3 import train
    from i6_experiments.users.zhang.recog import recog_training_exp, recog_exp, GetBestTuneValue
    from i6_experiments.users.zhang.datasets.librispeech import get_librispeech_task_raw_v2
    from i6_experiments.users.zhang.experiments.language_models.n_gram import get_prior_from_unigram

    if not enabled:
        return None

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    # TODO： find out how to apply recognition on other dataset
    task = get_librispeech_task_raw_v2(vocab=vocab,
                                       train_vocab_opts=train_vocab_opts,
                                       with_prior=with_prior,
                                       empirical_prior=empirical_prior,
                                       **(dataset_train_opts or {}))
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
    search_config = None
    if model_config:
        if "recog_language_model" in model_config:
            # recog_language_model = model_config["recog_language_model"].copy()
            # cls_name = recog_language_model.pop("class")
            # assert cls_name == "FeedForwardLm"
            # lm_checkpoint = get_ffnn_lm(task.train_dataset.vocab, **recog_language_model)
            search_config = model_config.pop("recog_language_model")

        model_def = ModelDefWithCfg(model_def, model_config)
    if not train_def:
        train_def = ctc_training

    model_with_checkpoint = train(
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
        time_rqmt=time_rqmt,
        env_updates=env_updates,
    )

    recog_post_proc_funcs = []
    if config.get("use_eos_postfix", False):
        recog_post_proc_funcs.append(_remove_eos_label_v2)

    lm_scale = decoding_config.get("lm_weight")
    if tune_hyperparameters:
        original_params = decoding_config
        params = copy.copy(original_params)
        params.pop("lm_weight_tune", None)
        params.pop("prior_weight_tune", None)
        default_lm = original_params.get("lm_weight")
        default_prior = original_params.get("prior_weight")
        lm_scores = []
        prior_scores = []
        lm_tune_ls = [scale/100 for scale in range(-50,51,5)] #[-0.5,-0.45....+0.45,+0.5]  [scale/100 for scale in range(-50,51,10/5)] for bpe10k/bpe128
        prior_tune_ls = [-0.05, -0.1, 0.0, 0.05, 0.1]
        for dc_lm in lm_tune_ls:
            params["lm_weight"] = default_lm + dc_lm
            task_copy = copy.deepcopy(task)
            # score = recog_training_exp(
            #     prefix + f"/tune/lm/{str(dc_lm).replace('.', '').replace('-', 'm')}",
            #     task_copy,
            #     model_with_checkpoint,
            #     recog_def=decoder_def,
            #     decoding_config=params,
            #     recog_post_proc_funcs=recog_post_proc_funcs,
            #     exclude_epochs=exclude_epochs,
            #     search_mem_rqmt=search_mem_rqmt,
            #     prior_from_max=prior_from_max,
            #     empirical_prior=emp_prior if with_prior and empirical_prior else None,
            #     dev_sets=["dev-other"],
            # )
            score, _ = recog_exp(
                prefix + f"/tune/lm/{str(dc_lm).replace('.', '').replace('-', 'm')}",
                task_copy,
                model_with_checkpoint,
                epoch=recog_epoch,
                recog_def=decoder_def,
                decoding_config=params,
                search_config=search_config,
                recog_post_proc_funcs=recog_post_proc_funcs,
                exclude_epochs=exclude_epochs,
                search_mem_rqmt=search_mem_rqmt,
                prior_from_max=prior_from_max,
                empirical_prior=emp_prior if with_prior and empirical_prior else None,
                dev_sets=["dev-other"],
                search_rqmt=search_rqmt,
            )
            lm_scores.append(score)

        if len(lm_scores):
            best_lm_tune = GetBestTuneValue(lm_scores, lm_tune_ls).out_best_tune
            tk.register_output(prefix + "/tune/lm_best", best_lm_tune)
            params["lm_weight"] = default_lm
            params["lm_weight_tune"] = best_lm_tune # Prior tuned on best lm_scale
            if os.path.exists(best_lm_tune): # Just for bypassing the static check
                lm_weight_tune = json.load(open(best_lm_tune))
                lm_weight_tune = lm_weight_tune["best_tune"]
                lm_scale = default_lm + lm_weight_tune
            original_params["lm_weight_tune"] = best_lm_tune  # This will be implicitly used by following exps, i.e through decoding_config
        if with_prior:
            for dc_prior in prior_tune_ls:
                params["prior_weight"] = default_prior + dc_prior
                task_copy = copy.deepcopy(task)
                # score = recog_training_exp(
                #     prefix + f"/tune/prior/{str(dc_prior).replace('.', '').replace('-', 'm')}",
                #     task_copy,
                #     model_with_checkpoint,
                #     recog_def=decoder_def,
                #     decoding_config=params,
                #     recog_post_proc_funcs=recog_post_proc_funcs,
                #     exclude_epochs=exclude_epochs,
                #     search_mem_rqmt=search_mem_rqmt,
                #     prior_from_max=prior_from_max,
                #     empirical_prior=emp_prior if with_prior and empirical_prior else None,
                #     dev_sets=["dev-other"],
                #     search_rqmt=search_rqmt,
                # )
                score, _ = recog_exp(
                    prefix + f"/tune/prior/{str(dc_prior).replace('.', '').replace('-', 'm')}",
                    task_copy,
                    model_with_checkpoint,
                    epoch=recog_epoch,
                    recog_def=decoder_def,
                    decoding_config=params,
                    search_config=search_config,
                    recog_post_proc_funcs=recog_post_proc_funcs,
                    exclude_epochs=exclude_epochs,
                    search_mem_rqmt=search_mem_rqmt,
                    prior_from_max=prior_from_max,
                    empirical_prior=emp_prior if with_prior and empirical_prior else None,
                    dev_sets=["dev-other"],
                    search_rqmt=search_rqmt,
                )
                prior_scores.append(score)
        if len(prior_scores):
            best_prior_tune = GetBestTuneValue(prior_scores, prior_tune_ls).out_best_tune
            tk.register_output(prefix + "/tune/prior_best", best_prior_tune)
            original_params["prior_weight_tune"] = best_prior_tune


    # recog_result = recog_training_exp(
    #     prefix, task, model_with_checkpoint, recog_def=decoder_def,
    #     decoding_config=decoding_config,
    #     recog_post_proc_funcs=recog_post_proc_funcs,
    #     exclude_epochs=exclude_epochs,
    #     search_mem_rqmt=search_mem_rqmt,
    #     prior_from_max=prior_from_max,
    #     empirical_prior=emp_prior if with_prior and empirical_prior else None,
    #     dev_sets=["test-other","dev-other"],
    #     search_rqmt=search_rqmt,
    # )
    recog_result, search_error = recog_exp(
        prefix, task, model_with_checkpoint,
        epoch=recog_epoch,
        recog_def=decoder_def,
        decoding_config=decoding_config,
        search_config=search_config,
        recog_post_proc_funcs=recog_post_proc_funcs,
        exclude_epochs=exclude_epochs,
        search_mem_rqmt=search_mem_rqmt,
        prior_from_max=prior_from_max,
        empirical_prior=emp_prior if with_prior and empirical_prior else None,
        dev_sets=["test-other","dev-other"],
        search_rqmt=search_rqmt,
        search_error_check=True,
    )

    _train_experiments[name] = model_with_checkpoint
    return model_with_checkpoint, recog_result, search_error, lm_scale


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


def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    return Model(**_get_ctc_model_kwargs_from_global_config(target_dim=target_dim))


ctc_model_def: ModelDef[Model]
ctc_model_def.behavior_version = 21
ctc_model_def.backend = "torch"
ctc_model_def.batch_size_factor = _batch_size_factor


def _get_ctc_model_kwargs_from_global_config(*, target_dim: Dim) -> Dict[str, Any]:
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    num_enc_layers = config.int("num_enc_layers", 12)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)

    enc_input_layer = config.typed_value("enc_input_layer", None)
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

    recog_language_model = config.typed_value("recog_language_model", None)
    recog_lm = None

    if recog_language_model:
        assert isinstance(recog_language_model, dict)
        recog_language_model = recog_language_model.copy()
        cls_name = recog_language_model.pop("class")
        assert cls_name == "FeedForwardLm"
        recog_lm = FeedForwardLm(vocab_dim=target_dim, **recog_language_model)

    return dict(
        in_dim=in_dim,
        enc_build_dict=config.typed_value("enc_build_dict", None),  # alternative more generic/flexible way
        num_enc_layers=num_enc_layers,
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_input_layer=enc_input_layer,
        enc_conformer_layer=enc_conformer_layer,
        enc_other_opts=enc_other_opts,
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_aux_logits=enc_aux_logits or (),
        recog_language_model=recog_lm,
    )


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


def ctc_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    use_fixed_ctc_grad = config.typed_value("use_fixed_ctc_grad", False)

    ctc_loss = rf.ctc_loss
    if use_fixed_ctc_grad:
        from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad

        assert use_fixed_ctc_grad == "v2"  # v2 has the fix for scaled/normalized CTC loss
        ctc_loss = ctc_loss_fixed_grad

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    if config.bool("use_eos_postfix", False):
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    collected_outputs = {} if aux_loss_layers else None
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_loss = ctc_loss(
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
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    log_probs = model.log_probs_wb_from_logits(logits)
    loss = ctc_loss(
        logits=log_probs,
        logits_normalized=True,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    loss.mark_as_loss(
        "ctc",
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
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

        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name="aed_fer", as_error=True)


ctc_training: TrainDef[Model]
ctc_training.learning_rate_control_error_measure = "ctc"


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
    # Just Initialisation
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
        # Since it is log_prob, + means the prob is actually product,
        # Note for CTC each time step is independent to each other.
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
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False  # not totally correct, but we treat it as such...


def model_recog_lm(
        *,
        model: Model,
        data: Tensor,
        data_spatial_dim: Dim,
        lm: str,
        lexicon: str,
        hyperparameters: dict,
        prior_file: tk.Path = None
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Uses a LM and beam search.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from torchaudio.models.decoder import ctc_decoder
    import torch
    from returnn.util.basic import cf
    # Get the logits from the model
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)

    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
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
    # if greedy:
    #     probs, greedy_res = torch.max(label_log_prob, dim=-1)
    #     greedy_res = greedy_res.unsqueeze(1)
    #
    #     scores = torch.sum(probs, dim=-1)
    #     scores = scores.unsqueeze(1)
    #
    #     beam_dim = rtf.TorchBackend.get_new_dim_raw(greedy_res, 1, name="beam_dim")
    #     dims = [batch_dim, beam_dim, enc_spatial_dim]
    #     hyps = rtf.TorchBackend.convert_to_tensor(greedy_res, dims=dims, sparse_dim=model.wb_target_dim, dtype="int64",
    #                                               name="hyps")
    #
    #     dims = [batch_dim, beam_dim]
    #     scores = Tensor("scores", dims=dims, dtype="float32", raw_tensor=scores)
    #
    #     return hyps, scores, enc_spatial_dim, beam_dim
    use_lm = hyp_params.pop("use_lm", False)
    import pdb
    #pdb.set_trace()
    if use_lm:
        if lm: # Directly give path only for count based n-gram(arpa)
            lm = str(cf(lm))
            # Other types distinguish with the name
        else: # extend to elif as adding other LM type
            assert lm_name.startswith("ffnn")
            assert model.recog_language_model
            assert isinstance(model.recog_language_model, FeedForwardLm)
            assert model.recog_language_model.vocab_dim == model.target_dim
            def extract_ctx_size(s):
                match = re.match(r"ffnn(\d+)_\d+", s)  # Extract digits after "ffnn" before "_"
                return match.group(1) if match else None
            context_size = int(extract_ctx_size(lm_name))
            #context_size = int(lm_name[len("ffnn"):])
            lm = FFNN_LM_flashlight(model.recog_language_model, model.recog_language_model.vocab_dim, context_size)
    else:
        lm = None

    use_lexicon = hyp_params.pop("use_lexicon", True)

    configs = {
        "tokens": list(model.wb_target_dim.vocab.labels),
        "blank_token": OUT_BLANK_LABEL,
        "sil_token": OUT_BLANK_LABEL,
        "unk_word": "<unk>",
        "beam_size_token": None,  # 16
        "beam_threshold": 50,  # 14. 1000000
    }
    configs["lexicon"] = lexicon if use_lexicon else None
    configs["lm"] = lm

    configs.update(hyp_params)

    decoder = ctc_decoder(**configs)
    enc_spatial_dim_torch = enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()
    if use_logsoftmax:
        decoder_results = decoder(label_log_prob, enc_spatial_dim_torch)
    else:
        decoder_results = decoder(logits.raw_tensor.cpu(), enc_spatial_dim_torch)
    # #--------------------------------------test---------------------------------------------------------

    #
    # def parse_lexicon_file(file_path):
    #     parsed_dict = {}
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         for line in file:
    #             parts = line.strip().split(maxsplit=1)  # Split into two parts: word and pieces
    #             if len(parts) == 2:  # Ensure there's a value after the key
    #                 word, pieces = parts
    #                 parsed_dict[pieces] = word
    #     return parsed_dict
    # if use_lexicon:
    #     lexicon_dict = parse_lexicon_file(configs["lexicon"])
    ctc_scores = [[l2.score for l2 in l1] for l1 in decoder_results]
    ctc_scores = torch.tensor(ctc_scores)
    pdb.set_trace()
    # ctc_scores_forced = []
    # sim_scores = []
    # lm_scores = []
    # ctc_losses = []
    # sentences = []
    # unmatch_idxs = []
    # ctc_loss = torch.nn.CTCLoss(model.blank_idx, "none")
    # '''Parrallezing viterbi across batch'''
    # from concurrent.futures import ProcessPoolExecutor
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)
    # from functools import partial
    # seq_list = [res[0].tokens.cpu() for res in decoder_results]
    # log_prob_list = list(label_log_prob.cpu())
    # spatial_dim_list = list(enc_spatial_dim_torch.cpu())
    # viterbi_batch_partial = partial(viterbi_batch, blank_idx=model.blank_idx)
    # #cpu_cores = multiprocessing.cpu_count()
    # print(f"using {min(32,label_log_prob.shape[0])} workers")
    # import pdb  # ---------
    # #pdb.set_trace()
    # with ProcessPoolExecutor(max_workers=min(32,label_log_prob.shape[0])) as executor:
    #     alignments, viterbi_scores = zip(*list(executor.map(viterbi_batch_partial, zip(seq_list, log_prob_list, spatial_dim_list))))
    #
    # for i in range(label_log_prob.shape[0]):
    #     seq = decoder_results[i][0].tokens # These are not padded
    #     log_prob = label_log_prob[i] # These are padded
    #     # alignment, viterbi_score = ctc_viterbi_one_seq(log_prob, seq, int(enc_spatial_dim_torch[i].item()), # int(enc_spatial_dim_torch.max())
    #     #                        blank_idx=model.blank_idx)
    #     alignment = alignments[i]
    #     viterbi_score = viterbi_scores[i]
    #     collapsed = ctc_collapse(alignment)
    #     if use_lexicon:
    #         def merge_tokens(token_list):
    #             # Merge bpe tokens according to lexicon
    #             merged_string = ""
    #             buffer = ""
    #             for token in token_list:
    #                 if token.endswith("@@"):
    #                     buffer += token + " "
    #                 else:
    #                     buffer += token
    #                     '''How does the ctcdecoder handle the OOV?'''
    #                     assert buffer in lexicon_dict.keys(), buffer + f" not in the lexicon!\n token list: {token_list} \n seq from search: {decoder.idxs_to_tokens(seq)}"
    #                     merged_string +=  lexicon_dict[buffer] + " "# Append buffer and curr
    #                     # ent token
    #                     buffer = ""  # Reset buffer
    #             return merged_string.strip()
    #         #sentence_from_viterbi = merge_tokens(decoder.idxs_to_tokens(collapsed))
    #     #if i == 3: #----test
    #         #pdb.set_trace()
    #     '''Forced alignment and feed to the same decoder'''
    #     alignment = torch.cat((alignment, torch.tensor([0 for _ in range(log_prob.shape[0] - alignment.shape[0])])))
    #     mask = torch.arange(model.target_dim.size + 1, device=log_prob.device).unsqueeze(0).expand(log_prob.shape) == alignment.unsqueeze(1)
    #     decoder_result = decoder(log_prob.masked_fill(~mask, float('-inf')).unsqueeze(0), enc_spatial_dim_torch[i].unsqueeze(0))
    #     ctc_scores_forced.append(decoder_result[0][0].score)
    #
    #     sentence = " ".join(list(decoder_results[i][0].words))
    #     word_seq = [decoder.word_dict.get_index(word) for word in decoder_results[i][0].words]
    #     lm_score = CTClm_score(word_seq, decoder)
    #     sentences.append(sentence)
    #     lm_scores.append(lm_score)
    #     sim_score = viterbi_score + hyp_params["lm_weight"]*lm_score
    #
    #     assert sim_score > ctc_scores[i] or abs(sim_score-ctc_scores[i]) < 1e-01
    #
    #     sim_scores.append(sim_score)
    #     ctc_losses.append(ctc_loss(log_prob, seq, [log_prob.shape[0]],
    #                            [seq.shape[0]]))
    #     '''Check if output alignment give by viterbi matches the input sequence'''
    #     assert (collapsed.tolist() == seq.tolist()), (f"Viterbi did not give path collapsed to the original sequence, idx: {i}!"
    #                                                   f"\n ctc_scores_forced: {torch.tensor(ctc_scores_forced).tolist()}"
    #                                                   f"\n sim_scores: {torch.tensor(sim_scores).tolist()} "
    #                                                   f"\n ctc_scores: {torch.tensor(ctc_scores).tolist()}")
    #     '''Check if output alignment give by viterbi matches the sequence output from decoder with masked emission'''
    #     if not collapsed.tolist() == decoder_result[0][0].tokens.tolist():
    #         unmatch_idxs.append(i)
    #         print(f"Viterbi did not give path collapsed to the same sequence as forced decoder, idx: {i}!")
    #     # assert (collapsed.tolist() == decoder_result[0][0].tokens.tolist()), (
    #     #     f"Viterbi did not give path collapsed to the same sequence as forced decoder at position{i}!"
    #     #     f"\n ctc_scores_forced: {torch.tensor(ctc_scores_forced).tolist()}"
    #     #     f"\n sim_scores: {torch.tensor(sim_scores).tolist()} "
    #     #     f"\n ctc_scores: {torch.tensor(ctc_scores).tolist()}")
    #     #pdb.set_trace()
    # print(f"\n ctc_scores_forced: {torch.tensor(ctc_scores_forced).tolist()}"
    #       f"\n sim_scores: {torch.tensor(sim_scores).tolist()} "
    #       f"\n ctc_scores: {torch.tensor(ctc_scores).tolist()}"
    #       f"\n unmatch_idxs: {unmatch_idxs}")
    # #pdb.set_trace()
    # print(f"Average difference of ctc_decoder score and viterbi score: {abs(np.mean(np.array(ctc_scores[:,0])-sim_scores))}")
    # if not use_lexicon:
    #     assert abs(np.mean(np.array(ctc_scores[:,0])-sim_scores)) < 1e-05
    #
    # assert scores.raw_tensor[0,:] - ctc_viterbi_one_seq(label_log_prob[0], decoder_results[0][0].tokens, int(enc_spatial_dim_torch.max()),
    #                            blank_idx=model.blank_idx) < tolerance, "CTCdecoder does use viterbi decoding!"
    # # -----------------------------------------------------------------------------------------------
    if use_lexicon:
        print("Use words directly!")
        if CHECK_DECODER_CONSISTENCY:
            for l1 in decoder_results:
                for l2 in l1:
                    lexicon_words = " ".join(l2.words)
                    token_words = " ".join([configs["tokens"][t] for t in l2.tokens])
                    assert not token_words.endswith(
                        "@@"), f"Token words ends with @@: {token_words}, Lexicon words: {lexicon_words}"
                    token_words = token_words.replace("@@ ", "")
                    assert lexicon_words == token_words, f"Words don't match: Lexicon words: {lexicon_words}, Token words: {token_words}"

        words = [[" ".join(l2.words) for l2 in l1] for l1 in decoder_results]

        words = np.array(words)
        words = np.expand_dims(words, axis=2)
        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        scores = torch.tensor(scores)

        beam_dim = Dim(words.shape[1], name="beam_dim")
        enc_spatial_dim = Dim(1, name="spatial_dim")
        words = rf._numpy_backend.NumpyBackend.convert_to_tensor(words, dims=[batch_dim, beam_dim, enc_spatial_dim],
                                                                 dtype="string", name="hyps")
        scores = Tensor("scores", dims=[batch_dim, beam_dim], dtype="float32", raw_tensor=scores)

        return words, scores, enc_spatial_dim, beam_dim
    else:
        def _pad_blanks(tokens, max_len):
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
        hyps = rtf.TorchBackend.convert_to_tensor(hyps, dims=dims, sparse_dim=model.wb_target_dim, dtype="int64",
                                                  name="hyps")

        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        max_length_3 = max([len(l) for l in scores])
        scores = torch.stack([_pad_scores(l, max_length_3) for l in scores])
        dims = [batch_dim, beam_dim]
        scores = Tensor("scores", dims=dims, dtype="float32", raw_tensor=scores)

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
        prior_file: tk.Path = None
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

    return decode_flashlight(model=model, label_log_prob=label_log_prob, enc_spatial_dim=enc_spatial_dim,
                             hyperparameters=hyperparameters, prior_file=prior_file)


# RecogDef API
model_recog_flashlight: RecogDef[Model]
model_recog_flashlight.output_with_beam = True
model_recog_flashlight.output_blank_label = OUT_BLANK_LABEL
model_recog_flashlight.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def decode_flashlight(
        *,
        model: Model,
        label_log_prob: Tensor,
        enc_spatial_dim: Dim,
        hyperparameters: dict,
        prior_file: tk.Path = None,
        train_lm=False
) -> Tuple[Tensor, Tensor, Dim, Dim] | list:
    from dataclasses import dataclass
    import torch
    import json
    from flashlight.lib.text.decoder import LM, LMState
    from i6_experiments.users.zeyer.utils.lru_cache import lru_cache
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
    def extract_ctx_size(s):
        match = re.match(r"ffnn(\d+)_\d+", s)  # Extract digits after "ffnn" before "_"
        return match.group(1) if match else None
    context_size = int(extract_ctx_size(lm_name))
    if train_lm:
        assert model.train_language_model
        assert model.train_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.train_language_model
    else:
        assert lm_name.startswith("ffnn")
        assert model.recog_language_model
        assert isinstance(model.recog_language_model, FeedForwardLm)
        assert model.recog_language_model.vocab_dim == model.target_dim

        # context_size = int(lm_name[len("ffnn"):])
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

    print(
        f"Memory usage {dev_s} before encoder forward:",
        " ".join(_collect_mem_stats()),
        "total:",
        util.human_bytes_size(total_mem) if total_mem else "(unknown)",
    )

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
                    lm_logits = rf.gather(lm_logits, axis=out_spatial_dim,
                                          indices=rf.last_frame_position_of_dim(out_spatial_dim))
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
            self.mapping_states[state] = FlashlightLMState(label_seq=[model.bos_idx] * context_size, prev_state=state)
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

    print(f"Memory usage {dev_s} after encoder forward:", " ".join(_collect_mem_stats()))

    def _output_hyps(hyp: list) -> str:
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
                    scores_per_batch += [-1e30] * (n_best - len(hyps_per_batch))
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
    # Test------------------------
    #pdb.set_trace()
    #model.blank_idx
    #-----------------------
    # import torch.nn.functional as F
    # collapsed_hyps = [[ctc_collapse(hyp, model.blank_idx) for hyp in seq] for seq in hyps_pt]
    # enc_spatial_dim_torch = enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()
    # padded_hypotheses = [
    #     [F.pad(hyp, (0, int(enc_spatial_dim_torch.max()) - hyp.shape[0]), value=0) for hyp in batch]
    #     for batch in collapsed_hyps
    # ]
    # hyps_pt = torch.stack([torch.stack(batch) for batch in padded_hypotheses])
    # Test------------------------
    #pdb.set_trace()
    #model.blank_idx
    #-----------------------
    beam_dim = Dim(n_best, name="beam")
    out_spatial_dim = enc_spatial_dim
    hyps_r = rf.convert_to_tensor(hyps_pt, dims=(batch_dim, beam_dim, out_spatial_dim), sparse_dim=model.wb_target_dim)
    scores_r = rf.convert_to_tensor(scores_pt, dims=(batch_dim, beam_dim))
    print(f"Memory usage ({dev_s}) after batch:", " ".join(_collect_mem_stats()))
    if train_lm:
        return hyps
    else:
        return hyps_r, scores_r, out_spatial_dim, beam_dim


def ctc_viterbi_one_seq(ctc_log_probs, seq, t_max, blank_idx):
    import torch
    mod_len = 2 * seq.shape[0] + 1
    mod_seq = torch.stack([seq, torch.full(seq.shape, blank_idx,device=seq.device)], dim=1).flatten()
    mod_seq = torch.cat((torch.tensor([blank_idx], device=mod_seq.device), mod_seq))
    V = torch.full((t_max, mod_len), float("-inf"))  # [T, 2S+1]
    V[0, 0] = ctc_log_probs[0, blank_idx]
    V[0, 1] = ctc_log_probs[0, seq[0]]

    backref = torch.full((t_max, mod_len), -1, dtype=torch.int64, device="cuda")

    for t in range(1, t_max):
        for s in range(mod_len):
            if s > 2 * t + 1:
                continue
            skip = False
            # if s % 2 != 0 and s >= 3:
            #     idx = (s - 1) // 2
            #     prev_idx = (s - 3) // 2
            #     if seq[idx] != seq[prev_idx]:
            #         skip = True
            if s % 2 != 0 and s >= 3:
                if mod_seq[s] != mod_seq[s-2]:
                    skip = True
            if skip:
                V[t, s] = max(V[t - 1, s], V[t - 1, s - 1], V[t - 1, s - 2]) + ctc_log_probs[t, mod_seq[s]]
                backref[t, s] = torch.argmax(torch.tensor([V[t - 1, s], V[t - 1, s - 1], V[t - 1, s - 2]]))
            else:
                V[t, s] = max(V[t - 1, s], V[t - 1, s - 1]) + ctc_log_probs[t, mod_seq[s]]
                backref[t, s] = torch.argmax(torch.tensor([V[t - 1, s], V[t - 1, s - 1]]))

    score = torch.max(V[t_max - 1, :])
    idx = torch.argmax(V[t_max - 1, :])
    res = [mod_seq[idx]]
    import pdb  # ---------
    #pdb.set_trace()
    for t in range(t_max - 1, 0, -1):
        next_idx = idx - backref[t, idx]
        res.append(mod_seq[next_idx])
        idx = next_idx

    res = torch.tensor(res).flip(0)
    return res, score

# TODO: dynmically get the pad_index
def trim_padded_sequence(sequence: torch.Tensor, pad_index: int = 0) -> torch.Tensor:
    """
    Removes trailing pad_index elements from a padded 1D sequence tensor.

    Args:
        sequence (torch.Tensor): 1D tensor containing the sequence with padding.
        pad_index (int): The padding index used in the sequence.

    Returns:
        torch.Tensor: The trimmed sequence without trailing padding.
    """
    # Find indices where the sequence is not the pad_index.
    non_pad_indices = (sequence != pad_index).nonzero(as_tuple=True)[0]

    if non_pad_indices.numel() == 0:
        # If the entire sequence is padding, return an empty tensor.
        return sequence.new_empty(0)

    # The last non-padding index; add 1 for slicing (because slicing is exclusive).
    last_index = non_pad_indices[-1].item() + 1

    return sequence[:last_index]

def viterbi_batch(inputs, blank_idx):
    '''
    inputs: (targets.raw_tensor[i, :], label_log_prob[i])
    '''
    target, log_prob, spatial_dim = inputs
    seq = trim_padded_sequence(target)
    return ctc_viterbi_one_seq(log_prob, seq, int(spatial_dim.item()),  # int(enc_spatial_dim_torch.max())
                        blank_idx=blank_idx)

def CTClm_score(seq, lm):
    """
    When lm is word level, sequence need to be converted to word...?
    """
    state = lm.start(False)
    score = 0
    for token in seq:
        state, cur_score = lm.score(state, token)
        score += cur_score
    score += lm.finish(state)[1] #maybe not
    return score

def ctc_collapse(ctc_sequence, blank_token):
    """
    Collapses a CTC decoded tensor by removing consecutive duplicates and blank tokens.

    Args:
        ctc_sequence (list or tensor): Decoded CTC output sequence (list of indices).
        blank_token (int): Index representing the blank token in the sequence.

    Returns:
        Tensor: Collapsed sequence without repeated characters and blanks.
    """
    import torch
    collapsed_sequence = []
    prev_token = None

    for token in ctc_sequence:
        if token != prev_token and token != blank_token:  # Remove repetition and blank
            collapsed_sequence.append(token)
        prev_token = token  # Update previous token

    return torch.tensor(collapsed_sequence)

def scoring(
        *,
        model: Model,
        data: Tensor,
        targets: Tensor,
        data_spatial_dim: Dim,
        lm: str,
        lexicon: str,
        hyperparameters: dict,
        prior_file: tk.Path = None
) -> Tensor:
    """
    Function is run within RETURNN.

    Uses a LM interpolation and prior correction.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from torchaudio.models.decoder import ctc_decoder
    import torch
    from returnn.util.basic import cf
    # Get the logits from the model
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)

    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
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

    use_lm = hyp_params.pop("use_lm", False)

    def extract_ctx_size(s):
        match = re.match(r"ffnn(\d+)_\d+", s)  # Extract digits after "ffnn" before "_"
        return match.group(1) if match else None

    context_size = int(extract_ctx_size(lm_name))

    if use_lm:
        if lm: # Directly give path only for count based n-gram(arpa)
            lm = str(cf(lm))
            # Other types distinguish with the name
        else: # extend to elif as adding other LM type
            assert lm_name.startswith("ffnn")
            assert model.recog_language_model
            assert isinstance(model.recog_language_model,FeedForwardLm)
            assert model.recog_language_model.vocab_dim == model.target_dim

            raw_lm: FeedForwardLm = model.recog_language_model
            #context_size = int(lm_name[len("ffnn"):])
            #lm = FFNN_LM_flashlight(model.recog_language_model, model.recog_language_model.vocab_dim, context_size)
            # --------------------Flashlight LM Test-------------------------------------
            from flashlight.lib.text.decoder import LM, LMState
            from dataclasses import dataclass
            from i6_experiments.users.zeyer.utils.lru_cache import lru_cache
            from returnn.util import basic as util

            lm_initial_state = raw_lm.default_initial_state(batch_dims=[])

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

            print(
                f"Memory usage {dev_s} before encoder forward:",
                " ".join(_collect_mem_stats()),
                "total:",
                util.human_bytes_size(total_mem) if total_mem else "(unknown)",
            )
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
                            lm_logits, lm_state = raw_lm(
                                rf.convert_to_tensor(state_.label_seq, dims=[spatial_dim], sparse_dim=model.target_dim),
                                spatial_dim=spatial_dim,
                                out_spatial_dim=out_spatial_dim,
                                state=lm_initial_state,
                            )  # Vocab / ...
                            lm_logits = rf.gather(lm_logits, axis=out_spatial_dim,
                                                  indices=rf.last_frame_position_of_dim(out_spatial_dim))
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
                    self.mapping_states[state] = FlashlightLMState(label_seq=[model.bos_idx] * context_size,
                                                                   prev_state=state)
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

            lm = FlashlightLM()
            # ---------------------------------------------------------------------------

    else:
        lm = None

    use_lexicon = hyp_params.pop("use_lexicon", True)

    configs = {
        "tokens": list(model.wb_target_dim.vocab.labels),
        "blank_token": OUT_BLANK_LABEL,
        "sil_token": OUT_BLANK_LABEL,
        "unk_word": "<unk>",
        "beam_size_token": None,  # 16
        "beam_threshold": 50,  # 14. 1000000
    }
    configs["lexicon"] = lexicon if use_lexicon else None
    configs["lm"] = lm

    configs.update(hyp_params)

    # decoder = ctc_decoder(**configs)
    # from flashlight.lib.text.decoder import LexiconFreeDecoderOptions, LexiconFreeDecoder, CriterionType
    # configs.update({"sil_score": 0.0, "criterion_type":CriterionType.CTC})
    # print(configs)
    # fl_decoder_opts = LexiconFreeDecoderOptions(**configs)
    # #     beam_size=configs["beam_size"],
    # #     beam_size_token=configs["beam_size_token"],
    # #     beam_threshold=configs["beam_threshold"],
    # #     lm_weight=configs["lm_weight"],
    # #     sil_score=0.0,
    # #     log_add=configs["log_add"],
    # #     criterion_type=CriterionType.CTC,
    # # )
    # decoder = LexiconFreeDecoder(fl_decoder_opts, lm, -1, model.blank_idx, [])

    enc_spatial_dim_torch = enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()
    scores = []
    # `````````````````test```````````````````````

    #ctc_scores = [[l2.score for l2 in l1] for l1 in decoder_results]
    # ctc_scores = torch.tensor(ctc_scores)
    # TODO: add ctc_loss for the case CTCdecoder use sum
    # ctc_loss = torch.nn.CTCLoss(model.blank_idx, "none")

    import pdb
    #pdb.set_trace()
    '''Parrallezing viterbi across batch'''
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    from functools import partial
    target_list = list(targets.raw_tensor.cpu())
    log_prob_list = list(label_log_prob.cpu())
    spatial_dim_list = list(enc_spatial_dim_torch.cpu())
    viterbi_batch_partial = partial(viterbi_batch, blank_idx=model.blank_idx)
    #cpu_cores = multiprocessing.cpu_count()
    print(f"using {min(32,label_log_prob.shape[0])} workers")
    with ProcessPoolExecutor(max_workers=min(32,label_log_prob.shape[0])) as executor:
        alignments, viterbi_scores = zip(*list(executor.map(viterbi_batch_partial, zip(target_list, log_prob_list, spatial_dim_list))))
    def ctc_collapse(ctc_sequence, blank_token=model.blank_idx):
        """
        Collapses a CTC decoded tensor by removing consecutive duplicates and blank tokens.

        Args:
            ctc_sequence (list or tensor): Decoded CTC output sequence (list of indices).
            blank_token (int): Index representing the blank token in the sequence.

        Returns:
            Tensor: Collapsed sequence without repeated characters and blanks.
        """
        collapsed_sequence = []
        prev_token = None

        for token in ctc_sequence:
            if token != prev_token and token != blank_token:  # Remove repetition and blank
                collapsed_sequence.append(token)
            prev_token = token  # Update previous token

        return torch.tensor(collapsed_sequence)

    def parse_lexicon_file(file_path):
        parsed_dict = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(maxsplit=1)  # Split into two parts: word and pieces
                if len(parts) == 2:  # Ensure there's a value after the key
                    word, pieces = parts
                    parsed_dict[pieces] = word
        return parsed_dict
    if use_lexicon:
        lexicon_dict = parse_lexicon_file(configs["lexicon"])
    # ctc_scores_forced = []
    # sim_scores = []
    # lm_scores = []
    # ctc_losses = []
    # ground_truths = []
    # unmatch_idxs = []
    # ctc_loss = torch.nn.CTCLoss(model.blank_idx, "none")
    if use_lm:
        #################################3
        # for i in range(label_log_prob.shape[0]):
        #     seq = trim_padded_sequence(targets.raw_tensor[i,:])  # These are padded
        #     log_prob = label_log_prob[i]  # These are padded
        #     alignment = alignments[i]
        #     collapsed = ctc_collapse(alignment)
        #     viterbi_score = viterbi_scores[i]
        #     if use_lexicon:
        #         def merge_tokens(token_list):
        #             # Merge bpe tokens according to lexicon
        #             merged_string = ""
        #             buffer = ""
        #             for token in token_list:
        #                 if token.endswith("@@"):
        #                     buffer += token + " "
        #                 else:
        #                     buffer += token
        #                     assert buffer in lexicon_dict.keys(), buffer + f" not in the lexicon!\n token list: {token_list} \n seq of input: {decoder.idxs_to_tokens(seq)}"
        #                     merged_string += lexicon_dict[buffer] + " "  # Append buffer and curr
        #                     # ent token
        #                     buffer = ""  # Reset buffer
        #             return merged_string.strip()
        #
        #         #sentence_from_viterbi = merge_tokens(decoder.idxs_to_tokens(collapsed))
        #     # alignment = torch.cat((alignment, torch.tensor([0 for _ in range(log_prob.shape[0] - alignment.shape[0])])))
        #     # mask = torch.arange(model.target_dim.size + 1, device=log_prob.device).unsqueeze(0).expand(
        #     #     log_prob.shape) == alignment.unsqueeze(1)
        #     # decoder_result = decoder(log_prob.masked_fill(~mask, float('-inf')).unsqueeze(0),
        #     #                          enc_spatial_dim_torch[i].unsqueeze(0))
        #     # ctc_scores_forced.append(decoder_result[0][0].score)
        #     sentence = " ".join([model.target_dim.vocab.id_to_label(h) for h in seq if h != model.blank_idx]).replace("@@ ","")
        #     word_seq = [decoder.word_dict.get_index(word) for word in sentence.split()]
        #     lm_score = CTClm_score(word_seq, lm)
        #     ground_truths.append(" ".join(sentence))
        #     lm_scores.append(lm_score)
        #     sim_scores.append(viterbi_score + hyp_params["lm_weight"] * lm_score)
            # ctc_losses.append(ctc_loss(log_prob, seq, [log_prob.shape[0]],
            #                            [seq.shape[0]]))
            # '''Check if output alignment give by viterbi matches the input sequence'''
            # assert (collapsed.tolist() == seq.tolist()), (
            #     f"Viterbi did not give path collapsed to the original sequence, idx: {i}!"
            #     f"\n ctc_scores_forced: {torch.tensor(ctc_scores_forced).tolist()}"
            #     f"\n sim_scores: {torch.tensor(sim_scores).tolist()} ")
            # '''Check if output alignment give by viterbi matches the sequence output from decoder with masked emission'''
            # if not collapsed.tolist() == decoder_result[0][0].tokens.tolist():
            #     unmatch_idxs.append(i)
            #     print(f"Viterbi did not give path collapsed to the same sequence as forced decoder, idx: {i}!")
            # # assert (collapsed.tolist() == decoder_result[0][0].tokens.tolist()), (
            # #     f"Viterbi did not give path collapsed to the same sequence as forced decoder at position{i}!"
            # #     f"\n ctc_scores_forced: {torch.tensor(ctc_scores_forced).tolist()}"
            # #     f"\n sim_scores: {torch.tensor(sim_scores).tolist()} "
            # #     f"\n ctc_scores: {torch.tensor(ctc_scores).tolist()}")
            # # pdb.set_trace()
        # print(f"\n ctc_scores_forced: {torch.tensor(ctc_scores_forced).tolist()}"
        #       f"\n sim_scores: {torch.tensor(sim_scores).tolist()} "
        #       f"\n unmatch_idxs: {unmatch_idxs}")
        #pdb.set_trace()
        ###################################
        lm_scores = []
        for i in range(label_log_prob.shape[0]):
            seq = trim_padded_sequence(targets.raw_tensor[i,:])
            target_words = " ".join([model.target_dim.vocab.id_to_label(idx) for idx in seq])
            target_words = target_words.replace("@@ ","")
            #target_words = target_words.replace(" <s>", "")
            word_seq = seq.tolist() #[decoder.word_dict.get_index(word) for word in target_words.split()] if use_lexicon else seq
            lm_score = CTClm_score(word_seq, lm)
            lm_scores.append(lm_score)
            scores.append(viterbi_scores[i] + hyp_params["lm_weight"]*lm_score)
        score_dim = Dim(1, name="score_dim")
        scores = Tensor("scores", dims=[batch_dim, score_dim], dtype="float32",
                        raw_tensor=torch.tensor(scores).reshape([label_log_prob.shape[0], 1]))

    else:
        scores = viterbi_scores
        score_dim = Dim(1, name="score_dim")
        scores = Tensor("scores", dims=[batch_dim, score_dim], dtype="float32",
                        raw_tensor=torch.tensor(scores).reshape([label_log_prob.shape[0], 1]))
    torch.cuda.empty_cache()
    #````````````````````````````````````````````

    return scores, score_dim


# RecogDef API
scoring: RecogDef[Model]
scoring.output_with_beam = True
scoring.output_blank_label = OUT_BLANK_LABEL
scoring.batch_size_dependent = False  # not totally correct, but we treat it as such...

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
        enc_build_dict: Optional[Dict[str, Any]] = None,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_input_layer: Optional[Dict[str, Any]] = None,
        enc_conformer_layer: Optional[Dict[str, Any]] = None,
        enc_other_opts: Optional[Dict[str, Any]] = None,
        recog_language_model: Optional[FeedForwardLm] = None,
    ):
        super(Model, self).__init__()

        self.in_dim = in_dim

        import numpy
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        if enc_build_dict:
            # Warning: We ignore the other args (num_enc_layers, enc_model_dim, enc_other_opts, etc).
            self.encoder = rf.build_from_dict(enc_build_dict, in_dim)
            self.encoder: ConformerEncoder  # might not be true, but assume similar/same interface

        else:
            if not enc_input_layer:
                enc_input_layer = ConformerConvSubsample(
                    in_dim,
                    out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],
                )

            enc_opts = {"input_layer": enc_input_layer, "num_layers": num_enc_layers}

            if enc_conformer_layer:
                enc_opts["encoder_layer"] = enc_conformer_layer

            enc_layer_drop = config.float("enc_layer_drop", 0.0)
            if enc_layer_drop:
                assert "sequential" not in enc_opts
                enc_opts["sequential"] = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)

            if enc_other_opts:
                for k, v in enc_other_opts.items():
                    assert k not in enc_opts, f"enc_other_opts key {k!r} already in enc_opts {enc_opts}"
                    enc_opts[k] = v

            self.encoder = ConformerEncoder(in_dim, enc_model_dim, **enc_opts)

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
            from .model_ext.disable_self_att import apply_disable_self_attention_

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
        self.blank_logit_shift = config.float("blank_logit_shift", 0.0)

        self.ctc_am_scale = config.float("ctc_am_scale", 1.0)
        self.ctc_prior_scale = config.float("ctc_prior_scale", 0.0)
        self.ctc_prior_type = config.value("ctc_prior_type", "batch")

        static_prior = config.typed_value("static_prior")
        self.static_prior = None
        if static_prior:
            assert isinstance(static_prior, dict)
            assert set(static_prior.keys()) == {"file", "type"}
            v = numpy.loadtxt(static_prior["file"])
            if static_prior["type"] == "log_prob":
                pass  # already log prob
            elif static_prior["type"] == "prob":
                v = numpy.log(v)
            else:
                raise ValueError(f"invalid static_prior type {static_prior['type']!r}")
            self.static_prior = rf.Parameter(
                rf.convert_to_tensor(v, dims=[self.wb_target_dim], dtype=rf.get_default_float_dtype()),
                auxiliary=True,
                non_critical_for_restore=True,
            )

        if target_dim.vocab and not wb_target_dim.vocab:
            from returnn.datasets.util.vocabulary import Vocabulary

            # Just assumption for code now, might extend this later.
            assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
            vocab_labels = list(target_dim.vocab.labels) + [model_recog.output_blank_label]
            wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                vocab_labels, user_defined_symbols={model_recog.output_blank_label: blank_idx}
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
                        non_critical_for_restore=True,
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
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

        self.decoder = None
        aux_attention_decoder = config.typed_value("aux_attention_decoder", None)
        if aux_attention_decoder:
            assert isinstance(aux_attention_decoder, dict)
            aux_attention_decoder = aux_attention_decoder.copy()
            aux_attention_decoder.setdefault("class", "returnn.frontend.decoder.transformer.TransformerDecoder")
            if isinstance(aux_attention_decoder.get("model_dim", None), int):
                aux_attention_decoder["model_dim"] = Dim(aux_attention_decoder["model_dim"], name="dec_model")
            self.decoder = rf.build_from_dict(
                aux_attention_decoder, encoder_dim=self.encoder.out_dim, vocab_dim=target_dim
            )

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
        self.recog_language_model = recog_language_model

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Dim]:
        """
        Encode, get CTC logits.
        Use :func:`log_probs_wb_from_logits` to get log probs
        (might be just log_softmax, but there are some other cases).

        :return: logits, enc, enc_spatial_dim
        """
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
            Also, potentially adds label smoothing on the gradients.
        """
        if not self.out_blank_separated:  # standard case, joint distrib incl blank
            if self.blank_logit_shift:
                logits += rf.sparse_to_dense(
                    self.blank_idx, label_value=self.blank_logit_shift, other_value=0, axis=self.wb_target_dim
                )
            log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)
        else:  # separate blank
            assert self.blank_idx == self.target_dim.dimension  # not implemented otherwise
            dummy_blank_feat_dim = Dim(1, name="blank_feat")
            logits_wo_blank, logits_blank = rf.split(
                logits, axis=self.wb_target_dim, out_dims=[self.target_dim, dummy_blank_feat_dim]
            )
            log_probs_wo_blank = rf.log_softmax(logits_wo_blank, axis=self.target_dim)
            log_probs_wo_blank = self._maybe_apply_on_log_probs(log_probs_wo_blank)
            if self.blank_logit_shift:
                logits_blank += self.blank_logit_shift
            log_probs_blank = rf.log_sigmoid(logits_blank)
            log_probs_emit = rf.squeeze(rf.log_sigmoid(-logits_blank), axis=dummy_blank_feat_dim)
            log_probs, _ = rf.concat(
                (log_probs_wo_blank + log_probs_emit, self.target_dim),
                (log_probs_blank, dummy_blank_feat_dim),
                out_dim=self.wb_target_dim,
            )
            log_probs.feature_dim = self.wb_target_dim
        log_probs = self._maybe_apply_on_log_probs(log_probs)
        if self.ctc_am_scale == 1 and self.ctc_prior_scale == 0:  # fast path
            return log_probs
        log_probs_am = log_probs
        log_probs = log_probs_am * self.ctc_am_scale
        if self.ctc_prior_scale:
            if self.ctc_prior_type == "batch":
                log_prob_prior = rf.reduce_logsumexp(
                    log_probs_am, axis=[dim for dim in log_probs_am.dims if dim != self.wb_target_dim]
                )
                assert log_prob_prior.dims == (self.wb_target_dim,)
            elif self.ctc_prior_type == "static":
                log_prob_prior = self.static_prior
                assert log_prob_prior.dims == (self.wb_target_dim,)
            else:
                raise ValueError(f"invalid ctc_prior_type {self.ctc_prior_type!r}")
            log_probs -= log_prob_prior * self.ctc_prior_scale
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
