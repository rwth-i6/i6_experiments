"""
Conformer encoder def
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
from i6_experiments.users.zhang.experiments.ctc import model_recog
from tools.hdf_dump_translation_dataset import UNKNOWN_LABEL

from ..configs import *
from ..configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor

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
    if tune_hyperparameters and decoding_config["use_lm"]:
        original_params = decoding_config
        params = copy.copy(original_params)
        params.pop("lm_weight_tune", None)
        params.pop("prior_weight_tune", None)
        default_lm = original_params.get("lm_weight")
        default_prior = original_params.get("prior_weight")
        lm_scores = []
        prior_scores = []
        lm_tune_ls = [-0.1,-0.05,0,0.05]#[scale/100 for scale in range(-50,51,10)] #[-0.5,-0.45....+0.45,+0.5]  [scale/100 for scale in range(-50,51,10/5)] for bpe10k/bpe128
        prior_tune_ls = [-0.05, -0.1, 0.0, 0.05, 0.1]
        for dc_lm in lm_tune_ls:
            params["lm_weight"] = default_lm + dc_lm
            task_copy = copy.deepcopy(task)
            # score = recog_training_exp(u
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
            if os.path.exists(best_lm_tune): # Just for bypassing the static check #TODO： this will not be excuted in first run
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
        recog_language_model: Optional[FeedForwardLm | TransformerDecoder] = None,
    ):
        super(Model, self).__init__()

        self.in_dim = in_dim
        self.num_enc_layers = num_enc_layers

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
