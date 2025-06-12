"""
CTC experiments.
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Optional, Callable, Union, Tuple, Sequence, Collection
import numpy as np
import re
import json

from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
from sisyphus import tk
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer, ConformerConvSubsample
from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob
from i6_experiments.users.zhang.experiments.lm.llm import get_llm

WORD_PPL = False # If convert all ppl to word level
BLSTM_Enc_dim = 1024 # Default 512, in this case leave model_config empty. Otherwise, the hash broken, dont know why
USE_flashlight_decoder = False
recog_info = "flashlight_" if USE_flashlight_decoder else "i6_"
def ctc_exp(lmname, lm, vocab, rescor_config:tuple=None, encoder:str="conformer",train:bool=False):
    """Experiments on CTC"""
    model_def = None
    if encoder == "conformer":
        # ---------------------------------------------------------
        # model name: f"v6-relPosAttDef-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100"
        #               f"-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2"
        #               f"-[vocab]" + f"-[sample]"
        # relPosAttDef: Use the default RelPosSelfAttention instead of the Shawn et al 2018 style, old RETURNN way.
        enc_conformer_layer_default = rf.build_dict(
            ConformerEncoderLayer,
            ff_activation=rf.build_dict(rf.relu_square),
            num_heads=8,
        )
        model_config = {"enc_conformer_layer": enc_conformer_layer_default, "feature_batch_norm": True}
        # from i6_experiments.users.zhang.experiments.encoder.conformer import ctc_model_def as conformer_model_def
        # model_def = conformer_model_def
    elif encoder == "blstm":
        from i6_experiments.users.zhang.experiments.encoder.blstm import ctc_model_def as blstm_model_def
        model_def = blstm_model_def #TODO: not sure how would this break the hash.
        # It breaks when move the def to a separate file
        if BLSTM_Enc_dim == 512:
            model_config = {}#{"enc_dim":BLSTM_Enc_dim}
        else:
            model_config = {"enc_dim":BLSTM_Enc_dim}#
    else:
        raise ValueError(f"Unknown encoder: {encoder}")


    recog_epoch = None
    if vocab == "bpe128":
        exclude_epochs = set(range(0, 501)) - set([477])#Reduce #ForwardJobs, 477 is the best epoch for most cases.
        recog_epoch = 500 if encoder == "blstm" else 477 #477 for Conformer
    if vocab == "bpe10k":
        exclude_epochs = set(range(0, 500))
        recog_epoch = 500

    tune_hyperparameters = False

    #
    recog_config_updates = {}
    tune_config_updates = {}
    #Use to make some search config difference for tuning and recog(after tuning), default: same

    with_prior = True
    prior_from_max = False # ! arg in recog.compute_prior, actually Unused
    empirical_prior = False
    #-------------------setting decoding config-------------------
    # Each time called ctc_exp there will be an independent copy of this config, TODO: declare this outside the function and use explicit copy
    decoding_config = {
        "log_add": False,
        "nbest": 50,
        "beam_size": 80, # 80 for previous exps on bpe128, this is also default beamsize for NoLM
        "beam_threshold": 1e6,  # 14. 1000000
        "lm_weight": 1.45,  # Tuned_best val: 1.45 NOTE: weights are exponentials of the probs.
        "use_logsoftmax": True,
        "use_lm": False,
        "use_lexicon": False, # Do open vocab search when using bpe lms.
    } \
    #     if USE_flashlight_decoder else {
    #     "nbest": 1,
    #     "beam_size": 80,
    #     "lm_weight": 1.45,
    #     "use_recombination": False,
    # } #TODO this somehow makes the scoring func broken


    if lmname != "NoLM" and not train:
        decoding_config["lm_order"] = lmname
        if lmname[0].isdigit(): # n-gram
            decoding_config["lm"] = lm
        else: # in var lm Should be a config
            model_config["recog_language_model"] = lm

        decoding_config["use_lm"] = True

    tune_config_updates["beam_size"] = 30 if tune_hyperparameters else None


    if re.match(r".*word.*", lmname): # Why  or "NoLM"  in lmname?
        decoding_config["use_lexicon"] = True
    else:
        decoding_config["use_lexicon"] = False
    if with_prior:
        decoding_config["prior_weight"] = 0.15  # 0.15 as initial ref for tuning or if not using emprirical prior


    from i6_experiments.users.zhang.experiments.ctc import train_exp as train_exp, model_recog, model_recog_lm, model_recog_flashlight, config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4, _get_cfg_lrlin_oclr_by_bs_nep, speed_pert_librosa_config, _raw_sample_rate
    search_mem_rqmt = 16 if vocab == "bpe10k"  else 6
    search_rqmt = {"cpu": search_mem_rqmt // 4, "time": 24} if vocab == "bpe10k" else dict()

    batch_size = None  # Use default
    # TODO: decide if total separate or keep a default setting
    if "ffnn" in lmname:
        tune_hyperparameters = True
        search_mem_rqmt = 12
        decoding_config["beam_size"] = {"bpe128":500, "bpe10k":150}[vocab]
        tune_config_updates["beam_size"] = 50 if tune_hyperparameters else None
        search_rqmt.update({"time": decoding_config["beam_size"]//3 + 4} if USE_flashlight_decoder else {})

        decoding_config["lm_weight"] = 0.5 #for ffnn128 on conformer, 0.5 was best in range 0.5-1.5
        tune_config_updates["tune_range"] = [scale/100 for scale in range(-20,21,5)]

        decoding_config["prior_weight"] = 0.15 # Seems okay for blstm512 and conformer...?
        tune_config_updates["priro_tune_range"] = [-0.1, -0.05, 0.0, 0.05, 0.1] # TODO: !! this 0 instead 0.0 might cause some typing and hence hash issue

    elif "trafo" in lmname: #TODO No yet available for bpe10k
        tune_hyperparameters = True
        search_mem_rqmt = 12
        tune_config_updates["beam_size"] = 15 if tune_hyperparameters else None

        decoding_config["lm_weight"] = 0.5 # 0.5 was best in 0.8 +- 0.5, tuned with 5 beam / init 0.35
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-15, 16, 5)]

        decoding_config["prior_weight"] = 0.15
        tune_config_updates["priro_tune_range"] = [-0.1, -0.05, 0.0, 0.05, 0.1]

        recog_config_updates["beam_size"] = {"blstm":300, "conformer":80}[encoder] # Actually redundant
        decoding_config["beam_size"] = recog_config_updates["beam_size"]
        search_rqmt.update({"time": decoding_config["beam_size"]*5 + 5} if USE_flashlight_decoder else {})

    elif "gram" in lmname and "word" not in lmname:
        tune_hyperparameters = False
        decoding_config["lm_weight"] = 0.5
        decoding_config["beam_size"] = 600

        tune_config_updates["beam_size"] = 100 if tune_hyperparameters else None

        #decoding_config["lm_weight"] = 1.0
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-30, 31, 15)]

        decoding_config["prior_weight"] = 0.15
        tune_config_updates["priro_tune_range"] = [-0.1, -0.05, 0.0, 0.05, 0.1]

        batch_size = 64_000_000 if decoding_config["beam_size"] >= 300 else None
        search_rqmt.update({"gpu_mem": 24 if batch_size*decoding_config["beam_size"] > 36_400_000_000 else 10, "time":12 if decoding_config["beam_size"] > 100 else 4, "mem": decoding_config["beam_size"]//100 + 6})

        #tune_config_updates["tune_range"] = [scale / 100 for scale in range(-15, 16, 5)]

    elif "NoLM" in lmname:
        decoding_config["beam_size"] = 80

    if vocab == "bpe10k" or "trafo" in lmname:
        if USE_flashlight_decoder:
            batch_size = 20_000_000 if decoding_config["beam_size"] < 20 else 60_000_000
            search_rqmt.update({"gpu_mem": 24 if decoding_config["beam_size"] < 20 else 48})
        elif "trafo" in lmname:
            batch_size = {"blstm": 1_800_000, "conformer": 1_500_000}[encoder] if decoding_config["beam_size"] > 50 \
                else {"blstm": 6_400_000, "conformer": 4_800_000}[encoder]
            search_rqmt.update({"gpu_mem": 32} if batch_size*decoding_config["beam_size"] <= 80_000_000 else {"gpu_mem": 32})
            if decoding_config["beam_size"] > 150:
                batch_size = {"blstm": 1_000_000, "conformer": 800_000}[encoder]
            if decoding_config["beam_size"] >= 280:
                batch_size = {"blstm": 800_000, "conformer": 500_000}[encoder]
    print(f"\n\nlm_name:{lmname}\nbatch_size{batch_size}")

    p0 = f"_p{str(decoding_config['prior_weight']).replace('.', '')}" + (
            "-emp" if empirical_prior else ("-from_max" if prior_from_max else "")) if with_prior else ""
    p1 = "sum" if decoding_config.get('log_add') else "max"
    p2 = f"n{decoding_config['nbest']}"
    p3 = f"b{decoding_config['beam_size']}t{tune_config_updates['beam_size']}" if tune_config_updates.get('beam_size') else f"b{decoding_config['beam_size']}"
    p3 = "" if "NoLM" in lmname else p3
    p3_ = f"trsh{decoding_config['beam_threshold']:.0e}".replace("+0", "").replace("+", "") if decoding_config.get('beam_threshold') else "trsh50"
    p3_ = "" if not USE_flashlight_decoder else p3_
    p4 = f"w{str(decoding_config['lm_weight']).replace('.', '')}" if lm else ""
    p5 = "_logsoftmax" if decoding_config.get('use_logsoftmax') else ""
    p6 = "_lexicon" if decoding_config.get('use_lexicon') else ""
    lm_hyperparamters_str = f"{p0}_{p1}_{p2}_{p3}_{p3_}{p4}{p5}{p6}"

    lm_hyperparamters_str = vocab + lm_hyperparamters_str  # Assume only experiment on one ASR model, so the difference of model itself is not reflected here

    alias_name = (f"ctc-baseline_" + ("flashlight_" if "gram" in lmname else recog_info)
                  + encoder + (f"{BLSTM_Enc_dim}" if encoder =="blstm" else "") + (("_decodingWith_" + lm_hyperparamters_str + lmname
                                               if lm else f"ctc-baseline-" + lm_hyperparamters_str + lmname) if not train else f"{vocab}-"))
    print(f"name{lmname}", "lexicon:" + str(decoding_config["use_lexicon"]))
    ################################
    from .ctc import recog_nn #, recog_trafo
    model_recog_noLM = recog_nn if not USE_flashlight_decoder else model_recog
    if USE_flashlight_decoder:
        if "NoLM" in lmname:
            recog_def = model_recog_noLM
        else:
            recog_def = model_recog_flashlight if "ffnn" in lmname or "trafo" in lmname else model_recog_lm
    else:
        if "ffnn" in lmname or "trafo" in lmname:
            recog_def = recog_nn
        elif "NoLM" in lmname:
            recog_def = model_recog_noLM
        else:
            assert "gram" in lmname, ValueError(f"Generic API for i6 search is to be done for {lmname}")

    if "gram" in lmname: # Currently only flashlight impl supports count based gram lm
        recog_def = model_recog_lm

    if rescor_config:
        decoding_config["rescoring"] = True
        decoding_config["lm_rescore"] = rescor_config
    return *train_exp(
            name=alias_name,
            config=config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
            decoder_def=recog_def,
            model_def=model_def,
            model_config=model_config,
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
            train_vocab_opts=None,
            decoding_config=decoding_config, #Careful, anything inside would be hashed if not removed before passing to build forward config
            exclude_epochs=exclude_epochs,
            with_prior=with_prior,
            empirical_prior=empirical_prior,
            prior_from_max=prior_from_max,
            tune_hyperparameters=tune_hyperparameters,
            tune_config_updates=tune_config_updates, #
            recog_config_updates=recog_config_updates, #
            search_mem_rqmt=search_mem_rqmt,
            search_rqmt=search_rqmt,
            recog_epoch=recog_epoch, #Ignored for training case
            recog=not train,
            batch_size=batch_size,
        )[1:], f"{p0}{p3}_{p3_}{p4}{p6}", decoding_config["lm_weight"], decoding_config["prior_weight"]



def py():
    """Sisyphus entry point"""
    """We have bpe128: [ctc blstm, ctc conformer], bpe10k:[ctc conformer]"""
    available = [("bpe128","ctc","blstm"),("bpe128","ctc","conformer"),("bpe10k","ctc","conformer")]
    models = {"ctc": ctc_exp, "transducer": None, "AED":None}
    encoder = "conformer" #blstm conformer
    train = False # Weather train the AM
    lm_types_names = set()
    for vocab in ["bpe128",
                  #"bpe10k", # 6.49  # For now only have Conformer CTC
                  ]:
        # from ..datasets.librispeech_lm import get_4gram_binary_lm

        from .language_models.n_gram import get_count_based_n_gram # If move to lm dir it somehow breaks the hash...
        lms = dict()
        lms.update({"NoLM": None})
        ppl_results = dict()
        exp_names_postfix = ""
        prune_num = 0
        # vocabs = {str(i) + "gram":_get_bpe_vocab(i) for i in [3,4,5,6,7,8,9,10]}
        ''' ----------------------Add bpe count based n-gram LMs(Only if ctc also operates on same bpe)--------------------'''
        for n_order in [#2,
            # 3,
            #4,
            #5,
            #6,# Slow recog
        ]:
            prune_threshs = [
                            1e-9, 1.3e-8, 6.7e-8,
                             3e-7, 7e-7, 1.7e-6,
                             #5.1e-6, #8.1e-6, #Not word for 6 gram
                            #1.1e-5, 1.6e-5, 1.9e-5
                             ]
            prune_threshs.sort()
            exp_names_postfix += str(n_order) + "_"
            if n_order != 4:
                prune_threshs = [x for x in prune_threshs if x and x < 9*1e-6]
                #print(prune_threshs)

            prune_threshs.append(None)
            for prune_thresh in prune_threshs: # + [None]: #[x * 1e-9 for x in range(1, 30, 3)]
                prune_num += 1 if prune_thresh else 0
                lm, ppl_log = get_count_based_n_gram(vocab, n_order, prune_thresh)
                lm_name = str(n_order) + "gram_" + vocab + (f"_{prune_thresh:.1e}" if prune_thresh else "").replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
                lms.update(dict([(lm_name, lm)]))
                ppl_results.update(dict([(lm_name, ppl_log)]))
                tk.register_output(f"datasets/LibriSpeech/lm/ppl/" + lm_name, ppl_log)
            lm_types_names.add(f"{n_order}gram")
        # if re.match("^bpe[0-9]+.*$", vocab):
        #     #bpe = bpe10k if vocab == "bpe10k" else _get_bpe_vocab(bpe_size=vocab[len("bpe") :])
        #     for n_order in [#2, 3,
        #               #4, 5, 6,
        #               #7, 8, # TODO: Currently KenLM only support up to 6 order
        #               #9, 10
        #               ]: # Assume we only do bpe for now
        #         for prune_thresh in [x*1e-9 for x in range(1,20,3)]:
        #             lm_name = str(n_order) + "gram" + f"{prune_thresh:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
        #             lm, ppl_var = get_count_based_n_gram(vocab, n_order, prune_thresh)
        #             lms.update(dict([(lm_name, lm)]))
        #             ppl_results.update(dict([(lm_name, ppl_var)]))
        #             tk.register_output(f"datasets/LibriSpeech/lm/ppl/" + lm_name + "_" + vocab, ppl_var)
        ''' ----------------------Add <word> count based n-gram LMs--------------------'''

        for n_order in [#2,
            # 3,
            #4, #5
            # 6 Slow recog
        ]:
            # prune_threshs = list(set([x * 1e-9 for x in range(1, 100, 6)] +   # These has correspond existing pruned LMs !! Float should not direct go in hashes
            #                          [x * 1e-7 for x in range(1, 200, 25)] +
            #                          [x * 1e-7 for x in range(1, 200, 16)] +
            #                          [x * 1e-7 for x in range(1, 10, 2)]
            #                          ))
            prune_threshs = [
                            # 1e-9, 1.3e-8, 6.7e-8,
                            #  3e-7, 7e-7, 1.7e-6,
                            #  5.1e-6,
                            #  8.1e-6, 1.1e-5, 1.6e-5, 1.9e-5
                             ]
            prune_threshs.sort()
            exp_names_postfix += str(n_order) + "_"
            if n_order != 4:
                prune_threshs = [x for x in prune_threshs if x and x < 9*1e-6]
                #print(prune_threshs)

            prune_threshs.append(None)
            for prune_thresh in prune_threshs: # + [None]: #[x * 1e-9 for x in range(1, 30, 3)]
                prune_num += 1 if prune_thresh else 0
                lm, ppl_log = get_count_based_n_gram("word", n_order, prune_thresh)
                lm_name = str(n_order) + "gram_word" + (f"{prune_thresh:.1e}" if prune_thresh else "").replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
                lms.update(dict([(lm_name, lm)]))
                ppl_results.update(dict([(lm_name, ppl_log)]))
                tk.register_output(f"datasets/LibriSpeech/lm/ppl/" + lm_name, ppl_log)
            lm_types_names.add(f"{n_order}gram")
        exp_names_postfix += f"ngram_pruned_{str(prune_num)}" if prune_num > 0 else ""

        ''' ----------------------Add $vocab$ FFNN LMs--------------------'''
        # /u/haoran.zhang/setups/2024-12-16--lm-ppl/work/i6_core/returnn/training/ReturnnTrainingJob.T5Vjltnx1Sp3/output/models/epoch.050
        # /u/haoran.zhang/setups/2024-12-16--lm-ppl/work/i6_core/returnn/training/ReturnnTrainingJob.UpknSQ5OLCQV/output/models/epoch.050.pt
        from .lm.ffnn import get_ffnn_lm
        epochs = [5, 10, 20, 40, 50] # For default 50 epoch training
        lm_configs = {
            "std":{"bpe128":{
                    "context_size": 8, #8,
                    "num_layers": 2,
                    "ff_hidden_dim": 2048,
                    "dropout": 0.1,
                },
            "bpe10k": {
                "context_size": 4,  # 15,
                "num_layers": 3,
                "ff_hidden_dim": 2048,
                "dropout": 0.1,
                }
            },
            "low":{"bpe128":{
                    "context_size": 5, #8,
                    "num_layers": 2,
                    "ff_hidden_dim": 1024,
                    "dropout": 0.2,
                },
            "bpe10k": {
                "context_size": 2,  # 15,
                "num_layers": 2,
                "ff_hidden_dim": 1024,
                "dropout": 0.2,
                }
            }
        }

        # f"ffnn{ctx_size}_{epoch}"
        #embed_dim 128, relu dropout=0.0,embed_dropout=0.0
        match = re.search(r"bpe(.+)", vocab)
        train_subsets = {"low": {"bpe128":30, "bpe10k":30}[vocab], "std": {"bpe128":80, "bpe10k":80}[vocab]} # for bpe128 75 already gives similar ppl as None
        model_capas = ["low", "std"]  # ?
        for model_capa in model_capas:
            for lm_checkpoint, ppl, epoch in get_ffnn_lm(get_vocab_by_str(vocab), **lm_configs[model_capa][vocab],
                                                         epochs=epochs, train_subset=train_subsets[model_capa]):#context_size=ctx_size, num_layers=2, ff_hidden_dim=2048)
                lm_config_ = lm_configs[model_capa][vocab].copy()
                lm_config_["class"] = "FeedForwardLm"
                ctx_size = lm_config_["context_size"]
                ffnnlm_name = f"ffnn{ctx_size}_{epoch}" + model_capa + "_bpe"+match.group(1)
                ppl_results.update(dict([(ffnnlm_name, ppl)]))
                ffnn_lm = {
                            "preload_from_files": {
                            "recog_lm": {
                                "prefix": "recog_language_model.",
                                "filename": lm_checkpoint.checkpoint,#"/u/marten.mueller/dev/ctc_baseline/work/i6_core/returnn/training/ReturnnTrainingJob.1QR8IB9ySxBq/output/models/epoch.050.pt",#,
                                },
                            },
                            "recog_language_model":lm_config_
                        }
                lms.update({ffnnlm_name: ffnn_lm})
                lm_types_names.add("ffnn")


        '''--------------Add trafo LMs-------------------'''
        """!!!Note: With small max_seq_length_default_target, e.g default 75
        , the trafo lm(its ppl) is only evaluated on short sequences but compute_ppl is on full sequence"""
        if vocab in ["bpe128"]: # Only have bpe128 now
            from .lm.trafo import get_trafo_lm
            epochs = [100] # 10. 20. 40 80 100 /  5 10 20 40 50
            trafo_lm_configs = {"bpe128":{
                        "num_layers": 12, #24 12
                        "model_dim": 512, #1024 512
                        "dropout": 0.0,
                    },
                "bpe10k": {
                    "num_layers": None,
                    "model_dim": None,
                    "dropout": None,
                }
            }
            trafo_lm_config_ = trafo_lm_configs[vocab].copy()
            trafo_lm_config_["class"] = "TransformerLm"

            match = re.search(r"bpe(.+)", vocab)

            from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe
            from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
            bpe = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=10_000 if vocab == "bpe10k" else int(match.group(1)))
            bpe = Bpe(dim=184, codes=bpe.bpe_codes, vocab=bpe.bpe_vocab, eos_idx=0, bos_idx=0,
                         unknown_label="<unk>")


            get_lm_config = trafo_lm_configs[vocab].copy()
            # Test newly trained trafo lm
            get_lm_config.update({
                "n_ep": 50, "bs_feat": 10_000,
                "num_layers": 12, "model_dim": 512,
                "max_seqs": 200, "max_seq_length_default_target": True})
            trafo_lm_config_.update({"num_layers": 12, "model_dim": 512})
            epochs = [20, 50]
            #--------------------------------
            #
            for lm_checkpoint, ppl, epoch in get_trafo_lm(bpe, **get_lm_config,epochs=epochs):
                trafo_lm_name = f"trafo_{epoch}" + "_bpe"+match.group(1)
                ppl_results.update(dict([(trafo_lm_name, ppl)]))
                trafo_lm = {
                            "preload_from_files": {
                            "recog_lm": {
                                "prefix": "recog_language_model.",
                                "filename": lm_checkpoint.checkpoint,
                                },
                            },
                            "recog_language_model":trafo_lm_config_
                        }
                lms.update({trafo_lm_name: trafo_lm})
                lm_types_names.add("trafo")
            exp_names_postfix += f"trafo_{str(len(epochs))}epochs_" if len(epochs) > 0 else ""
        '''----------------------------------------------'''

        # Try to use the out of downstream job which has existing logged output. Instead of just Forward job, which seems cleaned up each time
        for model, exp in models.items():
            if not exp:
                continue
            if (vocab, model, encoder) not in available:
                train = True
            wer_ppl_results = dict()
            #----------Test--------------------
            lms = ({"NoLM": None})
            # ffnn_lm = {
            #             "preload_from_files": {
            #             "recog_lm": {
            #                 "prefix": "recog_language_model.",
            #                 "filename": "/u/haoran.zhang/setups/2024-12-16--lm-ppl/work/i6_core/returnn/training/ReturnnTrainingJob.RS9ctpmX4WtM/output/models/epoch.050.pt",
            #                 #10k :/u/haoran.zhang/setups/2024-12-16--lm-ppl/work/i6_core/returnn/training/ReturnnTrainingJob.RS9ctpmX4WtM/output/models/epoch.050.pt
            #                 #bpe128: "/u/haoran.zhang/setups/2024-12-16--lm-ppl/work/i6_core/returnn/training/ReturnnTrainingJob.UpknSQ5OLCQV/output/models/epoch.050.pt"
            #                 },
            #             },
            #             "recog_language_model":lm_config_
            #         }
            # ffnnlm_name = "ffnn8_50" + "_bpe128"
            # lms = {ffnnlm_name: ffnn_lm}
            #------------------------------
            if train:
                lms = ({"NoLM": None})

            lm_hyperparamters_strs = dict()
            for name, lm in lms.items():
                # lm_name = lm if isinstance(lm, str) else lm.name
                rescor_config = None
                if name in ["NoLM"]:
                    rescor_config = ("Llama-3.2-1B", get_llm("Llama-3.2-1B"))
                wer_result_path, search_error, lm_tune, prior_tune, lm_hyperparamters_str, dafault_lm_scale, dafault_prior_scale = exp(name, lm, vocab,
                                                                                                                                       rescor_config = rescor_config,
                                                                                                                                       encoder=encoder, train=train)
                if rescor_config:
                    lm_hyperparamters_str += f"rescor_with{rescor_config[0]}"
                for lm_type in lm_types_names:
                    if lm_type in name:
                        lm_hyperparamters_strs[lm_type] = " " if not lm_hyperparamters_str else lm_hyperparamters_str
                if lm:
                    wer_ppl_results[name] = (ppl_results.get(name), wer_result_path, search_error, lm_tune, prior_tune, dafault_lm_scale, dafault_prior_scale)

            lms_info = "_".join([k+v for k,v in lm_hyperparamters_strs.items()])
            # TODO: lm_hyperparamters_str is not unique across lms, but unique for same type(n-gram ffnn trafo)
            if wer_ppl_results and not train:
                (names, res) = zip(*wer_ppl_results.items())
                results = [(x[0],x[1]) for x in res]
                search_errors = [x[2] for x in res]
                lm_tunes = [x[3] for x in res]
                prior_tunes = [x[4] for x in res]
                dafault_lm_scales = [x[5] for x in res]
                dafault_prior_scales = [x[6] for x in res]
                summaryjob = WER_ppl_PlotAndSummaryJob(names, results, lm_tunes, prior_tunes, search_errors, dafault_lm_scales, dafault_prior_scales)
                tk.register_output("wer_ppl/" + model + "_" + vocab + encoder + (f"{BLSTM_Enc_dim}" if encoder =="blstm" else "") + recog_info + lms_info + exp_names_postfix + "/summary", summaryjob.out_summary)
                tk.register_output("wer_ppl/" + model + "_" + vocab + encoder + (f"{BLSTM_Enc_dim}" if encoder =="blstm" else "") + recog_info + lms_info + exp_names_postfix + "/dev_other.png", summaryjob.out_plot1)
                tk.register_output("wer_ppl/" + model + "_" + vocab + encoder + (f"{BLSTM_Enc_dim}" if encoder =="blstm" else "") + recog_info + lms_info + exp_names_postfix + "/test_other.png",
                                   summaryjob.out_plot2)
