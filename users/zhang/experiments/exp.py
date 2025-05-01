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

def ctc_exp(lmname, lm, vocab, encoder:str="conformer",train:bool=False):
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
        model_config = {}
    else:
        raise ValueError(f"Unknown encoder: {encoder}")


    recog_epoch = None
    if vocab == "bpe128":
        exclude_epochs = set(range(0, 501)) - set([477])#Reduce #ForwardJobs, 477 is the best epoch for most cases.
        recog_epoch = 477
    if vocab == "bpe10k":
        exclude_epochs = set(range(0, 500))
        recog_epoch = 500

    tune_hyperparameters = True


    with_prior = True
    prior_from_max = False # ! arg in recog.compute_prior, actually Unused
    empirical_prior = False
    #-------------------setting decoding config-------------------
    # Each time called ctc_exp there will be an independent copy of this config, TODO: declare this outside the function and use explicit copy
    decoding_config = {
        "log_add": False,
        "nbest": 1,
        "beam_size": 80, # 80 for previous exps on bpe128
        #"beam_threshold": 1e6,  # 14. 1000000
        "lm_weight": 1.45,  # Tuned_best val: 1.45 NOTE: weights are exponentials of the probs.
        "use_logsoftmax": True,
        "use_lm": False,
        "use_lexicon": False, # Do open vocab search when using bpe lms.
    }
    if lmname != "NoLM":
        decoding_config["lm_order"] = lmname
        if lmname[0].isdigit():
            decoding_config["lm"] = lm
        else: # in var lm Should be a config
            model_config["recog_language_model"] = lm

    ##Temporary setting for ffnn###
    if "ffnn" in lmname:
        tune_hyperparameters = True
        decoding_config["beam_size"] = 25
        decoding_config["lm_weight"] = 0.5 #for ffnn128, 0.5 was best in range 0.5-1.5
    ################################

    decoding_config["use_lm"] = True if lm else False

    if re.match(r".*word.*", lmname): # Why  or "NoLM"  in lmname?
        decoding_config["use_lexicon"] = True
    else:
        decoding_config["use_lexicon"] = False
    if with_prior:
        decoding_config["prior_weight"] = 0.15  # 0.15 as initial ref for tuning or if not using emprirical prior

    p0 = f"_p{str(decoding_config['prior_weight']).replace('.', '')}" + (
            "-emp" if empirical_prior else ("-from_max" if prior_from_max else "")) if with_prior else ""
    p1 = "sum" if decoding_config['log_add'] else "max"
    p2 = f"n{decoding_config['nbest']}"
    p3 = f"b{decoding_config['beam_size']}"
    p4 = f"w{str(decoding_config['lm_weight']).replace('.', '')}"
    p5 = "_logsoftmax" if decoding_config['use_logsoftmax'] else ""
    p6 = "_lexicon" if decoding_config['use_lexicon'] else ""
    lm_hyperparamters_str = f"{p0}_{p1}_{p2}_{p3}_{p4}{p5}{p6}"
    lm_hyperparamters_str = vocab + lm_hyperparamters_str  # Assume only experiment on one ASR model, so the difference of model itself is not reflected here

    alias_name = "ctc-baseline_" + encoder  + (("_decodingWith_" + lm_hyperparamters_str + lmname
                                               if lm else f"ctc-baseline-" + lm_hyperparamters_str + lmname) if not train else "")
    print(f"name{lmname}", "lexicon:" + str(decoding_config["use_lexicon"]))

    from i6_experiments.users.zhang.experiments.ctc import train_exp as train_exp, model_recog_lm, model_recog_flashlight, config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4, _get_cfg_lrlin_oclr_by_bs_nep, speed_pert_librosa_config, _raw_sample_rate
    search_mem_rqmt = 16 if vocab == "bpe10k"  else 6
    search_rqmt = {"cpu": search_mem_rqmt//2, "time": 6} if vocab == "bpe10k" else None
    if "ffnn" in lmname:
        search_mem_rqmt = 24
        search_rqmt = {"time": decoding_config["beam_size"]//3 + 4}

    return train_exp(
            name=alias_name,
            config=config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
            decoder_def=model_recog_flashlight if "ffnn" in lmname else model_recog_lm,
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
            decoding_config=decoding_config,
            exclude_epochs=exclude_epochs,
            with_prior=with_prior,
            empirical_prior=empirical_prior,
            prior_from_max=prior_from_max,
            tune_hyperparameters=tune_hyperparameters,
            search_mem_rqmt=search_mem_rqmt,
            search_rqmt=search_rqmt,
            recog_epoch=recog_epoch, #Ignored for training case
            recog=not train,
        )

def py():
    """Sisyphus entry point"""
    models = {"ctc": ctc_exp, "transducer": None}
    for vocab in ["bpe128",
                  #"bpe10k", # 6.49  # Require much more time on recog even with lexicon
                  ]:
        # from ..datasets.librispeech_lm import get_4gram_binary_lm
        from .language_models.n_gram import get_count_based_n_gram # If move to lm dir it somehow breaks the hash...
        lms = dict()
        ppl_results = dict()
        # vocabs = {str(i) + "gram":_get_bpe_vocab(i) for i in [3,4,5,6,7,8,9,10]}
        # ----------------------Add bpe count based n-gram LMs(Only if ctc also operates on same bpe)--------------------
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
        # ----------------------Add word count based n-gram LMs--------------------
        exp_names_postfix = ""
        prune_num = 0

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
                            1e-9, 1.3e-8, 6.7e-8,
                             3e-7, 7e-7, 1.7e-6,
                             5.1e-6,
                             8.1e-6, 1.1e-5, 1.6e-5, 1.9e-5
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
        exp_names_postfix += f"ngram_pruned_{str(prune_num)}" if prune_num > 0 else ""

        # ----------------------Add $vocab$ FFNN LMs--------------------
        # /u/haoran.zhang/setups/2024-12-16--lm-ppl/work/i6_core/returnn/training/ReturnnTrainingJob.T5Vjltnx1Sp3/output/models/epoch.050
        # /u/haoran.zhang/setups/2024-12-16--lm-ppl/work/i6_core/returnn/training/ReturnnTrainingJob.UpknSQ5OLCQV/output/models/epoch.050.pt
        from .lm.ffnn import get_ffnn_lm
        epochs = [5, 10, 20, 40, 50]
        lm_configs = {"bpe128":{
                    "context_size": 8, #8,
                    "num_layers": 2,
                    "ff_hidden_dim": 2048,
                    "dropout": 0.1,
                },
            "bpe10k": {
                "context_size": 4,  # 15,
                "num_layers": 2,
                "ff_hidden_dim": 1024,
                "dropout": 0.0,
            }
        }
        lm_config_ = lm_configs[vocab].copy()
        lm_config_["class"] = "FeedForwardLm"
        ctx_size = lm_config_["context_size"]
        # f"ffnn{ctx_size}_{epoch}"
        #embed_dim 128, relu dropout=0.0,embed_dropout=0.0
        match = re.search(r"bpe(.+)", vocab)

        # for lm_checkpoint, ppl, epoch in get_ffnn_lm(get_vocab_by_str(vocab), **lm_configs[vocab],epochs=epochs):#context_size=ctx_size, num_layers=2, ff_hidden_dim=2048)
        #     ffnnlm_name = f"ffnn{ctx_size}_{epoch}" + "_bpe"+match.group(1)
        #     ppl_results.update(dict([(ffnnlm_name, ppl)]))
        #     ffnn_lm = {
        #                 "preload_from_files": {
        #                 "recog_lm": {
        #                     "prefix": "recog_language_model.",
        #                     "filename": lm_checkpoint.checkpoint,#"/u/marten.mueller/dev/ctc_baseline/work/i6_core/returnn/training/ReturnnTrainingJob.1QR8IB9ySxBq/output/models/epoch.050.pt",#,
        #                     },
        #                 },
        #                 "recog_language_model":lm_config_
        #             }
        #     lms.update({ffnnlm_name: ffnn_lm})

        #---------------Hot fix-------------------
        from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg, TrainDef, ModelDef
        from i6_experiments.users.zhang.experiments.lm.ffnn import lm_model_def, FeedForwardLm
        from i6_experiments.users.zhang.experiments.lm.lm_ppl import compute_ppl_single_epoch
        from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint
        from i6_core.returnn.training import PtCheckpoint
        from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset, LibrispeechLmDataset
        lm_dataset = LibrispeechLmDataset(vocab=get_vocab_by_str(vocab))
        model_def = ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    FeedForwardLm,
                    num_layers=2,
                    context_size=8,
                    embed_dropout=0,
                    dropout=0.0,
                    ff_hidden_dim=2048,
                )
            }
        )
        train_prefix_name = f"ffnn-n2-ctx8-embd128-d2048-bpe128-drop0.0-relu"
        exponents = {"bpe128": 2.3, "bpe10k": 1.1}
        for epoch in epochs:
            model_path = f"/u/haoran.zhang/setups/2024-12-16--lm-ppl/work/i6_core/returnn/training/ReturnnTrainingJob.UpknSQ5OLCQV/output/models/epoch.{epoch:03}.pt"
            model_ckpt = ModelWithCheckpoint(model_def, PtCheckpoint(tk.Path(model_path))).checkpoint
            ffnnlm_name = f"ffnn{ctx_size}_{epoch}" + "_bpe"+match.group(1)
            ppl = compute_ppl_single_epoch(prefix_name=train_prefix_name,model_with_checkpoint=model_ckpt,epoch=epoch,model_def=model_def,dataset=lm_dataset,
                                           dataset_keys=["transcriptions-train", "transcriptions-test-other", "transcriptions-dev-other"],exponent=exponents[vocab])
            ppl_results.update(dict([(ffnnlm_name, ppl)]))
            ffnn_lm = {
                        "preload_from_files": {
                        "recog_lm": {
                            "prefix": "recog_language_model.",
                            "filename": model_path,#"/u/marten.mueller/dev/ctc_baseline/work/i6_core/returnn/training/ReturnnTrainingJob.1QR8IB9ySxBq/output/models/epoch.050.pt",#,
                            },
                        },
                        "recog_language_model":lm_config_
                    }
            lms.update({ffnnlm_name: ffnn_lm})
        #---------------------------------------

        exp_names_postfix += f"_nn_{str(len(epochs))}epochs_" if len(epochs) > 0 else ""
        # -------------------------------------


        # Try to use the out of downstream job which has existing logged output. Instead of just Forward job, which seems cleaned up each time
        # lms.update({"NoLM": None})
        # for prunning in [(None, 5), (None, 6), (None, 7), (None, 8)]: # (thresholds, quantization level)
        #     official_4gram, ppl_official4gram = get_4gram_binary_lm(**dict(zip(["prunning","quant_level"],prunning)))
        #     lms.update({"4gram_word_official"+f"q{prunning[1]}": official_4gram})
        #     ppl_results.update({"4gram_word_official"+f"q{prunning[1]}": ppl_official4gram})
        #print(ppl_results)
        for model, exp in models.items():
            if not exp:
                continue
            wer_ppl_results = dict()
            #----------Test--------------------
            #lms = ({"NoLM": None})
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
            #lms.update({"NoLM": None})
            for name, lm in lms.items():
                # lm_name = lm if isinstance(lm, str) else lm.name
                wer_result_path, search_error, lm_hyperparamters_str, lm_scale = exp(name, lm, vocab,
                                                                                     encoder="blstm", train=True)
                lm_hyperparamters_str = " " if not lm_hyperparamters_str else lm_hyperparamters_str
                if lm:
                    wer_ppl_results[name] = (ppl_results.get(name), wer_result_path, search_error, lm_scale)
            if wer_ppl_results:
                (names, res) = zip(*wer_ppl_results.items())
                results = [(x[0],x[1]) for x in res]
                search_errors = [x[2] for x in res]
                lm_scales = [x[3] for x in res]
                summaryjob = WER_ppl_PlotAndSummaryJob(names, results, lm_scales, search_errors)
                tk.register_output("wer_ppl/"+ model + lm_hyperparamters_str + exp_names_postfix + "/summary", summaryjob.out_summary)
                tk.register_output("wer_ppl/"+ model + lm_hyperparamters_str + exp_names_postfix + "/dev_other.png", summaryjob.out_plot1)
                tk.register_output("wer_ppl/" + model + lm_hyperparamters_str + exp_names_postfix + "/test_other.png",
                                   summaryjob.out_plot2)
