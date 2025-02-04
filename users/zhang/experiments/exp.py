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

from sisyphus import tk
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer, ConformerConvSubsample
from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob

def ctc_exp(lmname, lm, vocab):
    """Experiments on CTC"""
    # ---------------------------------------------------------
    # model name: f"v6-relPosAttDef-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100"
    #               f"-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2"
    #               f"-[vocab]" + f"-[sample]"
    exclude_epochs = set(range(0, 500)) - set([477]) #Reduce #ForwardJobs, 477 is the best epoch for most cases.

    #-------------------setting decoding config-------------------
    decoding_config = {
        "log_add": False,
        "nbest": 1,
        "beam_size": 80,
        "lm_weight": 1.9,  # NOTE: weights are exponentials of the probs. 1.9 seems make the results worse by using selftrained lm
        "use_logsoftmax": True,
        "use_lm": True,
        "use_lexicon": False, # Do open vocab search when using bpe lms.
    }
    decoding_config["lm"] = lm
    # relPosAttDef: Use the default RelPosSelfAttention instead of the Shawn et al 2018 style, old RETURNN way.
    enc_conformer_layer_default = rf.build_dict(
        ConformerEncoderLayer,
        ff_activation=rf.build_dict(rf.relu_square),
        num_heads=8,
    )
    decoding_config["use_lm"] = True if lm else False
    if re.match(r".*word.*", lmname) or "NoLM" in lmname:
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

    alias_name = f"ctc-baseline" + "decodingWith_" + lm_hyperparamters_str + lmname if lm else f"ctc-baseline-" + vocab + lmname
    print(f"name{lmname}", "lexicon:" + str(decoding_config["use_lexicon"]))
    from i6_experiments.users.zhang.experiments.ctc import train_exp, model_recog_lm, config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4, _get_cfg_lrlin_oclr_by_bs_nep, speed_pert_librosa_config, _raw_sample_rate
    _, wer_result_path = train_exp(
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
        train_vocab_opts=None,
        decoding_config=decoding_config,
        exclude_epochs=exclude_epochs,
        search_mem_rqmt=30 if vocab == "bpe10k" else 6
    )
    return wer_result_path, lm_hyperparamters_str

def py():
    """Sisyphus entry point"""
    models = {"ctc": ctc_exp, "transducer": None}
    for vocab in ["bpe128",
                  #"bpe10k", # 6.49  # Require much more time on recog even with lexicon
                  ]:
        # from ..datasets.librispeech_lm import get_4gram_binary_lm
        from .language_models.n_gram import get_count_based_n_gram # If move to lm it somehow breaks the hash...
        lms = dict()
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
        prune_threshs = list(set([x * 1e-9 for x in range(1, 100, 6)] +
                                 [x * 1e-7 for x in range(1, 200, 25)] +
                                 [x * 1e-7 for x in range(1, 200, 16)] +
                                 [x * 1e-7 for x in range(1, 10, 2)]
                                 ))
        prune_threshs.sort()
        for n_order in [  # 2, 3,
            4, 5
            # 6 Slow recog
        ]:
            exp_names_postfix += str(n_order) + "_"
            if n_order == 5:
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
        exp_names_postfix += f"pruned_{str(prune_num)}"
        # Try to use the out of downstream job which has existing logged output. Instead of just Forward job, which seems cleaned up each time
        lms.update({"NoLM": None})
        # for prunning in [(None, 5), (None, 6), (None, 7), (None, 8)]: # (thresholds, quantization level)
        #     official_4gram, ppl_official4gram = get_4gram_binary_lm(**dict(zip(["prunning","quant_level"],prunning)))
        #     lms.update({"4gram_word_official"+f"q{prunning[1]}": official_4gram})
        #     ppl_results.update({"4gram_word_official"+f"q{prunning[1]}": ppl_official4gram})
        #print(ppl_results)
        for model, exp in models.items():
            if not exp:
                continue
            wer_ppl_results = dict()
            for name, lm in lms.items():
                # lm_name = lm if isinstance(lm, str) else lm.name
                wer_result_path, lm_hyperparamters_str = exp(name, lm, vocab)
                if lm:
                    wer_ppl_results[name] = (ppl_results.get(name), wer_result_path)
            (names, results) = zip(*wer_ppl_results.items())
            summaryjob = WER_ppl_PlotAndSummaryJob(names, results)
            tk.register_output("wer_ppl/"+ model + lm_hyperparamters_str + exp_names_postfix + "/summary", summaryjob.out_summary)
            tk.register_output("wer_ppl/"+ model + lm_hyperparamters_str + exp_names_postfix + "/plot.png", summaryjob.out_plot)
