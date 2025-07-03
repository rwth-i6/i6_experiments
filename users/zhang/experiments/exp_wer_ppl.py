"""
CTC experiments (refactored).
"""

from __future__ import annotations

import re
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING

from i6_experiments.users.schmitt.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
from sisyphus import tk

from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob, GnuPlotJob
from .lm_getter import build_all_lms  # NEW

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
# --- Constants ---
USE_flashlight_decoder = False
EVAL_DATASET_KEYS = ["test-other","dev-other","test-clean","dev-clean"]
DEFAULT_PRIOR_WEIGHT = 0.15
DEFAULT_PRIOR_TUNE_RANGE = [-0.1, -0.05, 0.0, 0.05, 0.1]
DEFAUL_RESCOR_LM_SCALE = 0.2
CHEAT_N_BEST = False
# --- Helpers for ctc_exp ---

def get_encoder_model_config(encoder: str) -> Tuple[dict, Optional[callable]]:
    import returnn.frontend as rf
    from returnn.frontend.encoder.conformer import ConformerEncoderLayer

    model_def = None

    if encoder == "conformer":
        enc_conformer_layer_default = rf.build_dict(
            ConformerEncoderLayer,
            ff_activation=rf.build_dict(rf.relu_square),
            num_heads=8,
        )
        model_config = {"enc_conformer_layer": enc_conformer_layer_default, "feature_batch_norm": True}

    elif encoder == "blstm":
        from i6_experiments.users.zhang.experiments.encoder.blstm import ctc_model_def as blstm_model_def
        model_def = blstm_model_def
        model_config = {"enc_dim": 1024}

    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    return model_config, model_def

def get_decoding_config(lmname: str, lm, vocab: str, encoder: str, nbest: int =50, beam_size: int=80) -> Tuple[dict, dict, dict, dict, bool, Optional[int]]:
    if nbest:
        assert beam_size > nbest
    decoding_config = {
        "log_add": False,
        "nbest": nbest,
        "beam_size": beam_size,
        "beam_threshold": 1e6,
        "lm_weight": 1.45,
        "use_logsoftmax": True,
        "use_lm": False,
        "use_lexicon": False,
    }
    tune_config_updates = {}
    recog_config_updates = {}
    search_rqmt = {}
    tune_hyperparameters = False
    batch_size = None

    if lmname != "NoLM":
        decoding_config["use_lm"] = True
        decoding_config["lm_order"] = lmname

        if lmname[0].isdigit():
            decoding_config["lm"] = lm
        else:
            decoding_config["recog_language_model"] = lm

    if re.match(r".*word.*", lmname):
        decoding_config["use_lexicon"] = True

    decoding_config["prior_weight"] = DEFAULT_PRIOR_WEIGHT
    tune_config_updates["priro_tune_range"] = DEFAULT_PRIOR_TUNE_RANGE

    if "ffnn" in lmname:
        tune_hyperparameters = True
        decoding_config["beam_size"] = 500 if vocab == "bpe128" else 150
        decoding_config["lm_weight"] = 0.5
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-20, 21, 5)]

    elif "trafo" in lmname:
        tune_hyperparameters = True
        decoding_config["beam_size"] = 80 if encoder == "conformer" else 300
        decoding_config["nbest"] = min(decoding_config["nbest"], decoding_config["beam_size"])
        decoding_config["lm_weight"] = 0.5
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-15, 16, 5)]


    elif "gram" in lmname and "word" not in lmname:
        decoding_config["beam_size"] = 600
        decoding_config["lm_weight"] = 0.5
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-30, 31, 15)]

    if vocab == "bpe10k" or "trafo" in lmname:
        if USE_flashlight_decoder:
            batch_size = 20_000_000 if decoding_config["beam_size"] < 20 else 60_000_000
            search_rqmt.update({"gpu_mem": 24 if decoding_config["beam_size"] < 20 else 48})
        elif "trafo" in lmname:
            batch_size = {"blstm": 1_800_000, "conformer": 1_000_000}[encoder] if decoding_config["beam_size"] > 50 \
                else {"blstm": 6_400_000, "conformer": 4_800_000}[encoder]
            search_rqmt.update({"gpu_mem": 24} if batch_size*decoding_config["beam_size"] <= 80_000_000 else {"gpu_mem": 32})
            if decoding_config["beam_size"] > 150:
                batch_size = {"blstm": 1_000_000, "conformer": 800_000}[encoder]
            if decoding_config["beam_size"] >= 280:
                batch_size = {"blstm": 800_000, "conformer": 500_000}[encoder]

    return decoding_config, tune_config_updates, recog_config_updates, search_rqmt, tune_hyperparameters, batch_size

def build_alias_name(lmname: str, decoding_config: dict, tune_config_updates: dict, vocab: str, encoder: str, with_prior: bool, prior_from_max: bool, empirical_prior: bool) -> tuple[str, str]:
    p0 = f"_p{str(decoding_config['prior_weight']).replace('.', '')}" + (
        "-emp" if empirical_prior else ("-from_max" if prior_from_max else "")) if with_prior else ""
    p3 = f"b{decoding_config['beam_size']}n{decoding_config['nbest']}"
    p4 = f"w{str(decoding_config['lm_weight']).replace('.', '')}" if decoding_config.get("use_lm") else ""
    p5 = f"re_{decoding_config['rescore_lm_name'].replace('.', '_')}" if decoding_config.get("rescoring") else ""
    p6 = f"rw{str(decoding_config['rescore_lmscale']).replace('.', '')}" if decoding_config.get("rescoring") else ""
    p7 = f"_tune" if tune_config_updates.get("tune_range_2") or tune_config_updates.get("prior_tune_range_2") else ""
    lm_hyperparamters_str = vocab + p0 + "_" + p3 + p4 + ("flash_light" if USE_flashlight_decoder else "")
    lm2_hyperparamters_str = "_" + p5 + "_" + p6 + p7
    alias_name = f"ctc-baseline_{encoder}_decodingWith_1st-{lmname}_{lm_hyperparamters_str}_2rd{lm2_hyperparamters_str}"
    first_pass_name = f"ctc-baseline_{encoder}_decodingWith_{lm_hyperparamters_str}_{lmname}"
    return alias_name, first_pass_name


def select_recog_def(lmname: str, USE_flashlight_decoder: bool) -> callable:
    from .ctc import recog_nn, model_recog, model_recog_lm, model_recog_flashlight

    if USE_flashlight_decoder:
        if "NoLM" in lmname:
            return model_recog_lm
        elif "ffnn" in lmname or "trafo" in lmname:
            return model_recog_flashlight
        else:
            return model_recog_lm
    else:
        if "ffnn" in lmname or "trafo" in lmname:
            return recog_nn
        elif "NoLM" in lmname:
            return recog_nn
        else:
            return model_recog_lm

# --- Main ctc_exp ---

def ctc_exp(
    lmname,
    lm,
    vocab,
    rescore_lm: Optional[ModelWithCheckpoint, dict] = None,
    rescore_lm_name: str = None,
    encoder: str = "conformer",
    train: bool = False,
):
    from i6_experiments.users.zhang.experiments.ctc import (
        train_exp,
        config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
        _get_cfg_lrlin_oclr_by_bs_nep,
        speed_pert_librosa_config,
        _raw_sample_rate,
    )

    # ---- Set up model and config ----
    model_config, model_def = get_encoder_model_config(encoder)
    (
        decoding_config,
        tune_config_updates,
        recog_config_updates,
        search_rqmt,
        tune_hyperparameters,
        batch_size,
    ) = get_decoding_config(lmname, lm, vocab, encoder, beam_size=300, nbest=200)

    if decoding_config.get("recog_language_model", False):
        model_config["recog_language_model"] = decoding_config["recog_language_model"]
    # ---- Additional parameters from original ----
    with_prior = True
    prior_from_max = False
    empirical_prior = False

    # Choose exclude_epochs and recog_epoch as in original logic:
    exclude_epochs = None
    recog_epoch = None
    if vocab == "bpe128":
        exclude_epochs = set(range(0, 501)) - set([477])
        recog_epoch = 500 if encoder == "blstm" else 477
    if vocab == "bpe10k":
        exclude_epochs = set(range(0, 500))
        recog_epoch = 500

    # For not-training, recog should be True
    recog = not train



    recog_def = select_recog_def(lmname, USE_flashlight_decoder)
    tune_rescore_scale = False
    if rescore_lm or rescore_lm_name:
        if lm is not None: #First pass with a LM
            decoding_config["tune_with_rescoring"] = True
        decoding_config["cheat"] = CHEAT_N_BEST
        decoding_config["check_search_error_rescore"] = True
        decoding_config["rescoring"] = True
        decoding_config["lm_rescore"] = rescore_lm
        decoding_config["rescore_lmscale"] = DEFAUL_RESCOR_LM_SCALE
        decoding_config["rescore_priorscale"] = 0.15
        decoding_config["rescore_lm_name"] = rescore_lm_name
        decoding_config["vocab"] = get_vocab_by_str(vocab)
        if "ffnn" in rescore_lm_name:
            tune_rescore_scale = True
            decoding_config["rescore_lmscale"] = 0.5
            decoding_config["rescore_priorscale"] = 0.10
            tune_config_updates["tune_range_2"] = [scale / 100 for scale in range(-50, 51, 2)]
            tune_config_updates["prior_tune_range_2"] = [scale / 100 for scale in range(-10, 3, 2)]

        elif "trafo" in rescore_lm_name:
            tune_rescore_scale = True
            decoding_config["rescore_lmscale"] = 0.5
            decoding_config["rescore_priorscale"] = 0.10
            tune_config_updates["tune_range_2"] = [scale / 100 for scale in range(-50, 51, 2)]
            tune_config_updates["prior_tune_range_2"] = [scale / 100 for scale in range(-10, 3, 2)]

        elif "Llama" in rescore_lm_name:
            tune_rescore_scale = True
            decoding_config["rescore_lmscale"] = 0.5
            decoding_config["rescore_priorscale"] = 0.10
            tune_config_updates["tune_range_2"] = [scale / 100 for scale in range(-50, 51, 2)]
            tune_config_updates["prior_tune_range_2"] = [scale / 100 for scale in range(-10, 3, 2)]

        elif "gram" in rescore_lm_name and "word" not in rescore_lm_name:
            tune_rescore_scale = True
            decoding_config["rescore_lmscale"] = 0.5
            decoding_config["rescore_priorscale"] = 0.10
            tune_config_updates["tune_range_2"] = [scale / 100 for scale in range(-50, 51, 2)]
            tune_config_updates["prior_tune_range_2"] = [scale / 100 for scale in range(-10, 3, 2)]

    if train:
        decoding_config = {
        "log_add": False,
        "nbest": 1,
        "beam_size": 1,
        "use_logsoftmax": True,
        "use_lm": False,
        "use_lexicon": False,
    }
        alias_name = f"ctc-baseline_{encoder}_recog_training"
        first_pass_name = alias_name # Just one pass

    else:
        alias_name, first_pass_name = build_alias_name(
            lmname, decoding_config, tune_config_updates, vocab, encoder, with_prior, prior_from_max, empirical_prior
        )
    print(alias_name)
    # ---- Search memory requirement ----
    search_mem_rqmt = 16 if vocab == "bpe10k" else 6

    # This part is actually redundant
    p0 = f"_p{str(decoding_config['prior_weight']).replace('.', '')}" + (
            "-emp" if empirical_prior else ("-from_max" if prior_from_max else "")) if with_prior and decoding_config.get("prior_weight") else ""
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

    # ---- Run train_exp ----
    return (
        *train_exp(
            name=alias_name,
            first_pass_name=first_pass_name,
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
            tune_rescore_scale=tune_rescore_scale,
            tune_config_updates=tune_config_updates,
            recog_config_updates=recog_config_updates,
            search_mem_rqmt=search_mem_rqmt,
            search_rqmt=search_rqmt,
            recog_epoch=recog_epoch,
            recog=recog,
            batch_size=batch_size,
            eval_dataset_keys=EVAL_DATASET_KEYS,
        )[1:],
        f"{p0}{p3}_{p3_}{p4}{p6}",  # add param string as needed
        decoding_config.get("lm_weight", 0),
        decoding_config.get("prior_weight", 0),
    )

# --- Main py() function ---

def py():
    # ! Note: for now when use rescoring, in first pass prior will not be considered, especially by greedy case
    # Beware that when do rescoring and use first pass lm, prior will be counted twice
    available = [("bpe128", "ctc", "blstm"), ("bpe128", "ctc", "conformer"), ("bpe10k", "ctc", "conformer")]
    models = {"ctc": ctc_exp}
    encoder = "conformer" # blstm conformer
    train = False

    for vocab in ["bpe128",
                  #"bpe10k",
                  ]:

        word_ppl = False # Default
        lm_kinds = ["ffnn",
                    #"trafo", #Trafo has better result on second pass
                    ] # LM that do first pass,
        lm_kinds_2 = ["ngram", # LM that do second pass
                    "ffnn", "trafo", #"LLM"
                    ]
        # if encoder == "conformer": # Don know why, for conformer now the WER of NN LMs are better by second pass...
        #     lm_kinds = []
        if "LLM" in lm_kinds_2:
            word_ppl = True
        lms, ppl_results, lm_types_names = build_all_lms(vocab, lm_kinds=lm_kinds)  # NEW
        lms.update({"NoLM": None})
        rescor_lms, ppl_results_2, lm_types_names_2 = build_all_lms(vocab, lm_kinds=lm_kinds_2, as_ckpt=True, word_ppl=word_ppl)


        rescor_lms.update({"NoLM": None})
        ppl_results_2.update({"uniform": float(get_vocab_by_str(vocab).dim)})

        #TODO also add the ppl of llm
        # print(lms)
        # print(rescor_lms)
        for model, exp in models.items():
            if (vocab, model, encoder) not in available:
                train = True

            wer_ppl_results = dict()
            wer_ppl_results_2 = dict()
            for name, lm in lms.items(): # First pass lms
                break_flag = False
                for name_2, lm_2 in rescor_lms.items(): # Second pass lms
                    #lm_hyperparamters_str seems not needed?
                    # TODO: there is no distinguish between 1/2 pass scales here
                    if any([lm_kind in name_2 for lm_kind in lm_kinds]): # Dont do second pass with first pass LMs
                        continue
                    if lm: # Do second pass with same LM in first pass, scales tune with rescoring on Greedy
                        name_2 = name
                        lm_2 = rescor_lms[name]
                        break_flag = True # Do it only once
                    print(name, name_2)
                    wer_result_path, search_error, search_error_rescore, lm_tune, prior_tune, lm_hyperparamters_str, dafault_lm_scale, dafault_prior_scale = exp(
                        name, lm, vocab,
                        rescore_lm=lm_2,
                        rescore_lm_name = name_2,
                        encoder=encoder, train=train
                    )
                    if lm_2:
                        wer_ppl_results_2[name_2] = (
                        ppl_results_2.get(name_2), wer_result_path, search_error, search_error_rescore, lm_tune, prior_tune, dafault_lm_scale,
                        dafault_prior_scale)
                    else:
                        if lm: # First pass with a lm
                            wer_ppl_results_2[name] = (
                                ppl_results_2.get(name), wer_result_path, search_error, search_error_rescore, lm_tune, prior_tune, dafault_lm_scale,
                        dafault_prior_scale)
                        else: # NoLM at all := Uniform LM.?
                            wer_ppl_results_2["uniform"] = (
                            ppl_results_2.get("uniform"), wer_result_path, search_error, search_error_rescore, None, None, 0, 0)
                    if break_flag: # Ensure for lm do first pass not do second pass multiple times
                        break
                #print(wer_ppl_results_2)
                # if lm:
                #     wer_ppl_results[name] = (
                #     ppl_results.get(name), wer_result_path, search_error, lm_tune, prior_tune, dafault_lm_scale,
                #     dafault_prior_scale)
            if wer_ppl_results_2 and not train:
                names, res = zip(*wer_ppl_results_2.items())
                results = [(x[0], x[1]) for x in res]
                search_errors = [x[2] for x in res]
                search_errors_rescore = [x[3] for x in res]
                lm_tunes = [x[4] for x in res]
                prior_tunes = [x[5] for x in res]
                dafault_lm_scales = [x[6] for x in res]
                dafault_prior_scales = [x[7] for x in res]

                summaryjob = WER_ppl_PlotAndSummaryJob(names, results, lm_tunes, prior_tunes, search_errors, search_errors_rescore, dafault_lm_scales, dafault_prior_scales, eval_dataset_keys=EVAL_DATASET_KEYS)
                gnuplotjob = GnuPlotJob(summaryjob.out_summary, EVAL_DATASET_KEYS)
                alias_prefix = f"wer_ppl/{'1st_pass_' if lm_kinds else ''}2rd_pass_" + model + "_" + vocab + encoder + ("n_best_cheat" if CHEAT_N_BEST else "") + ("with_llm" if "LLM" in lm_kinds else "")
                tk.register_output(alias_prefix + "/summary", summaryjob.out_summary)
                for i, key in enumerate(EVAL_DATASET_KEYS):
                    tk.register_output(alias_prefix + f"/{key}.png", summaryjob.out_plots[i])
                    tk.register_output(alias_prefix + f"gnuplot/{key}.png", gnuplotjob.out_plots[key])



            # if wer_ppl_results and not train:
            #     names, res = zip(*wer_ppl_results.items())
            #     results = [(x[0], x[1]) for x in res]
            #     search_errors = [x[2] for x in res]
            #     search_errors_rescore = [x[3] for x in res]
            #     lm_tunes = [x[3] for x in res]
            #     prior_tunes = [x[4] for x in res]
            #     dafault_lm_scales = [x[5] for x in res]
            #     dafault_prior_scales = [x[6] for x in res]
            #
            #     summaryjob = WER_ppl_PlotAndSummaryJob(names, results, lm_tunes, prior_tunes, search_errors,
            #                                            search_errors_rescore, dafault_lm_scales, dafault_prior_scales,
            #                                            eval_dataset_keys=EVAL_DATASET_KEYS)
            #     alias_prefix = "wer_ppl/" + model + "_" + vocab + encoder + ("n_best_cheat" if CHEAT_N_BEST else "")
            #     tk.register_output("wer_ppl/" + model + "_" + vocab + encoder + "/summary",
            #                        summaryjob.out_summary)
            #     for i, key in enumerate(EVAL_DATASET_KEYS):
            #         tk.register_output(alias_prefix + f"/{key}.png", summaryjob.out_plots[i])
            #         tk.register_output(alias_prefix + f"/{key}.png", summaryjob.out_plots[i])
