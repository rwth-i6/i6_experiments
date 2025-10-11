"""
CTC experiments on Apptek datasets(refactored).
"""

from __future__ import annotations

import re
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING

from i6_experiments.users.zhang.experiments.lm_getter import build_all_lms
from i6_experiments.users.schmitt.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
from returnn_common.datasets_old_2022_10.interface import VocabConfig
from sisyphus import tk

from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob, GnuPlotJob
#from .lm_getter import build_all_lms, build_ffnn_lms  # NEW

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe

# --- Decoding Parameters ---
USE_flashlight_decoder = True
DEV_DATASET_KEYS = ["dev-other",
                    "dev-clean"
                    ]
EVAL_DATASET_KEYS = ["test-other","dev-other",
                     #"test-clean","dev-clean"
                     ]
DEFAULT_PRIOR_WEIGHT = 0.3
DEFAULT_LM_SCALE_DICT = {"Llama-3.1-8B": 1.0, "Qwen3-1.7B-Base": 1.0,
                         "Qwen3-8B-Base":1.1, "phi-4": 1.1, "Llama-3.2-1B": 1.1}
DEFAULT_PRIOR_SCALE_DICT = {"Llama-3.1-8B": 0.40, "Qwen3-1.7B-Base": 0.30,
                            "Qwen3-8B-Base":0.40, "phi-4": 0.40, "Llama-3.2-1B": 0.40}
DEFAULT_PRIOR_TUNE_RANGE = [-0.1, -0.05, 0.0, 0.05, 0.1]
DEFAULT_LM_WEIGHT = 0.5
DEFAUL_RESCOR_LM_SCALE = DEFAULT_LM_WEIGHT # Keep this same, otherwise tune with rescoring will broken

# !! PLOT need to be set in main exp

CHEAT_N_BEST = False
TUNE_WITH_CHEAT = True and CHEAT_N_BEST
TUNE_TWO_ROUND = True


BEAM_SIZE = 1000
TRAFO_BEAM = 300
NBEST = 1000 # Use 100 for plot

TUNE_ON_GREEDY_N_LIST = False

LLM_WITH_PROMPT = False
LLM_WITH_PROMPT_EXAMPLE = True and LLM_WITH_PROMPT
LLM_FXIED_CTX = False and not LLM_WITH_PROMPT# Will be Imported by llm.get_llm()
LLM_FXIED_CTX_SIZE = 8

LLM_PREV_ONE_CTX = True and not LLM_FXIED_CTX
CHEAT_CTX = False and LLM_PREV_ONE_CTX
CTX_LEN_LIMIT = 100

TUNE_LLM_SCALE_EVERY_PASS = LLM_PREV_ONE_CTX and not CHEAT_CTX

DIAGNOSE = True and not TUNE_LLM_SCALE_EVERY_PASS
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

def get_decoding_config(lmname: str, lm, vocab: str, encoder: str, nbest: int =50, beam_size: int=80, real_vocab: VocabConfig = None) -> Tuple[dict, dict, dict, dict, bool, Optional[int]]:
    if nbest:
        assert beam_size >= nbest
    decoding_config = {
        "log_add": False, #Flashlight
        "nbest": 200,
        "beam_size": 200,
        "beam_threshold": 1e6, #Flashlight
        "lm_weight": 1.45,
        "use_logsoftmax": True,
        "use_lm": False,
        "use_lexicon": False, #Flashlight
        "vocab": real_vocab or get_vocab_by_str(vocab),
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

    if re.match(r".*word.*", lmname) and USE_flashlight_decoder:
        decoding_config["use_lexicon"] = True

    decoding_config["prior_weight"] = DEFAULT_PRIOR_WEIGHT
    tune_config_updates["priro_tune_range"] = DEFAULT_PRIOR_TUNE_RANGE

    if "ffnn" in lmname:
        tune_hyperparameters = False #This control one pass tune
        decoding_config["beam_size"] = BEAM_SIZE if vocab == "bpe128" else 150
        decoding_config["lm_weight"] = 0.2
        decoding_config["prior_weight"] = 0.15
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-50, 51, 5)]
        batch_size = 11_200_000
        search_rqmt.update({"gpu_mem": 11, "time":3})

    elif "trafo" in lmname:
        tune_hyperparameters = False
        decoding_config["beam_size"] = TRAFO_BEAM if encoder == "conformer" else 300
        decoding_config["nbest"] = min(300, decoding_config["beam_size"])
        decoding_config["lm_weight"] = DEFAULT_LM_WEIGHT
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-15, 16, 5)]
        batch_size = 1_600_000 if TRAFO_BEAM > 200 else 4_800_000
        search_rqmt.update({"gpu_mem": 80})


    elif "gram" in lmname and "word" not in lmname:
        decoding_config["beam_size"] = 600
        decoding_config["lm_weight"] = DEFAULT_LM_WEIGHT
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-30, 31, 15)]

    # if vocab == "bpe10k" or "trafo" in lmname:
    #     if "trafo" in lmname:
    #         #         ffnn_config["batch_size"] = 9_600_000
    #         #         search_rqmt.update({"gpu_mem": 11, "time":2})
    #         batch_size = {"blstm": 1_800_000, "conformer": 1_000_000 if "ES" in lmname else 1_000_000}[encoder] if decoding_config["beam_size"] > 50 \
    #             else {"blstm": 6_400_000, "conformer": 4_800_000 if "ES" in lmname else 4_800_000}[encoder]
    #         search_rqmt.update({"gpu_mem": 48} if batch_size*decoding_config["beam_size"] <= 80_000_000 else {"gpu_mem": 48})
    #         if decoding_config["beam_size"] > 150:
    #             batch_size = {"blstm": 1_000_000, "conformer": 800_000}[encoder]
    #         if decoding_config["beam_size"] >= 280:
    #             batch_size = {"blstm": 800_000, "conformer": 500_000}[encoder]

    return decoding_config, tune_config_updates, recog_config_updates, search_rqmt, tune_hyperparameters, batch_size

def build_alias_name(lmname: str, decoding_config: dict, tune_config_updates: dict, vocab: str, encoder: str, with_prior: bool, prior_from_max: bool, empirical_prior: bool) -> tuple[str, str]:
    p0 = f"_p{str(decoding_config['prior_weight']).replace('.', '')}" + (
        "-emp" if empirical_prior else ("-from_max" if prior_from_max else "")) if with_prior else ""
    p3 = f"b{decoding_config['beam_size']}n{decoding_config['nbest']}"
    p4 = f"w{str(decoding_config['lm_weight']).replace('.', '')}" if decoding_config.get("use_lm") else ""
    p5 = f"re_{decoding_config['rescore_lm_name'].replace('.', '_')}" if decoding_config.get("rescoring") else ""
    p6 = f"rw{str(decoding_config['rescore_lmscale']).replace('.', '')}" if decoding_config.get("rescoring") else ""
    p7 = f"_tune" if tune_config_updates.get("tune_range_2") or tune_config_updates.get("prior_tune_range_2") else ""
    lm_hyperparamters_str = vocab + p0 + "_" + p3 + p4 + ("flash_light" if USE_flashlight_decoder and "word" in lmname else "")
    lm2_hyperparamters_str = "_" + p5 + "_" + p6 + p7
    alias_name = f"LBS-ctc-baseline_{encoder}_decodingWith_1st-{lmname}_{lm_hyperparamters_str}_{'LMTune' if not TUNE_ON_GREEDY_N_LIST else ''}_2rd{lm2_hyperparamters_str}"
    first_pass_name = f"LBS-ctc-baseline_{encoder}_decodingWith_{lm_hyperparamters_str}_{lmname}_{'LMTune' if not TUNE_ON_GREEDY_N_LIST else ''}"
    return alias_name, first_pass_name


def select_recog_def(lmname: str, USE_flashlight_decoder: bool) -> callable:
    from .ctc import recog_nn, model_recog, model_recog_lm#, model_recog_flashlight

    if USE_flashlight_decoder and "word" in lmname:
        #elif "ffnn" in lmname or "trafo" in lmname:
            #return model_recog_flashlight
        return model_recog_lm
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
    *,
    lm_weight: Optional[float] = None,
    prior_weight: Optional[float] = None,
    lm_vocab: Optional[Bpe : SentencePieceModel] = None,
    rescore_lm: Optional[ModelWithCheckpoint, dict] = None,
    rescore_lm_name: str = None,
    encoder: str = "conformer",
    train: bool = False,
):
    #print(rescore_lm_name, lm_vocab, lmname)
    from i6_experiments.users.zhang.experiments.ctc import (
        train_exp,
        config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
        _get_cfg_lrlin_oclr_by_bs_nep,
        speed_pert_librosa_config,
        _raw_sample_rate,
    )
    import copy
    if lm_vocab is None:
        lm_vocab = vocab
        if rescore_lm_name and "spm10k" in rescore_lm_name:
            lm_vocab = "spm10k"
    # ---- Set up model and config ----
    model_config, model_def = get_encoder_model_config(encoder)
    #print(f"datasetkeys: {task.eval_datasets.keys()}")
    (
        decoding_config,
        tune_config_updates,
        recog_config_updates,
        search_rqmt,
        tune_hyperparameters,
        batch_size,
    ) = get_decoding_config(lmname, lm, vocab, encoder, beam_size=BEAM_SIZE, nbest=NBEST)

    decoding_config["lm_weight"] = lm_weight if lm_weight is not None else decoding_config["lm_weight"]
    decoding_config["prior_weight"] = prior_weight if prior_weight is not None else decoding_config["prior_weight"]

    if decoding_config.get("recog_language_model", False):
        model_config["recog_language_model"] = decoding_config.pop("recog_language_model")#decoding_config["recog_language_model"]
    # ---- Additional parameters from original ----
    with_prior = True
    prior_from_max = False
    empirical_prior = False

    # Choose exclude_epochs and recog_epoch as in original logic:
    exclude_epochs = None
    recog_epoch = None
    if vocab == "bpe128":
        recog_epoch = 500 if encoder == "blstm" else 477
    elif vocab == "bpe10k":
        recog_epoch = 500
    elif vocab == "spm10k":
        recog_epoch = 625

    # For not-training, recog should be True
    recog = not train



    def set_tune_range_by_name(rescore_lm_name, tune_config_updates,
                               first_pass:bool = False,
                               default_lm: float = DEFAULT_LM_WEIGHT, default_prior: float = DEFAULT_PRIOR_WEIGHT):
        lm_key = f"tune_range{'_2' if not first_pass else ''}"
        prior_key = f"prior_tune_range{'_2' if not first_pass else ''}"
        if "ffnn" in rescore_lm_name:
            tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in range(-30, 51, 2)]
            tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-30, 21, 2)]

        elif "trafo" in rescore_lm_name:
            tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in range(-30, 51, 2)]
            tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-30, 21, 2)]
            # tune_config_updates[lm_key] = [scale / 100 for scale in range(-20, 31, 2)]
            # tune_config_updates[prior_key] = [scale / 100 for scale in range(-10, 21, 2)]

        elif "gram" in rescore_lm_name: # and "word" not in rescore_lm_name:
            tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in range(-30, 51, 2)]
            tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-30, 21, 2)]

        elif any(llmname in rescore_lm_name for llmname in ["Llama", "Qwen", "phi"]):
            if CHEAT_CTX or not LLM_PREV_ONE_CTX:
                tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in range(-30, 51, 2) if default_lm + scale / 100 > 0]
                tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-30, 21, 2) if default_prior + scale / 100 > 0]
            else:
                tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in [-30,0,30]]
                tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-5, 6, 5)]
                # This will be used as offset
                tune_config_updates["second_tune_range"] = [scale / 100 for scale in range(-10, 11, 5)]
    recog_def = select_recog_def(lmname, USE_flashlight_decoder)
    from .ctc import recog_nn, model_recog, model_recog_lm
    if recog_def != model_recog_lm:
        for key in ["log_add", "beam_threshold", "use_lexicon"]:
            decoding_config.pop(key, None)
    tune_rescore_scale = False

    # if not TUNE_ON_GREEDY_N_LIST:  # TODO: Warning, very unclean, when use this with given rescore_lm..->
    #     # branch in the below will be exc, and same setting will be repeated
    #     # Make sure they are the same
    #     decoding_config["rescore_lmscale"] = DEFAUL_RESCOR_LM_SCALE  # 0.5
    #     decoding_config["rescore_priorscale"] = 0.30
    #     decoding_config["tune_with_rescoring"] = True
    #
    #     # ## Just safe guard, for now need them to be same
    #     # decoding_config["prior_weight"] = decoding_config["rescore_priorscale"]
    #     # decoding_config["lm_weight"] = decoding_config["rescore_lmscale"]
    #     set_tune_range_by_name(lmname, tune_config_updates,
    #                            default_lm=decoding_config["lm_weight"],
    #                            default_prior=decoding_config["prior_weight"],
    #                            first_pass=True)  # !!This overwrites the setting done in get_decoding_config

    if rescore_lm is None and lm is None:
        print("Pure greedy!!")
        decoding_config["beam_size"] = 1
        decoding_config["nbest"] = 1
        for key in ["lm_weight", "prior_weight"]:
            decoding_config.pop(key, None)
        with_prior = False

    if lm is not None:
        from i6_experiments.users.zhang.experiments.lm_getter import get_lm_by_name
        first_pass_lmname = "ffnn8_50std_bpe128"
        decoding_config["base_one_pass"] = {
            "nbest": 100,
            "beam_size": 100,
            "lm_weight": None,
            "use_logsoftmax": True,
            "use_lm": False,
            "lm_order": None,
            "lm": None,
            "rescoring": True,
            "rescore_lm_name": first_pass_lmname,
            "lm_rescore": get_lm_by_name(first_pass_lmname, task_name="LBS"),
            "lm_vocab": None,
            } if TUNE_ON_GREEDY_N_LIST else \
            {
            "nbest": 100,
            "beam_size": 150,
            "lm_weight": 0.5,
            "prior_weight": 0.3,
            "use_logsoftmax": True,
            "use_lm": True,
            "lm_order": first_pass_lmname,
            "lm": None,
            "recog_language_model":get_lm_by_name(first_pass_lmname, task_name="LBS", as_ckpt=False),
            "rescoring": True,
            "rescore_lm_name": lmname,
            "rescore_lmscale": decoding_config.get("lm_weight", 0.5),
            "rescore_priorscale": decoding_config.get("prior_weight", 0.3),
            "lm_rescore": get_lm_by_name(lmname, task_name="LBS"),
            "lm_vocab": None, #None is okay as long as vocab match AM
        }

    decoding_config["two_round_tune"] = TUNE_TWO_ROUND

    if rescore_lm or rescore_lm_name:
        tune_rescore_scale = True
        decoding_config["cheat"] = CHEAT_N_BEST
        decoding_config["cheat_tune"] = TUNE_WITH_CHEAT
        decoding_config["diagnose"] = DIAGNOSE
        decoding_config["check_search_error_rescore"] = True
        decoding_config["rescoring"] = True
        decoding_config["lm_rescore"] = rescore_lm
        decoding_config["rescore_lmscale"] = DEFAULT_LM_SCALE_DICT.get(rescore_lm_name,None) or DEFAUL_RESCOR_LM_SCALE  # 0.5
        decoding_config["rescore_priorscale"] = DEFAULT_PRIOR_SCALE_DICT.get(rescore_lm_name,None) or 0.30
        print(f"Default scales for {rescore_lm_name}: LM{decoding_config['rescore_lmscale']}, prior{decoding_config['rescore_priorscale']}")
        decoding_config["rescore_lm_name"] = rescore_lm_name
        decoding_config["lm_vocab"] = get_vocab_by_str(lm_vocab)
        set_tune_range_by_name(rescore_lm_name, tune_config_updates,
                               default_lm=decoding_config["rescore_lmscale"],
                               default_prior=decoding_config["rescore_priorscale"], first_pass=False)

    if lm is not None:  # First pass with a LM, setting the tuning strategy
        decoding_config["tune_with_rescoring"] = True and (lm_weight is None and prior_weight is None)  # Set to false if do one pass tuning
        tune_hyperparameters = False and not decoding_config["tune_with_rescoring"] # 2 false -> no tuning
        #decoding_config["prior_weight"] = decoding_config["rescore_priorscale"]  # Just safe guard, for now need them to be same
        if all([decoding_config["tune_with_rescoring"], tune_hyperparameters]):
            set_tune_range_by_name(lmname, tune_config_updates,
                                   default_lm=decoding_config["lm_weight"],
                                   default_prior=decoding_config["prior_weight"],
                                   first_pass=True)

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
    search_mem_rqmt = 16 if vocab == "bpe10k" or vocab == "spm10k" else 6
    print(f"basic mem_rqmt: {search_mem_rqmt}")

    # This part is actually redundant
    p0 = f"_p{str(decoding_config['prior_weight']).replace('.', '')}" + (
            "-emp" if empirical_prior else ("-from_max" if prior_from_max else "")) if with_prior and decoding_config.get("prior_weight") else ""
    p1 = "sum" if decoding_config.get('log_add') else "max"
    p2 = f"n{decoding_config['nbest']}"
    p3 = f"b{decoding_config['beam_size']}t{tune_config_updates['beam_size']}" if tune_config_updates.get('beam_size') else f"b{decoding_config['beam_size']}"
    p3 = "" if "NoLM" in lmname else p3
    p3_ = f"trsh{decoding_config['beam_threshold']:.0e}".replace("+0", "").replace("+", "") if decoding_config.get('beam_threshold') else "trsh50"
    p3_ = "" if not USE_flashlight_decoder and "word" in lmname else p3_
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
            #exclude_epochs=exclude_epochs,
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
            dev_dataset_keys=DEV_DATASET_KEYS,
            eval_dataset_keys=EVAL_DATASET_KEYS,
        )[1:],
        f"{p0}{p3}_{p3_}{p4}{p6}",  # add param string as needed
        decoding_config.get("lm_weight", 0) if rescore_lm is None else decoding_config.get("rescore_lmscale", 0),
        decoding_config.get("prior_weight", 0) if rescore_lm is None else decoding_config.get("rescore_priorscale", 0),
    )

# --- Main py() function ---

from i6_experiments.users.zhang.utils.report import GetOutPutsJob
def py():
    # from apptek_asr.artefacts import ArtefactSpecification
    # from apptek_asr.artefacts.factory import AbstractArtefactRepository
    # from sisyphus import gs
    # runtime_name = "ApptekCluster-ubuntu2204-tf2.15.1-pt2.3.0-2024-04-24"
    # aar = AbstractArtefactRepository()
    #
    # artefacts = {
    #     "runtime_spec": ArtefactSpecification("runtime", runtime_name),
    # }
    # gs.worker_wrapper = artefacts["runtime_spec"].build(aar).worker_wrapper

    # ! Note: for now when use rescoring, in first pass prior will not be considered, especially by greedy case
    # Beware that when do rescoring and use first pass lm, prior will be counted twice
    available = [("bpe128", "ctc", "blstm"), ("bpe128", "ctc", "conformer"), ("bpe10k", "ctc", "conformer")]
    models = {"ctc": ctc_exp}
    encoder = "conformer"
    train = False
    cuts = {"conformer": 65, "blstm":37}
    greedy_first_pass = True
    for vocab in ["bpe128",
                  #"bpe10k",
                  #"spm10k",
                  ]:
        word_ppl = False # Default
        # LM that do first pass,
        lm_kinds = {#"ffnn",
                    #"trafo", #nn has better result on second pass for cfm
                    #"trafo_n24d1024",
                    "4gram_word_official_LBS"
                    }
        lm_kinds_2 = {#"ngram", # LM that do second pass
                    #"word_ngram",
                    #"ffnn",
                    #"6gram",
                    #"trafo",
                    #"LLM"
                    }
        insert_spm10k_lm = False
        LLM_and_Batch_size = {"meta-llama/Llama-3.2-1B": 80,  # 40 on 48gb,
                              "meta-llama/Llama-3.1-8B": 175,#120 + ctx 100 for 141g,# 70 for 80gb # batch 50 + ctx 200 -> 40GB peak
                               #"Qwen/Qwen3-0.6B-Base": 51,
                          "Qwen/Qwen3-1.7B-Base": 80,  # 40 on 48gb# 80 + ctx 100 -> 79 peak
                              # "Qwen/Qwen3-4B-Base":24,
                              #"Qwen/Qwen3-8B-Base":50,
                             "microsoft/phi-4": 150,#120 for 141g, 50 for 80gb, #(batch 42 + ctx 200 -> 47gb peak. batch 150 + ctx 100 -> 105/130gb peak)
                              # "mistralai/Mistral-7B-v0.3": 4,
                              }  # Keys of this determines which LLM will be built by lm_getter

        #lm_kinds = [] if "ffnn" not in lm_kinds_2 else lm_kinds
        if "LLM" in lm_kinds_2 or "word_ngram" in lm_kinds_2 or insert_spm10k_lm:
            word_ppl = True
        lms, ppl_results, _ = build_all_lms(vocab, lm_kinds=lm_kinds, word_ppl=word_ppl, only_best=True, only_transcript=False,
                                            task_name="LBS")  # NEW
        #lms = {}
        #ppl_results = {}
        lms.update({"NoLM": None})
        # if not greedy_first_pass:
        #     lm_kinds_2.update(lm_kinds) # Redundant setting for get first pass result
        rescor_lms, ppl_results_2, _ = build_all_lms(vocab, lm_kinds=lm_kinds_2, as_ckpt=True, word_ppl=word_ppl, only_best=True, only_transcript=False,
                                                     task_name="LBS", llmids_batch_sizes=LLM_and_Batch_size)
        rescor_lms.update({"NoLM": None})
        if insert_spm10k_lm:
            from i6_experiments.users.zhang.experiments.lm_getter import build_trafo_lm_spm
            other_lms, other_lms_ppl, _ =  build_trafo_lm_spm()
            rescor_lms.update(other_lms)
            ppl_results_2.update(other_lms_ppl)

        rescor_lms.update({"NoLM": None})
        ppl_results_2.update({"uniform": {k:10240.0 for k in EVAL_DATASET_KEYS}})

        # print(lms)
        # print(rescor_lms)

        def first_pass_with_lm_exp(exp, model_name, lms, rescor_lms, lm_kinds, lm_kinds_2, lm_weight: Optional[float] = None, prior_weight:Optional[float] = None):
            if not greedy_first_pass:
                lms.pop("NoLM",None)
            nonlocal vocab, encoder, train, ppl_results_2, word_ppl
            wer_results = dict()
            for name, lm in lms.items():  # First pass lms
                # Do once one pass
                one_pass_res = exp(
                    name, lm, vocab,
                    encoder=encoder, train=train,
                    lm_vocab="spm10k" if "spm10k" in name else None,
                    lm_weight=lm_weight,
                    prior_weight=prior_weight,
                )
                break_flag = False
                wer_ppl_results_2 = dict()
                for name_2, lm_2 in rescor_lms.items():  # Second pass lms
                    if lm is None and lm_2 is not None: # Skip 2 pass on greedy first pass
                        continue
                    two_pass_same_lm = False
                    one_pass_lm = name_2
                    if name_2 == name:
                        wer_ppl_results_2[name_2] = (
                            ppl_results_2.get(name_2), *one_pass_res)
                        wer_results[one_pass_lm] = one_pass_res[-5]
                        two_pass_same_lm = True
                    print(name, name_2)
                    (wer_result_path, search_error, search_error_rescore, lm_tune, prior_tune, output_dict, rescor_ppls,
                     lm_hyperparamters_str, dafault_lm_scale, dafault_prior_scale) = exp(
                        name, lm, vocab,
                        rescore_lm=lm_2,
                        rescore_lm_name=name_2,
                        encoder=encoder, train=train,
                        lm_vocab="spm10k" if "spm10k" in name_2 else None,
                        lm_weight=lm_weight,
                        prior_weight=prior_weight,
                    )
                    if lm_2:
                        wer_ppl_results_2[f"{name} + {name_2}" if two_pass_same_lm else name_2] = (
                            ppl_results_2.get(name_2) if not rescor_ppls or CHEAT_CTX else rescor_ppls, wer_result_path, search_error, search_error_rescore, lm_tune,
                            prior_tune,
                            dafault_lm_scale,
                            dafault_prior_scale)
                        wer_results[f"{name} + {name_2}" if two_pass_same_lm else name_2] = output_dict
                    else: # Second pass NoLM, assert unique first pass lm
                        wer_ppl_results_2["uniform"] = (
                            ppl_results_2.get("uniform"), wer_result_path, search_error, search_error_rescore,
                            None,
                            None, 0, 0)
                        wer_results["uniform"] = output_dict
                        # if lm:  # First pass with a lm while second pass no LM, log the data from first pass lm
                        #     wer_ppl_results_2["uniform"] = (
                        #         ppl_results_2.get("uniform"), wer_result_path, search_error, search_error_rescore, lm_tune,
                        #         prior_tune, dafault_lm_scale,
                        #         dafault_prior_scale)
                        # else:  # NoLM at all := Uniform LM.?
                        #     if not word_ppl:
                        #         wer_ppl_results_2["uniform"] = (
                        #             ppl_results_2.get("uniform"), wer_result_path, search_error, search_error_rescore,
                        #             None,
                        #             None, 0, 0)
                    if break_flag:  # Ensure for lm do first pass not do second pass multiple times
                        break
                # print(wer_ppl_results_2)
                # if lm:
                #     wer_ppl_results[name] = (
                #     ppl_results.get(name), wer_result_path, search_error, lm_tune, prior_tune, dafault_lm_scale,
                #     dafault_prior_scale)
                if wer_ppl_results_2 and not train and lm is not None:
                    #print(wer_ppl_results_2)
                    names, res = zip(*wer_ppl_results_2.items())
                    results = [(x[0], x[1]) for x in res]
                    search_errors = [x[2] for x in res]
                    search_errors_rescore = [x[3] for x in res]
                    lm_tunes = [x[4] for x in res]
                    prior_tunes = [x[5] for x in res]
                    dafault_lm_scales = [x[6] for x in res]
                    dafault_prior_scales = [x[7] for x in res]

                    summaryjob = WER_ppl_PlotAndSummaryJob(names, results, lm_tunes, prior_tunes, search_errors,
                                                           search_errors_rescore, dafault_lm_scales, dafault_prior_scales,
                                                           eval_dataset_keys=EVAL_DATASET_KEYS)
                    gnuplotjob = GnuPlotJob(summaryjob.out_summary, EVAL_DATASET_KEYS, curve_point=cuts[encoder])
                    llm_related_name_ext = f"{'cheat_ctx' if CHEAT_CTX else ''}" + f"{'prompted' if LLM_WITH_PROMPT else ''}{'_eg' if LLM_WITH_PROMPT_EXAMPLE else ''}_LLMs" + ((f"ctx{LLM_FXIED_CTX_SIZE}" if LLM_FXIED_CTX else "") + (
                        f"prev_{CTX_LEN_LIMIT}ctx" if LLM_PREV_ONE_CTX else "")) if "LLM" in lm_kinds_2 else ""
                    alias_prefix = (
                            f"LBS_wer_ppl/{f'1st_pass_{name}'}2rd_pass{len(rescor_lms)}_" + model_name + "_" + vocab + encoder
                            + ("n_best_cheat" if CHEAT_N_BEST else "")
                            + llm_related_name_ext + (f"Beam_{BEAM_SIZE}_{NBEST}_best"))
                    summaryjob.add_alias(alias_prefix+"/summary_job")
                    tk.register_output(alias_prefix + "/report_summary",
                                       GetOutPutsJob(outputs=wer_results).out_report_dict)
                    tk.register_output(alias_prefix + "/summary", summaryjob.out_summary)
                    for i, key in enumerate(EVAL_DATASET_KEYS):
                        #tk.register_output(alias_prefix + f"/{key}.png", summaryjob.out_plots[i])
                        tk.register_output(alias_prefix + f"/gnuplot/{key}.pdf", gnuplotjob.out_plots[key])
                        tk.register_output(alias_prefix + f"/gnuplot/{key}_regression", gnuplotjob.out_equations[key])

        for model_name, exp in models.items():
            if (vocab, model_name, encoder) not in available:
                train = True
            #wer_ppl_results = dict()
            for lm_weight, prior_weight in [#(None,None), # Tune scale
                                            #(0.6,0.3),# For FFNN,
                                            (0.8,0.2),
                                            ]:
                first_pass_with_lm_exp(exp, model_name, lms, rescor_lms, lm_kinds, lm_kinds_2, lm_weight=lm_weight, prior_weight=prior_weight)

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
