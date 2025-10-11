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
from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
from i6_experiments.users.zhang.utils.tools import DummyJob
from returnn_common.datasets_old_2022_10.interface import VocabConfig
from sisyphus import tk

from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob, GnuPlotJob
#from .lm_getter import build_all_lms, build_ffnn_lms  # NEW

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe

RETURNN_ROOT = "/home/mgunz/setups/2024-07-08--zeyer-setup-apptek/recipe/returnn" #"/nas/models/asr/hzhang/setups/2025-07-20--combined/returnn"
FINE_TUNED_MODEL = True # If use the FT model
CKPT_EPOCH = 25 if FINE_TUNED_MODEL else 625
# --- Decoding Parameters ---
USE_flashlight_decoder = False
from i6_experiments.users.zhang.experiments.apptek_exp_wer_ppl import seg_key
#seg_key = "ref" #aptk_leg ref
AGGREGATED_WER = False
DEV_DATASET_KEYS = [f"test_set.ES_ES.f8kHz.mtp_eval-v2.{seg_key}.ff_wer"] + [f"{key}.{seg_key}.ff_wer" for key in DEV_KEYS if "callhome" not in key]# or seg_key == "ref"] #if "conversation" not in key] #Evaluate on concatenated DEV_KEYS-> not implemented
EVAL_DATASET_KEYS = [f"{key}.{seg_key}.ff_wer" for key in TEST_KEYS if "mtp_eval-v2" not in key] #+ DEV_DATASET_KEYS + ["test_set.ES_ES.f16kHz.eval_voice_call-v3.ref.ff_wer", "test_set.ES_US.f16kHz.dev_conversations_202411-v2.ref.ff_wer"]#[f"{key}.ref.ff_wer" for key in DEV_KEYS + TEST_KEYS]#['test_set.ES.f8kHz.mtp_dev_heldout-v2.aptk_leg.ff_wer', 'test_set.ES.f8kHz.mtp_dev_heldout-v2.ref.ff_wer'] #


DEFAULT_PRIOR_WEIGHT = 0.3
DEFAULT_LM_SCALE_DICT = {"Llama-3.1-8B": 0.35, "Qwen3-1.7B-Base": 0.3, "phi-4": 0.45, "Llama-3.2-1B": 0.3}
DEFAULT_PRIOR_SCALE_DICT = {"Llama-3.1-8B": 0.22, "Qwen3-1.7B-Base": 0.18, "phi-4": 0.26, "Llama-3.2-1B": 0.2}
DEFAULT_PRIOR_TUNE_RANGE = [-0.1, -0.05, 0.0, 0.05, 0.1]
DEFAULT_LM_WEIGHT = 0.5
DEFAUL_RESCOR_LM_SCALE = DEFAULT_LM_WEIGHT # Keep this same, otherwise tune with rescoring will broken

# !! PLOT need to be set in main exp

trans_only_LM = False

CHEAT_N_BEST = False and seg_key == "ref"
TUNE_WITH_CHEAT = True and CHEAT_N_BEST
TUNE_TWO_ROUND = True

BEAM_SIZE = 300
FFNN_BEAM_SIZE = 500
NBEST = 80 # Use 100 for plot

USE_GIVEN_NBEST = True
if USE_GIVEN_NBEST:
    UNAVAILABE = ["common_voice", "mtp_eval-v2", "voice_call-v3"]
    DEV_DATASET_KEYS = [f"{key}.{seg_key}.ff_wer" for key in DEV_KEYS if "callhome" not in key and "heldout" not in key]  # or seg_key == "ref"] #if "conversation" not in key] #Evaluate on concatenated DEV_KEYS-> not implemented
    EVAL_DATASET_KEYS = [f"{key}.{seg_key}.ff_wer" for key in TEST_KEYS if
                         all(infix not in key for infix in UNAVAILABE)]  # + DEV_DATASET_KEYS + ["test_set.ES_ES.f16kHz.eval_voice_call-v3.ref.ff_wer", "test_set.ES_US.f16kHz.dev_conversations_202411-v2.ref.ff_wer"]#[f"{key}.ref.ff_wer" for key in DEV_KEYS + TEST_KEYS]#['test_set.ES.f8kHz.mtp_dev_heldout-v2.aptk_leg.ff_wer', 'test_set.ES.f8kHz.mtp_dev_heldout-v2.ref.ff_wer'] #

from i6_experiments.users.zhang.experiments.apptek_exp_wer_ppl import TUNE_ON_GREEDY_N_LIST
#These following do not have affect Set them in apptek_exp_wer_ppl and sync here
#TUNE_ON_GREEDY_N_LIST = False
#----------unused-----------------

LLM_WITH_PROMPT = False
LLM_WITH_PROMPT_EXAMPLE = True and LLM_WITH_PROMPT
LLM_FXIED_CTX = False and not LLM_WITH_PROMPT# Will be Imported by llm.get_llm()
LLM_FXIED_CTX_SIZE = 8
#----------unused-----------------

LLM_PREV_ONE_CTX = True and not LLM_FXIED_CTX
CHEAT_CTX = True and LLM_PREV_ONE_CTX and seg_key == "ref"
CTX_LEN_LIMIT = 100

TUNE_LLM_SCALE_EVERY_PASS = LLM_PREV_ONE_CTX and not CHEAT_CTX

DIAGNOSE = True and not TUNE_LLM_SCALE_EVERY_PASS

# --- Helpers for ctc_exp ---

def get_decoding_config(lmname: str, lm, vocab: str, encoder: str, nbest: int =50, beam_size: int=80, real_vocab: VocabConfig = None) -> Tuple[dict, dict, dict, dict, bool, Optional[int]]:
    if nbest:
        assert beam_size > nbest
    decoding_config = {
        #"log_add": False, #Flashlight
        "nbest": nbest,
        "beam_size": beam_size,
        #"beam_threshold": 1e6, #Flashlight
        "lm_weight": 1.45,
        "use_logsoftmax": True,
        "use_lm": False,
        #"use_lexicon": False, #Flashlight
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
        decoding_config["beam_size"] = FFNN_BEAM_SIZE
        decoding_config["lm_weight"] = DEFAULT_LM_WEIGHT
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-50, 51, 5)]
        # if decoding_config["beam_size"] > 300: This will be overwritten in recog_model
        #     search_rqmt.update({"gpu_mem": 80})
        #     batch_size = 3_200_000

    elif "trafo" in lmname:
        tune_hyperparameters = False
        decoding_config["beam_size"] = 80 if encoder == "conformer" else 300
        decoding_config["nbest"] = min(decoding_config["nbest"], decoding_config["beam_size"])
        decoding_config["lm_weight"] = DEFAULT_LM_WEIGHT
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-15, 16, 5)]


    elif "gram" in lmname and "word" not in lmname:
        decoding_config["beam_size"] = 600
        decoding_config["lm_weight"] = DEFAULT_LM_WEIGHT
        tune_config_updates["tune_range"] = [scale / 100 for scale in range(-30, 31, 15)]

    if vocab == "bpe10k" or "trafo" in lmname:
        if USE_flashlight_decoder:
            batch_size = 20_000_000 if decoding_config["beam_size"] < 20 else 60_000_000
            search_rqmt.update({"gpu_mem": 24 if decoding_config["beam_size"] < 20 else 48})
        elif "trafo" in lmname:
            batch_size = {"blstm": 1_800_000, "conformer": 1_000_000 if "ES" in lmname else 1_000_000}[encoder] if decoding_config["beam_size"] > 50 \
                else {"blstm": 6_400_000, "conformer": 4_800_000 if "ES" in lmname else 4_800_000}[encoder]
            search_rqmt.update({"gpu_mem": 48} if batch_size*decoding_config["beam_size"] <= 80_000_000 else {"gpu_mem": 48})
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
    alias_name = f"{seg_key}_apptek-ctc-baseline{'_FT' if FINE_TUNED_MODEL else ''}_{encoder}_decodingWith_1st-{lmname}_{lm_hyperparamters_str}_{'LMTune' if not TUNE_ON_GREEDY_N_LIST else ''}_2rd{lm2_hyperparamters_str}"
    first_pass_name = f"{seg_key}_apptek-ctc-baseline{'_FT' if FINE_TUNED_MODEL else ''}_{encoder}_decodingWith_{lm_hyperparamters_str}_{lmname}_{'LMTune' if not TUNE_ON_GREEDY_N_LIST else ''}"
    if USE_GIVEN_NBEST:
        alias_name = f"{seg_key}_apptek-ctc-baseline{'_FT' if FINE_TUNED_MODEL else ''}_{encoder}_decodingWith_GivenNbest_2rd{lm2_hyperparamters_str}"
        first_pass_name = f"{seg_key}_apptek-ctc-baseline{'_FT' if FINE_TUNED_MODEL else ''}_{encoder}_decodingWith_GivenNbest"

    return alias_name, first_pass_name


def select_recog_def(lmname: str, USE_flashlight_decoder: bool) -> callable:
    from .ctc import recog_nn, model_recog, model_recog_lm#, model_recog_flashlight

    if USE_flashlight_decoder:
        if "NoLM" in lmname:
            return model_recog_lm
        #elif "ffnn" in lmname or "trafo" in lmname:
            #return model_recog_flashlight
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
    *,
    model,
    prior_file,
    vocab_config,
    model_config,
    i6_models,
    lm_vocab: Optional[Bpe : SentencePieceModel] = None,
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
    import copy
    model_config = copy.deepcopy(model_config)
    if lm_vocab is None:
        lm_vocab = vocab
    #       --- get Task ---
    from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.task import get_asr_task_given_spm
    task = get_asr_task_given_spm(spm=vocab_config, returnn_root=tk.Path(RETURNN_ROOT),aggregate_wer=AGGREGATED_WER)
    #print(f"datasetkeys: {task.eval_datasets.keys()}")
    (
        decoding_config,
        tune_config_updates,
        recog_config_updates,
        search_rqmt,
        tune_hyperparameters,
        batch_size,
    ) = get_decoding_config(lmname, lm, vocab, encoder, beam_size=BEAM_SIZE, nbest=NBEST, real_vocab=vocab_config)

    decoding_config["extern_imports"] = [i6_models]
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
            tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in range(-50, 31, 2)]
            tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-30, 21, 2)]

        elif "trafo" in rescore_lm_name:
            tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in range(-50, 31, 2)]
            tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-30, 21, 2)]
            # tune_config_updates[lm_key] = [scale / 100 for scale in range(-20, 31, 2)]
            # tune_config_updates[prior_key] = [scale / 100 for scale in range(-10, 21, 2)]

        elif "gram" in rescore_lm_name: # and "word" not in rescore_lm_name:
            tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in range(-50, 31, 2)]
            tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-30, 21, 2)]

        elif any(llmname in rescore_lm_name for llmname in ["Llama", "Qwen", "phi"]):
            if CHEAT_CTX or not LLM_PREV_ONE_CTX:
                tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in range(-50, 31, 2) if default_lm + scale / 100 > 0]
                tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-30, 21, 2) if default_prior + scale / 100 > 0]
            else:
                tune_config_updates[lm_key] = [default_lm + scale / 100 for scale in range(-5, 6, 5)]
                tune_config_updates[prior_key] = [default_prior + scale / 100 for scale in range(-2, 3, 2)]
                # This will be used as offset
                tune_config_updates["second_tune_range"] = [scale / 100 for scale in range(-5, 5, 3)]

    recog_def = select_recog_def(lmname, USE_flashlight_decoder)
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
        decoding_config["lm_vocab"] = vocab_config if "ES" in rescore_lm_name else get_vocab_by_str(lm_vocab)
        set_tune_range_by_name(rescore_lm_name, tune_config_updates,
                               default_lm=decoding_config["rescore_lmscale"],
                               default_prior=decoding_config["rescore_priorscale"], first_pass=False)

    if lm is not None:  # First pass with a LM, tuned it with rescoring
        decoding_config["tune_with_rescoring"] = True  # Set to false if do one pass tuning
        #decoding_config["prior_weight"] = decoding_config["rescore_priorscale"]  # Just safe guard, for now need them to be same
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
    if USE_GIVEN_NBEST:
        from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import NbestListDataset
        assert vocab_config is not None
        decoding_config["Nbest_dataset"] = NbestListDataset(spm=vocab_config)
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
            task=task,
            model_with_checkpoints=model,
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
            prior_file=prior_file,
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
#This have 0 prior for disabled idx, cautious about bias
PRIOR_PATH = {("spm10k", "ctc", "conformer"): tk.Path("/nas/models/asr/hzhang/setups/2025-07-20--combined/data/ES/prior/ctc_mbw_16kHz_spm10k.txt"),
              }

from i6_experiments.users.zhang.experiments.apptek_exp_wer_ppl import GetOutPutsJob
def py():
    # ! Note: for now when use rescoring, in first pass prior will not be considered, especially by greedy case
    # Beware that when do rescoring and use first pass lm, prior will be counted twice
    available = [("spm10k", "ctc", "conformer")]
    models = {"ctc": ctc_exp}
    encoder = "conformer"
    train = False
    insert_spm10k_lm = False

    cuts = {"conformer": 65, "blstm":37}
    # ---- Set up model and config ----
    from i6_experiments.users.zhang.experiments.apptek.am.ctc_spm10k_16khz_mbw import get_model_and_vocab, \
        NETWORK_CONFIG_KWARGS
    model, spm, i6_models = get_model_and_vocab(fine_tuned_model=FINE_TUNED_MODEL)
    model_config = {"network_config_kwargs": NETWORK_CONFIG_KWARGS,
                    "preload_from_files": {
                        "am": {
                            "prefix": "AM.",
                            "filename": model.get_epoch(CKPT_EPOCH).checkpoint
                        },
                    },
                    "allow_random_model_init": True,
                    }

    for k, v in spm["vocabulary"].items():
        print(f"{k}: {v}")
    # print(f"vocab setting: {spm}")
    vocab_config = SentencePieceModel(dim=spm["vocabulary"]["vocabulary_size"], model_file=spm["spm"])
    greedy_first_pass = False
    for vocab in [#"bpe128",
                  #"bpe10k",
                  "spm10k",
                  ]:
        word_ppl = True # Default if not plot
        # LM that do first pass,
        lm_kinds = {"ffnn",
                    #"trafo", #nn has better result on second pass for cfm
                    }
        lm_kinds_2 = {#"ngram", # Ngrams with varies PPLs
                    "4gram",
                    #"word_ngram",
                    #"ffnn",
                    #"trafo",
                    #"LLM"
                    }
        #lm_kinds = [] if "ffnn" not in lm_kinds_2 else lm_kinds
        reduce_offset = max(0,10*(CTX_LEN_LIMIT//100 - 1)) if LLM_PREV_ONE_CTX else -20
        LLM_and_Batch_size = {"meta-llama/Llama-3.2-1B": 40 - reduce_offset,  # 40*6,
                              #"meta-llama/Llama-3.1-8B": 40 - reduce_offset,#base 80 for 141 40 for 80
                              # "Qwen/Qwen3-0.6B-Base": 51,
                              "Qwen/Qwen3-1.7B-Base": 40 - reduce_offset,  # 40*6,#15 has peak 19GB on 48G, so can be at least doubled
                              # "Qwen/Qwen3-4B-Base":40,
                              #"Qwen/Qwen3-8B-Base":40,
                             #"microsoft/phi-4": 40 - reduce_offset,# base 80 for 141 40 for 80
                              # "mistralai/Mistral-7B-v0.3": 4,
                              }  # Keys of this determines which LLM will be built by lm_getter


        if "LLM" in lm_kinds_2:
            word_ppl = True
            #lm_kinds = ["ffnn"]
            #lm_kinds_2 = ["trafo", "LLM"]
        lms, ppl_results, _ = build_all_lms(vocab_config, lm_kinds=lm_kinds, word_ppl=word_ppl, only_best=True, only_transcript=trans_only_LM,
                                            task_name="ES")  # NEW
        #lms = {}
        #ppl_results = {}
        lms.update({"NoLM": None})
        # if not greedy_first_pass:
        #     lm_kinds_2.update(lm_kinds) # Redundant setting for get first pass result
        rescor_lms, ppl_results_2, _ = build_all_lms(vocab_config, lm_kinds=lm_kinds_2, as_ckpt=True, word_ppl=word_ppl, only_best=True, only_transcript=trans_only_LM,
                                                     task_name="ES", llmids_batch_sizes=LLM_and_Batch_size)
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

        def greedy_first_pass_exp(exp, model_name, lms, rescor_lms, lm_kinds, lm_kinds_2):
            nonlocal vocab, encoder, train, ppl_results_2, word_ppl
            wer_ppl_results_2 = dict()
            wer_results = dict()
            if len(lms) > 1 or (len(lms) == 1 and lms.get("NoLM", 0) == 0):
                print(f"Why set first pass LM while using this method? lms passed {lms}")
            lms = {"NoLM": None}
            for name, lm in lms.items():  # First pass lms
                # Do once one pass
                (wer_result_path, search_error, search_error_rescore, lm_tune, prior_tune, output_dict,
                 lm_hyperparamters_str, dafault_lm_scale, dafault_prior_scale) = exp(
                    name, lm, vocab,
                    encoder=encoder, train=train,
                    lm_vocab="spm10k" if "spm10k" in name else None,
                    model=model,
                    model_config=model_config,
                    vocab_config=vocab_config,
                    prior_file=PRIOR_PATH[(vocab, "ctc", encoder)],
                    i6_models=i6_models,
                )
                break_flag = False
                for name_2, lm_2 in rescor_lms.items():  # Second pass lms
                    # lm_hyperparamters_str seems not needed?
                    # TODO: there is no distinguish between 1/2 pass scales here
                    if any([lm_kind in name_2 for lm_kind in lm_kinds]):  # Dont do second pass with first pass LMs
                        continue
                    two_pass_same_lm = False
                    if name_2 == name:
                        wer_ppl_results_2[name_2] = (
                            ppl_results_2.get(name_2), wer_result_path, search_error, search_error_rescore, lm_tune,
                            prior_tune,
                            dafault_lm_scale,
                            dafault_prior_scale)
                        wer_results[name_2] = output_dict
                        two_pass_same_lm = True
                    if lm:  # Do second pass with same LM in first pass, scales tune with rescoring on Greedy
                        name_2 = name
                        lm_2 = rescor_lms[name]
                        break_flag = True  # Do it only once
                    print(name, name_2)
                    (wer_result_path, search_error, search_error_rescore, lm_tune, prior_tune, output_dict,
                     lm_hyperparamters_str, dafault_lm_scale, dafault_prior_scale) = exp(
                        name, lm, vocab,
                        rescore_lm=lm_2,
                        rescore_lm_name=name_2,
                        encoder=encoder, train=train,
                        model=model,
                        model_config=model_config,
                        vocab_config=vocab_config,
                        prior_file=PRIOR_PATH[(vocab, "ctc", encoder)],
                        i6_models=i6_models,
                    )
                    if lm_2:
                        wer_ppl_results_2[f"{name} + {name_2}" if two_pass_same_lm else name_2] = (
                            ppl_results_2.get(name_2), wer_result_path, search_error, search_error_rescore, lm_tune,
                            prior_tune,
                            dafault_lm_scale,
                            dafault_prior_scale)
                        wer_results[f"{name} + {name_2}" if two_pass_same_lm else name_2] = output_dict
                    else:
                        if lm:  # First pass with a lm
                            wer_ppl_results_2[name] = (
                                ppl_results_2.get(name), wer_result_path, search_error, search_error_rescore, lm_tune,
                                prior_tune, dafault_lm_scale,
                                dafault_prior_scale)
                            wer_results[name] = output_dict
                        else:  # NoLM at all := Uniform LM.?
                            #if not word_ppl:
                            wer_ppl_results_2["uniform"] = (
                                ppl_results_2.get("uniform"), wer_result_path, search_error, search_error_rescore,
                                None,
                                None, 0, 0)
                            wer_results["uniform"] = output_dict
                    if break_flag:  # Ensure for lm do first pass not do second pass multiple times
                        break
                # print(wer_ppl_results_2)
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

                summaryjob = WER_ppl_PlotAndSummaryJob(names, results, lm_tunes, prior_tunes, search_errors,
                                                       search_errors_rescore, dafault_lm_scales, dafault_prior_scales,
                                                       eval_dataset_keys=EVAL_DATASET_KEYS)
                gnuplotjob = GnuPlotJob(summaryjob.out_summary, EVAL_DATASET_KEYS, curve_point=73)
                llm_related_name_ext = f"{'prompted' if LLM_WITH_PROMPT else ''}{'_eg' if LLM_WITH_PROMPT_EXAMPLE else ''}_LLMs" + ((f"ctx{LLM_FXIED_CTX_SIZE}" if LLM_FXIED_CTX else "") + (
                    f"prev_{CTX_LEN_LIMIT}ctx" if LLM_PREV_ONE_CTX else "")) if "LLM" in lm_kinds_2 else ""
                alias_prefix = (
                        f"wer_ppl/{f'1st_pass_{name}'}2rd_pass{len(rescor_lms)}_" + model_name + "_" + vocab + encoder
                        + ("n_best_cheat" if CHEAT_N_BEST else "")
                        + llm_related_name_ext + (f"Beam_{BEAM_SIZE}_{NBEST}_best"))
                tk.register_output(alias_prefix + "/report_summary", GetOutPutsJob(outputs=wer_results).out_report_dict)
                tk.register_output(alias_prefix + "/summary", summaryjob.out_summary)
                for i, key in enumerate(EVAL_DATASET_KEYS):
                    tk.register_output(alias_prefix + f"/{key}.png", summaryjob.out_plots[i])
                    tk.register_output(alias_prefix + f"/gnuplot/{key}.pdf", gnuplotjob.out_plots[key])

        def first_pass_with_lm_exp(exp, model_name, lms, rescor_lms, lm_kinds, lm_kinds_2):
            lms.pop("NoLM",None)
            nonlocal vocab, encoder, train, ppl_results_2, word_ppl
            wer_results = dict()
            for name, lm in lms.items():  # First pass lms
                # Do once one pass
                one_pass_res = exp(
                    name, lm, vocab,
                    encoder=encoder, train=train,
                    lm_vocab="spm10k" if "spm10k" in name else None,
                    model=model,
                    model_config=model_config,
                    vocab_config=vocab_config,
                    prior_file=PRIOR_PATH[(vocab, "ctc", encoder)],
                    i6_models=i6_models,
                )
                break_flag = False
                wer_ppl_results_2 = dict()
                two_pass_same_lm = False
                for name_2, lm_2 in rescor_lms.items():  # Second pass lms
                    two_pass_same_lm = False
                    one_pass_lm = name_2
                    if name_2 == name:
                        wer_ppl_results_2[name_2] = (
                            ppl_results_2.get(name_2), *one_pass_res)
                        wer_results[one_pass_lm] = one_pass_res[-5]
                        two_pass_same_lm = True
                    print(name, name_2)
                    (wer_result_path, search_error, search_error_rescore, lm_tune, prior_tune, output_dict,rescor_ppls,
                     lm_hyperparamters_str, dafault_lm_scale, dafault_prior_scale) = exp(
                        name, lm, vocab,
                        rescore_lm=lm_2,
                        rescore_lm_name=name_2,
                        encoder=encoder, train=train,
                        lm_vocab="spm10k" if "spm10k" in name_2 else None,
                        model=model,
                        model_config=model_config,
                        vocab_config=vocab_config,
                        prior_file=PRIOR_PATH[(vocab, "ctc", encoder)],
                        i6_models=i6_models,
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
                if not two_pass_same_lm:
                    wer_ppl_results_2[name] = (
                        ppl_results.get(name), *one_pass_res)
                    wer_results[name] = one_pass_res[-5]
                # print(wer_ppl_results_2)
                # if lm:
                #     wer_ppl_results[name] = (
                #     ppl_results.get(name), wer_result_path, search_error, lm_tune, prior_tune, dafault_lm_scale,
                #     dafault_prior_scale)
                if wer_ppl_results_2 and not train:
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
                                                           aggregated=AGGREGATED_WER,
                                                           eval_dataset_keys=EVAL_DATASET_KEYS)
                    gnuplotjob = GnuPlotJob(summaryjob.out_summary, EVAL_DATASET_KEYS, curve_point=cuts[encoder])
                    llm_related_name_ext = f"{'cheat_ctx' if CHEAT_CTX else ''}" + f"{'prompted' if LLM_WITH_PROMPT else ''}{'_eg' if LLM_WITH_PROMPT_EXAMPLE else ''}_LLMs" + ((f"ctx{LLM_FXIED_CTX_SIZE}" if LLM_FXIED_CTX else "") + (
                        f"prev_{CTX_LEN_LIMIT}ctx" if LLM_PREV_ONE_CTX else "")) if "LLM" in lm_kinds_2 else ""
                    alias_prefix = (
                            f"ES_wer_ppl/{seg_key}_{f'1st_pass_{name}'}2rd_pass{len(rescor_lms)}_" + model_name + "_" + vocab + encoder
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
            if greedy_first_pass:
                greedy_first_pass_exp(exp, model_name, lms, rescor_lms, lm_kinds, lm_kinds_2)
            else:
                first_pass_with_lm_exp(exp, model_name, lms, rescor_lms, lm_kinds, lm_kinds_2)
            tk.register_output("/tools/debugDummy_time", DummyJob().output)

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
