"""
CTC experiments on LBS datasets(refactored).
"""

from __future__ import annotations

import re
import os
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING, List, Union
import os
from i6_experiments.users.zhang.experiments.lm_getter import build_all_lms
from i6_experiments.users.schmitt.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
from i6_experiments.users.zhang.experiments.llm_postfix.error_correction import LLMECConfig, TapExample
from returnn_common.datasets_old_2022_10.interface import VocabConfig
from sisyphus import tk
from dataclasses import dataclass

from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob, GnuPlotJob
#from .lm_getter import build_all_lms, build_ffnn_lms  # NEW

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe

PLOT = False
# --- Decoding Parameters ---
USE_flashlight_decoder = False
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
# -----------------Error Correction related-----------------
TASK_instruct = "EC" # "EC"
N_expand = 5
REJECTION_RATIO = 0.25 # Length ratio for heuristic rejection
STRATEGY = "nbest_reason_rewrite"  # "top1_only" Only correct the top1 or "nbest_reason_rewrite"
NBEST_K = 5  # Only considered when use "nbest_reason_rewrite" strategy
CONTEXT_MODE = "none"  # "none" | "prev_top1" | "prev_corrected"
USE_TAP = False and STRATEGY == "nbest_reason_rewrite"
CONTEXT_WINDOW = 50
PROMPT_LANG = "EN"
FEW_SHOT = True
JSON_OUT = False or (FEW_SHOT and TASK_instruct == "Nbest_expand")
NAME_FOCUSED = True
DEFAULT_PROMPT = True
TAP_EXAMPLE = TapExample(
    hyps=[
        "i could not believe that he would come back again",
        "i could not believe he would come back again",
        "i could not believe that he will come back again",
        "i could not have believed that he would come back again",
        "i could not believe that he could come back again",
    ],
    ref="i could not have believed that he would come back again",
    domain="english read speech (librispeech-like)",
)

# ----------------------------------------------------------------------
# Few-shot examples for error correction (single hyp or n-best)
# ----------------------------------------------------------------------

def get_ec_examples(nbest: bool = False, json: bool = False, top_k: int = 3) -> str:
    """
    Few-shot examples for ASR error correction (English only).

    nbest: if True, use n-best style examples; otherwise single-hyp examples
    json:  if True, wrap inputs in {"text": "..."}
    top_k: for n-best examples, number of hypotheses to show (max limited by template)
    input: {f'{"text": "they{apostrophe}re going to meat us at the station"}' if json else f"they{apostrophe}re going to meat us at the station"}
output: {'{"text": "they are going to meet us at the station"}' if json else "they are going to meet us at the station"}
    """
    header = "Below are some examples for your task:\nExamples:\n"
    apostrophe = "'"
    # Single-hyp examples
    if not nbest:
        ex = f"""{header}
input: {'{"text": "eye have never scene anything like this befor"}' if json else "eye have never scene anything like this befor"}
output: {'{"text": "i have never seen anything like this before"}' if json else "i have never seen anything like this before"}
"""
        return ex

    # N-best examples: each hyp has some correct parts, final output merges the plausible parts

    hyps1 = """
1. we need to book the ticket for tomorrow mourning
2. we need to look the tickets for tomorrow morning
3. we will need to book the ticket for tomorrow morning
4. we need to book the tickets for the moral morning
5. we need to book tickets for tomorrow morning
""".strip("\n").split("\n")[:top_k]

    hyps2 = """
1. the conference will start at nine thirty on monday
2. the conference starts at nine thirty on monday morning
3. the conference will start at nine thirteen on monday morning
4. the conference will start at nine thirty monday morning
5. the conferences will start at nine thirty on monday morning
""".strip("\n").split("\n")[:top_k]

    hyps1_str = "\n".join(hyps1)
    hyps2_str = "\n".join(hyps2)

    ex_nbest = f"""{header}
input:
{hyps1_str}
output:
we need to book the tickets for tomorrow morning

input:
{hyps2_str}
output:
the conference will start at nine thirty on monday morning
"""
    return ex_nbest

# ----------------------------------------------------------------------
# Few-shot examples for N-best expansion (English only)
# ----------------------------------------------------------------------

def get_nbest_expand_examples(json: bool = False, top_k: int = 5) -> str:
    """
    Few-shot examples for N-best expansion (English only).

    json:  if True, wrap input as {"text": "..."} and output as {"expanded_nbest": [...]}
    top_k: number of expanded hypotheses to show per example (max 5)
    """
    header = "Below are some examples for your task:"
    inp = "input"
    out = "output"
    expanded_key = "expanded_nbest"

    # Base ASR hypotheses (LibriSpeech-like style)
    base1 = "i have to go to boston tomorrow morning"
    base2 = "i am not sure if we will be able to arrive on time"
    base3 = "the situation in the meeting was quite complicated"

    # Expanded variants (5 per example)
    exp1 = [
        "i have to go to boston tomorrow morning",
        "i have to go to boston tomorrow evening",
        "i have to go to boston early tomorrow morning",
        "i have to go to boston tomorrow very early",
        "i have to go to boston tomorrow morning again",
    ]
    exp2 = [
        "i am not sure if we will be able to arrive on time",
        "i am not sure whether we will be able to arrive on time",
        "i am not sure if we will be able to get there on time",
        "i am not sure if we will be able to arrive just on time",
        "i am not sure if we will be able to arrive on time today",
    ]
    exp3 = [
        "the situation in the meeting was quite complicated",
        "the situation in the meeting was very complicated",
        "the situation at the meeting was quite complicated",
        "the situation in the meeting seemed complicated",
        "the situation in the meeting was really complicated",
    ]

    exp1 = exp1[:top_k]
    exp2 = exp2[:top_k]
    exp3 = exp3[:top_k]

    def fmt_input(text: str) -> str:
        return f'{{"text": "{text}"}}' if json else text

    def fmt_output(lst: List[str]) -> str:
        if json:
            quoted = ", ".join([f'"{x}"' for x in lst])
            return f'{{"{expanded_key}": [{quoted}]}}'
        else:
            return "\n".join(f"- {x}" for x in lst)

    ex = f"""{header}

{inp}: {fmt_input(base1)}
{out}:
{fmt_output(exp1)}

{inp}: {fmt_input(base2)}
{out}:
{fmt_output(exp2)}

{inp}: {fmt_input(base3)}
{out}:
{fmt_output(exp3)}
"""
    return ex

# ----------------------------------------------------------------------
# System prompts (English only)
# ----------------------------------------------------------------------

def get_ec_system_prompt(
    few_shot: bool = False,
    names_only: bool = False,
    json: bool = False,
    nbest: bool = False,
    top_k: int = 3,
) -> str:
    """
    System prompt for error correction on English ASR (LibriSpeech).
    """
    core = (
        "You are a second-pass transcriber that is given the first-pass transcription "
        "of an acoustic utterance in English. Please fix any transcription errors of "
        "the first-pass transcriber to minimize the edit distance to the unknown "
        "reference transcription. The original was spoken, so do not correct "
        "disfluencies in the text and "
        f"{'only correct' if names_only else 'rather focus on correcting'} proper names. "
        "Write only the updated sentence without any additional comments. "
        "Write only lowercased words without punctuation except apostrophe.\n\n"
    )

    if few_shot:
        core += get_ec_examples(nbest=nbest, json=json, top_k=top_k)
    return core


def get_nbest_expand_system_prompt(
    few_shot: bool = False,
    json: bool = False,
    top_k: int = 5,
) -> str:
    """
    System prompt for N-best expansion on English ASR.
    """
    core = """
You generate alternative hypotheses for an ASR segment in English.
Your goal is to enrich and diversify the n-best list without drifting away
from the original meaning or acoustic plausibility.

Rules:
- Use the input hypothesis as the main anchor.
- Produce alternative hypotheses that could realistically appear in an ASR n-best list.
- Each hypothesis must differ slightly from the others (wording, small substitutions, reorderings).
- Preserve plausible phonetic similarity; avoid adding new semantic content.
- Keep everything in lowercase.
- No punctuation.
- No explanations.

Output format:
Return your output using Python dict syntax:

{"expanded_nbest": ["hypothesis 1", "hypothesis 2", "hypothesis 3"]}
""".strip() + "\n\n"

    if few_shot:
        core += get_nbest_expand_examples(json=json, top_k=top_k)
    return core

ENGLISH_EC_SYSTEM_PROMPT = get_ec_system_prompt(few_shot=True, names_only=False, json=False, nbest=False, top_k=3)
# -----------------Search config-----------------
trans_only_LM = False

CHEAT_N_BEST = False
TUNE_WITH_CHEAT = False and CHEAT_N_BEST
TUNE_TWO_ROUND = True

BEAM_SIZE = 80
FFNN_BEAM_SIZE = 300
TRAFO_BEAM = 80
NBEST = 80 # Use 100 for plot


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
CHEAT_CTX = True and LLM_PREV_ONE_CTX
CTX_LEN_LIMIT = 100

TUNE_LLM_SCALE_EVERY_PASS = LLM_PREV_ONE_CTX and not CHEAT_CTX

DIAGNOSE = True and not TUNE_LLM_SCALE_EVERY_PASS

# --- Helpers for ctc_exp ---

def get_decoding_config(lmname: str, lm, vocab: str, encoder: str, nbest: int =50, beam_size: int=80) -> Tuple[dict, dict, dict, dict, bool, Optional[int]]:
    if nbest:
        assert beam_size >= nbest
    decoding_config = {
        #"log_add": False, #Flashlight
        "nbest": nbest,
        "beam_size": beam_size,
        #"beam_threshold": 1e6, #Flashlight
        "lm_weight": 1.45,
        "use_logsoftmax": True,
        "use_lm": False,
        #"use_lexicon": False, #Flashlight
        "vocab": get_vocab_by_str(vocab),
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
        decoding_config["beam_size"] = TRAFO_BEAM if encoder == "conformer" else 300
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

    alias_name = f"LBS-ctc-baseline_{encoder}_decodingWith_1st-{lmname}_{lm_hyperparamters_str}_{'LMTune' if not TUNE_ON_GREEDY_N_LIST else ''}_2rd{lm2_hyperparamters_str}"

    EC_config = decoding_config.get("EC_config",None)
    EC_model_name = EC_config.model_name if EC_config else None
    alias_name += f"_{TASK_instruct}_with_{os.path.basename(EC_model_name)}" if EC_model_name else ""
    first_pass_name = f"LBS-ctc-baseline_{encoder}_decodingWith_{lm_hyperparamters_str}_{lmname}_{'LMTune' if not TUNE_ON_GREEDY_N_LIST else ''}"
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

def ctc_exp(
    lmname,
    lm,
    vocab,
    *,
    lm_vocab: Optional[Bpe : SentencePieceModel] = None,
    rescore_lm: Optional[ModelWithCheckpoint, dict] = None,
    rescore_lm_name: str = None,
    EC_config: Optional[LLMECConfig] = None,
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
    if lm_vocab is None:
        lm_vocab = vocab
    (
        decoding_config,
        tune_config_updates,
        recog_config_updates,
        search_rqmt,
        tune_hyperparameters,
        batch_size,
    ) = get_decoding_config(lmname, lm, vocab, encoder, beam_size=BEAM_SIZE, nbest=NBEST)

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

    if EC_config:
        decoding_config["EC_config"] = EC_config
    if lm is not None:  # First pass with a LM, tuned it with rescoring
        decoding_config["tune_with_rescoring"] = True# and not (USE_GIVEN_NBEST and not COMBINE_NLIST) # Set to false if do one pass tuning
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
@dataclass
class SummaryEntry:
    """
    One row that goes into the WER/PPL summary.
    Mirrors the tuple structure you previously had:
    (ppl, wer_result_path, search_error, search_error_rescore,
     lm_tune, prior_tune, default_lm_scale, default_prior_scale)
    """
    ppl: Dict[str, Union[tk.Variable, float]]  # or whatever type ppl_results[_] has
    wer_result_path: str
    search_error: float
    search_error_rescore: float
    lm_tune: Optional[dict]
    prior_tune: Optional[dict]
    default_lm_scale: float
    default_prior_scale: float
    miscs: Dict[str, Any]


def build_ec_configs() -> List[Tuple[str, Optional[LLMECConfig]]]:
    """
    Build all Error Correction (EC) configs.
    Returns list of (ec_name, EC_config_or_None).
    """
    import math
    reduce_offset = (2 if FEW_SHOT else 0) if STRATEGY == "top1_only" else math.ceil(1.5*NBEST_K*(2.2 if FEW_SHOT else 1))
    reduce_offset *= 2 if FEW_SHOT and TASK_instruct == "Nbest_expand" else 1
    EC_LLMs_Batch_size = {
        "meta-llama/Llama-3.2-3B-Instruct": ((80 if TASK_instruct == "EC" else 60) if DEFAULT_PROMPT else 60) - reduce_offset,
        "Qwen/Qwen2.5-3B-Instruct": (70 if DEFAULT_PROMPT else 55) - reduce_offset,
        "meta-llama/Llama-3.1-8B-Instruct": ((100 if TASK_instruct == "EC" else 80) if DEFAULT_PROMPT else 80) - reduce_offset,
        "Qwen/Qwen3-4B-Instruct-2507": (60 if DEFAULT_PROMPT else 45) - reduce_offset,
        #"Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4": (70 if USE_TAP else 200) - reduce_offset,
    }
    from i6_experiments.users.zhang.experiments.llm_postfix.error_correction import get_model_size_and_quant

    EC_configs: List[Tuple[str, Optional[LLMECConfig]]] = [
        (
            os.path.basename(model_id),
            LLMECConfig(
                task=TASK_instruct,
                N_expand=N_expand,
                provider="hf",
                expect_json=JSON_OUT,
                model_name=model_id,
                device="auto",
                dtype="bfloat16",
                system_prompt=get_ec_system_prompt(few_shot=FEW_SHOT, names_only=NAME_FOCUSED, json=JSON_OUT,
                                                   nbest=STRATEGY == "nbest_reason_rewrite", top_k=NBEST_K),
                prompt_lang=PROMPT_LANG,
                tap_examples=[TAP_EXAMPLE] if USE_TAP and get_model_size_and_quant(os.path.basename(model_id))[
                    0] > 10 else None,
                strategy=STRATEGY,             # Only correct the top1
                nbest_k=NBEST_K,               # Only considered when use "nbest_reason_rewrite" strategy
                order_by_score=True,
                rejection_ratio=REJECTION_RATIO,
                score_policy="keep_top1",
                name_focused=NAME_FOCUSED,
                json_key='expanded_nbest',
                context_mode=CONTEXT_MODE,     # "none" | "prev_top1" | "prev_corrected"
                context_window=CONTEXT_WINDOW,
                keep_apostrophe=True,
                hf_batch_size=EC_LLMs_Batch_size[model_id],
                max_new_tokens=128 if TASK_instruct == "EC" else 1024,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,       # HF only
            ),
        )
        for model_id in EC_LLMs_Batch_size.keys()
    ]

    # Remote-API EC LLMs
    EC_LLMs_hf_api = {
        # "meta-llama/Llama-3.3-70B-Instruct",
        # "Qwen/Qwen2.5-72B-Instruct",
    }
    EC_configs += [
        (
            os.path.basename(model_id),
            LLMECConfig(
                provider="hf_api",
                model_name=model_id,
                device="cpu",
                dtype="bfloat16",
                system_prompt=get_ec_system_prompt(few_shot=FEW_SHOT, names_only=NAME_FOCUSED, json=JSON_OUT, nbest=STRATEGY=="nbest_reason_rewrite", top_k=NBEST_K),
                prompt_lang=PROMPT_LANG,
                strategy=STRATEGY,         # Only correct the top1
                tap_examples=[TAP_EXAMPLE] if USE_TAP and get_model_size_and_quant(os.path.basename(model_id))[0] > 10 else None,
                nbest_k=NBEST_K,
                order_by_score=True,
                score_policy="keep_top1",
                context_mode="none",       # Better do not use context for remote api
                context_window=CONTEXT_WINDOW,
                max_new_tokens=128,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
            ),
        )
        for model_id in EC_LLMs_hf_api
    ]

    # Baseline "NoLM" EC config (no error correction)
    EC_configs.append(("NoLM", None))
    return EC_configs


def build_llm_configs_for_rescoring(vocab_config, lm_kinds, lm_kinds_2, word_ppl, trans_only_LM, task_name):
    """
    Build the LM sets (first pass and rescoring) and their PPL result dicts.
    Also inject "NoLM" and uniform baselines.
    """
    # LLMs for rescoring (non-EC)
    reduce_offset = max(0, 10 * (CTX_LEN_LIMIT // 100 - 1)) if LLM_PREV_ONE_CTX else -20
    LLM_and_Batch_size = {
        "meta-llama/Llama-3.2-1B": 40 - reduce_offset,
        "meta-llama/Llama-3.1-8B": 40 - reduce_offset,
        # "Qwen/Qwen3-1.7B-Base": 40 - reduce_offset,
        # "microsoft/phi-4": 40 - reduce_offset,
    }

    # First-pass LMs
    lms, ppl_results, _ = build_all_lms(
        vocab_config,
        lm_kinds=lm_kinds,
        word_ppl=word_ppl,
        only_best=True,
        only_transcript=trans_only_LM,
        task_name=task_name,
    )
    lms.update({"NoLM": None})

    # Second-pass / rescoring LMs
    rescor_lms, ppl_results_2, _ = build_all_lms(
        vocab_config,
        lm_kinds=lm_kinds_2,
        as_ckpt=True,
        word_ppl=word_ppl,
        only_best=not PLOT,
        only_transcript=trans_only_LM,
        task_name=task_name,
        llmids_batch_sizes=LLM_and_Batch_size,
    )
    rescor_lms.update({"NoLM": None})

    # Uniform baseline for rescoring (used e.g. by EC)
    ppl_results_2.update({"uniform": {k: 10240.0 for k in EVAL_DATASET_KEYS}})

    return lms, rescor_lms, ppl_results, ppl_results_2


def build_llm_name_suffix() -> str:
    """
    Build the suffix related to LLM usage, context, EC, etc.
    This is used in alias_prefix and makes naming more systematic.
    """
    llm_suffix = ""
    if "LLM" in LM_KINDS_2_GLOBAL:  # small hack: if you want you can pass this in as arg instead
        llm_suffix = (
            f"{'cheat_ctx' if CHEAT_CTX else ''}"
            f"{'prompted' if LLM_WITH_PROMPT else ''}"
            f"{'_eg' if LLM_WITH_PROMPT_EXAMPLE else ''}_LLMs"
        )
        if LLM_FXIED_CTX:
            llm_suffix += f"ctx{LLM_FXIED_CTX_SIZE}"
        if LLM_PREV_ONE_CTX:
            llm_suffix += f"prev_{CTX_LEN_LIMIT}ctx"
    llm_suffix += (f"{TASK_instruct}" + f"{'few_shot' if FEW_SHOT else ''}"
                   + f"{'_top1_only' if STRATEGY == 'top1_only' else f'_{NBEST_K}_best'}"
                   + f"{f'_{N_expand}expands' if TASK_instruct == 'Nbest_expand' else f''}"
                   + f"{CONTEXT_MODE if CONTEXT_MODE != 'none' else ''}"
                   + f"lenthres{REJECTION_RATIO}".replace('.','_')
                   + f"_{PROMPT_LANG}{'TAP' if USE_TAP else ''}" + f"{'_default_prompt' if DEFAULT_PROMPT else ''}"
                   + f"{'_name_focused' if NAME_FOCUSED and DEFAULT_PROMPT else ''}"
                   )
    return llm_suffix


def create_summary_and_register(
    wer_ppl_results: Dict[str, SummaryEntry],
    wer_results: Dict[str, dict],
    encoder: str,
    model_name: str,
    vocab: str,
    eval_dataset_keys,
):
    """
    Build WER_ppl_PlotAndSummaryJob input from collected SummaryEntry rows
    and register outputs. This now aggregates over *all* EC configs.
    """
    if not wer_ppl_results:
        return

    names = list(wer_ppl_results.keys())
    entries = list(wer_ppl_results.values())
    miscs = [e.miscs for e in entries]
    ppl_list = [e.ppl for e in entries]
    wer_result_paths = [e.wer_result_path for e in entries]

    search_errors = [e.search_error for e in entries]
    search_errors_rescore = [e.search_error_rescore for e in entries]
    lm_tunes = [e.lm_tune for e in entries]
    prior_tunes = [e.prior_tune for e in entries]
    default_lm_scales = [e.default_lm_scale for e in entries]
    default_prior_scales = [e.default_prior_scale for e in entries]

    summaryjob = WER_ppl_PlotAndSummaryJob(
        names,
        list(zip(ppl_list, wer_result_paths)),
        lm_tunes,
        prior_tunes,
        search_errors,
        search_errors_rescore,
        default_lm_scales,
        default_prior_scales,
        eval_dataset_keys=eval_dataset_keys,
        misc=miscs,
    )

    # Suffix for LLM/context etc, still centralised
    llm_suffix = build_llm_name_suffix()

    # Now: one alias that covers *all* EC configs together.
    alias_prefix = (
        f"LBS_wer_ppl/"
        f"1st_pass_{model_name}"
        f"2rd_pass_{len(names)}_"
        f"{model_name}_{vocab}{encoder}"
        f"{'n_best_cheat' if CHEAT_N_BEST else ''}"
        f"{llm_suffix}"
        f"Beam_{BEAM_SIZE}_{NBEST}_best"
        "_ECagg"  # indicate that this is aggregated over EC configs
    )

    summaryjob.add_alias(alias_prefix + "/summary_job")
    scoring_summaryjob = GetOutPutsJob(outputs=wer_results)
    scoring_summaryjob.add_alias(alias_prefix + "/scorer_summary_job")
    tk.register_output(
        alias_prefix + "/report_summary",
        scoring_summaryjob.out_report_dict,
    )
    tk.register_output(alias_prefix + "/wers", summaryjob.out_tables["wers"])

    # for key in eval_dataset_keys:
    #     tk.register_output(
    #         alias_prefix + f"/gnuplot/{key}.pdf", gnuplotjob.out_plots[key]
    #     )
    #     tk.register_output(
    #         alias_prefix + f"/gnuplot/{key}_regression",
    #         gnuplotjob.out_equations[key],
    #     )


def run_first_and_second_pass(
    exp,
    model_name: str,
    vocab: str,
    encoder: str,
    train: bool,
    lms: Dict[str, object],
    rescor_lms: Dict[str, object],
    ppl_results: Dict[str, dict],
    ppl_results_2: Dict[str, dict],
    ec_configs: List[Tuple[str, Optional[LLMECConfig]]],
) -> None:
    """
    Perform first-pass + second-pass decoding and aggregate results over:
      - all first-pass LMs
      - all rescore LMs
      - all EC configs

    Everything ends up in one wer_ppl_results / wer_results set,
    so the summary job naturally aggregates across EC configs.
    """
    # shallow copy so we can mutate safely
    lms = dict(lms)
    # do not use "NoLM" as first-pass LM
    lms.pop("NoLM", None)

    wer_ppl_results: Dict[str, SummaryEntry] = {}
    wer_results: Dict[str, dict] = {}

    def pure_greedy():
        greedy_res = exp(
            "NoLM",
            None,
            vocab,
            encoder=encoder,
            train=train,
            lm_vocab=None,
        )
        wer_ppl_results['Greedy'] = SummaryEntry(
            ppl={k:10024.0 for k in EVAL_DATASET_KEYS},
            wer_result_path=greedy_res[0],
            search_error=greedy_res[1],
            search_error_rescore=greedy_res[2],
            lm_tune=greedy_res[3],
            prior_tune=greedy_res[4],
            miscs=greedy_res[7],
            default_lm_scale=greedy_res[-2],
            default_prior_scale=greedy_res[-1],
        )
        wer_results['Greedy'] = greedy_res[-6]
    pure_greedy()
    for first_name, first_lm in lms.items():
        # ---- First pass (no EC) ----
        one_pass_res = exp(
            first_name,
            first_lm,
            vocab,
            encoder=encoder,
            train=train,
            lm_vocab="spm10k" if "spm10k" in first_name else None,
        )

        # store pure first-pass result
        wer_ppl_results[first_name] = SummaryEntry(
            ppl=ppl_results.get(first_name),
            wer_result_path=one_pass_res[0],
            search_error=one_pass_res[1],
            search_error_rescore=one_pass_res[2],
            lm_tune=one_pass_res[3],
            prior_tune=one_pass_res[4],
            miscs=one_pass_res[7],
            default_lm_scale=one_pass_res[-2],
            default_prior_scale=one_pass_res[-1],
        )
        wer_results[first_name] = one_pass_res[-6]

        # ---- Second pass + EC sweep ----
        for rescore_name, rescore_lm in rescor_lms.items():
            for ec_name, ec_cfg in ec_configs:
                if ec_cfg is not None and rescore_lm is not None:
                    if ec_cfg.task == "EC": # Do not combine EC with rescoring
                        continue

                print(first_name, rescore_name, ec_name)

                (
                    wer_result_path,
                    search_error,
                    search_error_rescore,
                    lm_tune,
                    prior_tune,
                    output_dict,
                    rescor_ppls,
                    miscs,
                    lm_hyperparamters_str,
                    default_lm_scale,
                    default_prior_scale,
                ) = exp(
                    first_name,
                    first_lm,
                    vocab,
                    rescore_lm=rescore_lm,
                    rescore_lm_name=rescore_name,
                    EC_config=ec_cfg,
                    encoder=encoder,
                    train=train,
                    lm_vocab="spm10k" if "spm10k" in rescore_name else None,
                )

                # ----- naming + PPL source -----
                if rescore_lm is not None:
                    # LM rescoring (with or without EC, though EC+LM currently skipped by logic above)
                    base_key = (
                        f"{first_name} + {rescore_name}"
                        if rescore_name == first_name
                        else rescore_name
                    )
                    if ec_cfg is not None:
                        key_name = f"{base_key} + Nbest_expand_{ec_name}"
                    else:
                        key_name = base_key

                    ppl_src = (
                        ppl_results_2.get(rescore_name)
                        if not rescor_ppls or CHEAT_CTX
                        else rescor_ppls
                    )
                else:
                    # NoLM second pass: uniform ppl, usually EC-driven
                    if ec_cfg is None:
                        key_name = "NoLM"
                    else:
                        key_name = f"EC_{ec_name}"

                    ppl_src = ppl_results_2.get("uniform")

                wer_ppl_results[key_name] = SummaryEntry(
                    ppl=ppl_src,
                    wer_result_path=wer_result_path,
                    search_error=search_error,
                    search_error_rescore=search_error_rescore,
                    lm_tune=lm_tune,
                    prior_tune=prior_tune,
                    miscs=miscs,
                    default_lm_scale=default_lm_scale,
                    default_prior_scale=default_prior_scale,
                )
                wer_results[key_name] = output_dict

    # ---- Final summary (aggregated over all EC configs) ----
    if not train:
        create_summary_and_register(
            wer_ppl_results=wer_ppl_results,
            wer_results=wer_results,
            encoder=encoder,
            model_name=model_name,
            vocab=vocab,
            eval_dataset_keys=EVAL_DATASET_KEYS,
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
    global CUTS
    CUTS = {"conformer": 65, "blstm": 37}

    for vocab in ["bpe128"]:
        word_ppl = True
        lm_kinds = {"trafo",
                    #"trafo_n24d1280_rope_ffgated",
                    }
        lm_kinds_2 = {
            "LLM",
            #"trafo",
        }  # adapt as needed
        if TASK_instruct == "EC":
            lm_kinds_2 = {}
        # for build_llm_name_suffix helper
        global LM_KINDS_2_GLOBAL
        LM_KINDS_2_GLOBAL = lm_kinds_2

        lms, rescor_lms, ppl_results, ppl_results_2 = build_llm_configs_for_rescoring(
            vocab,
            lm_kinds=lm_kinds,
            lm_kinds_2=lm_kinds_2,
            word_ppl=word_ppl,
            trans_only_LM=trans_only_LM,
            task_name="LBS",
        )

        ec_configs = build_ec_configs()  # includes "NoLM" EC baseline etc.

        for model_name, exp in models.items():
            if (vocab, model_name, encoder) not in available:
                train = True

            run_first_and_second_pass(
                exp=exp,
                model_name=model_name,
                vocab=vocab,
                encoder=encoder,
                train=train,
                lms=lms,
                rescor_lms=rescor_lms,
                ppl_results=ppl_results,
                ppl_results_2=ppl_results_2,
                ec_configs=ec_configs,
            )