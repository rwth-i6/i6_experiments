"""
CTC experiments on Apptek datasets(refactored).
"""

from __future__ import annotations

import re
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING, List, Union
import os
from i6_experiments.users.zhang.experiments.lm_getter import build_all_lms
from i6_experiments.users.schmitt.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
from i6_experiments.users.zhang.experiments.llm_postfix.error_correction import LLMECConfig, TapExample
from i6_experiments.users.zhang.utils.tools import DummyJob
from returnn_common.datasets_old_2022_10.interface import VocabConfig
from sisyphus import tk
from dataclasses import dataclass
from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob, GnuPlotJob
#from .lm_getter import build_all_lms, build_ffnn_lms  # NEW

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
    from i6_core.serialization.base import ExternalImport

RETURNN_ROOT = "/home/mgunz/setups/2024-07-08--zeyer-setup-apptek/recipe/returnn" #"/nas/models/asr/hzhang/setups/2025-07-20--combined/returnn"
FINE_TUNED_MODEL = True # If use the FT model
CKPT_EPOCH = 25 if FINE_TUNED_MODEL else 625
# --- Decoding Parameters ---
USE_flashlight_decoder = False
from i6_experiments.users.zhang.experiments.apptek_exp_wer_ppl import seg_key
#seg_key = "ref" #aptk_leg ref
PLOT = False
DEV = False
EXCLUDE_LIST = ["napoli", "callcenter", "voice_call", "tvshows", "mtp_eval-v2"]

DEV_DATASET_KEYS = [f"test_set.ES_ES.f8kHz.mtp_eval-v2.{seg_key}.ff_wer"] + [f"{key}.{seg_key}.ff_wer" for key in DEV_KEYS if "callhome" not in key]# or seg_key == "ref"] #if "conversation" not in key] #Evaluate on concatenated DEV_KEYS-> not implemented
DEV_DATASET_KEYS_FIRST_PASS = [f"test_set.ES_ES.f8kHz.mtp_eval-v2.{seg_key}.ff_wer"] + [f"{key}.{seg_key}.ff_wer" for key in DEV_KEYS if "callhome" not in key]
EVAL_DATASET_KEYS = [f"{key}.{seg_key}.ff_wer" for key in TEST_KEYS if "mtp_eval-v2" not in key] #+ DEV_DATASET_KEYS + ["test_set.ES_ES.f16kHz.eval_voice_call-v3.ref.ff_wer", "test_set.ES_US.f16kHz.dev_conversations_202411-v2.ref.ff_wer"]#[f"{key}.ref.ff_wer" for key in DEV_KEYS + TEST_KEYS]#['test_set.ES.f8kHz.mtp_dev_heldout-v2.aptk_leg.ff_wer', 'test_set.ES.f8kHz.mtp_dev_heldout-v2.ref.ff_wer'] #
EVAL_DATASET_KEYS = [f"{key}.{seg_key}.ff_wer" for key in TEST_KEYS if not any(exclude in key for exclude in EXCLUDE_LIST)] if PLOT else EVAL_DATASET_KEYS
EVAL_DATASET_KEYS = DEV_DATASET_KEYS if DEV else ([f"{key}.{seg_key}.ff_wer" for key in TEST_KEYS if any(infix in key for infix in ["mtp_eval-v2", "napoli"])]
                                                  #+ DEV_DATASET_KEYS
                                                  )

USE_GIVEN_NBEST = False
COMBINE_NLIST = False and USE_GIVEN_NBEST
TUNE_WITH_ORIG_NBEST = True# and COMBINE_NLIST
if USE_GIVEN_NBEST :
    assert seg_key == "ref"
    UNAVAILABE = ["mtp_eval-v2", "voice_call-v3"] #"common_voice",
    if TUNE_WITH_ORIG_NBEST:
        DEV_DATASET_KEYS = [f"{key}.{seg_key}.ff_wer" for key in DEV_KEYS if "callhome" not in key] + [f"test_set.ES_ES.f8kHz.mtp_eval-v2.{seg_key}.ff_wer"]
    else:
        DEV_DATASET_KEYS = [f"{key}.{seg_key}.ff_wer" for key in DEV_KEYS
                            if "callhome" not in key and "conversation" not in key]  # or seg_key == "ref"] #if "conversation" not in key] #Evaluate on concatenated DEV_KEYS-> not implemented
    EVAL_DATASET_KEYS = [f"{key}.{seg_key}.ff_wer" for key in TEST_KEYS if
                         all(infix not in key for infix in UNAVAILABE)]  # + DEV_DATASET_KEYS + ["test_set.ES_ES.f16kHz.eval_voice_call-v3.ref.ff_wer", "test_set.ES_US.f16kHz.dev_conversations_202411-v2.ref.ff_wer"]#[f"{key}.ref.ff_wer" for key in DEV_KEYS + TEST_KEYS]#['test_set.ES.f8kHz.mtp_dev_heldout-v2.aptk_leg.ff_wer', 'test_set.ES.f8kHz.mtp_dev_heldout-v2.ref.ff_wer'] #
    #EVAL_DATASET_KEYS = [key for key in EVAL_DATASET_KEYS if "tvshow" in key or "napoli" in key] #or "voice_call" in key
AGGREGATED_WER = False and not USE_GIVEN_NBEST and not DEV

DEFAULT_PRIOR_WEIGHT = 0.3
DEFAULT_LM_SCALE_DICT = {"Llama-3.1-8B": 0.35, "Qwen3-1.7B-Base": 0.3, "phi-4": 0.45, "Llama-3.2-1B": 0.3}
DEFAULT_PRIOR_SCALE_DICT = {"Llama-3.1-8B": 0.22, "Qwen3-1.7B-Base": 0.18, "phi-4": 0.26, "Llama-3.2-1B": 0.2}
DEFAULT_PRIOR_TUNE_RANGE = [-0.1, -0.05, 0.0, 0.05, 0.1]
DEFAULT_LM_WEIGHT = 0.5
DEFAUL_RESCOR_LM_SCALE = DEFAULT_LM_WEIGHT # Keep this same, otherwise tune with rescoring will broken

# -----------------Error Correction related-----------------
TASK_instruct = "Nbest_expand" # "EC" Nbest_expand
N_expand = 5
REJECTION_RATIO = 0.1 # Length ratio for heuristic rejection
STRATEGY = "top1_only"  # "top1_only" Only correct the top1 or "nbest_reason_rewrite"
NBEST_K = 5  # Only considered when use "nbest_reason_rewrite" strategy
CONTEXT_MODE = "none"  # "none" | "prev_top1" | "prev_corrected"
USE_TAP = False and STRATEGY == "nbest_reason_rewrite"
CONTEXT_WINDOW = 50
PROMPT_LANG = "ES"
FEW_SHOT = True
JSON_OUT = False or (FEW_SHOT and TASK_instruct == "Nbest_expand")
NAME_FOCUSED = True
DEFAULT_PROMPT = True
TAP_EXAMPLE = TapExample(
    hyps=[
        "déjame a seguir mirando la cuenta tú estás ahí",
        "déjame a seguir mirando la cuenta estás ahí",
        "déjame a ir mirando la cuenta tú estás ahí",
        "déjame a ir mirando la cuenta estás ahí",
        "déjame a seguir mirando la cuenta que estás ahí",
        "déjame que a seguir mirando la cuenta tú estás ahí",
    ],
    ref="déjame ir mirando la página que tú estabas ahí",
    domain="Spanish conversational"
)

def get_example(lang: str = "EN", nbest: bool = False, json: bool = False, top_k: int = 3) -> str:
    example = f"""{'A continuación se muestran algunos ejemplos para su tarea:' if lang == "ES" else "Below are some examples for your task:"}
{'Ejemplos' if lang == "ES" else "Examples"}:

{'entrada' if lang == "ES" else "input"}: {'buenos dias a todoz y todas'}
{'salida' if lang == "ES" else "output"}: {'buenos dias a todos y todas' if not json else '{"text": "buenos dias a todos y todas"}'}

{'entrada' if lang == "ES" else "input"}: {'el real madri a ganado la liga'}
{'salida' if lang == "ES" else "output"}: {'el real madrid ha ganado la liga' if not json else '{"text": "el real madrid ha ganado la liga"}'}
"""
    hyps1 = '\n'.join("""1. hm y a sonia que tal
    2. y a sonia que tal
    3. hm y la sonia que tal
    4. y a sonia que está
    5. mhm y a sonia que tal""".split("\n")[:top_k])
    hyps2 = '\n'.join("""1. buenos días juan que soy clara que que estoy aquí en alemania
    2. buenos días juan soy clara que que estoy aquí en alemania
    3. buenos días juan soy clara que que estoy aquí
    4. buenos días juan soy clara que que estoy en alemania
    5. buenos días juan soy clara que estoy aquí en alemania""".split("\n")[:top_k])
    example_nbest = f"""{'A continuación se muestran algunos ejemplos para su tarea:' if lang == "ES" else "Below are some examples for your task:"}
{'entrada' if lang == "ES" else "input"}:
{hyps1}
{'salida' if lang == "ES" else "output"}: 
hm hm la sonia que está

{'entrada' if lang == "ES" else "input"}:
{hyps2}

{'salida' if lang == "ES" else "output"}:
buenos días juan soy clara que que estoy aquí en alemania
"""
    return example if not nbest else example_nbest

def get_nbest_expand_examples(lang: str = "ES", json: bool = False, top_k: int = 5) -> str:
    """
    Few-shot examples for N-best expansion.
    lang: 'ES' or 'EN'
    json: if True, wrap each input in {"text": "..."}
    top_k: number of expanded hypotheses to show (max 5)
    """
    # Language strings
    if lang == "ES":
        hdr = "A continuación se muestran algunos ejemplos para su tarea:"
        inp = "entrada"
        out = "salida"
        expanded = "expanded_nbest"
    else:
        hdr = "Below are some examples for your task:"
        inp = "input"
        out = "output"
        expanded = "expanded_nbest"

    # Base ASR hypotheses
    base1 = "tengo que ir a madrid mañana por la mañana"
    base2 = "no sé si vamos a poder llegar a tiempo"
    base3 = "la situación en el partido era bastante complicada"

    # Expanded variants (5 per example)
    exp1 = [
        "tengo que ir a madrid mañana por la mañana",
        "tengo que ir a madrid mañana por la tarde",
        "tengo que ir a madrid mañana muy temprano",
        "tengo que ir a madrid por la mañana mañana",
        "tengo que ir a madrid mañana temprano"
    ]
    exp2 = [
        "no sé si vamos a poder llegar a tiempo",
        "no sé si vamos a poder llegar a buen tiempo",
        "no sé si vamos a poder llegar justo a tiempo",
        "no sé si vamos a poder llegar a tiempo hoy",
        "no sé si vamos a poder llegar a tiempo al final"
    ]
    exp3 = [
        "la situación en el partido era bastante complicada",
        "la situación en el partido era muy complicada",
        "la situación del partido era bastante complicada",
        "la situación en el partido se veía complicada",
        "la situación en el partido era realmente complicada"
    ]

    # Select top_k
    exp1 = exp1[:top_k]
    exp2 = exp2[:top_k]
    exp3 = exp3[:top_k]

    def fmt_input(text):
        return f'{{"text": "{text}"}}' if json else text

    def fmt_output(lst):
        if json:
            # Valid JSON with proper quoting
            quoted = ', '.join([f'"{x}"' for x in lst])
            return f'{{"{expanded}": [{quoted}]}}'
        else:
            # Plain text list
            return "\n".join(f"- {x}" for x in lst)

    # Build examples
    ex = f"""{hdr}

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

def get_system_prompt1(names_focus: bool = False, lang: str = "EN", few_shot: bool = False, task: str = "EC") -> str:
    if task == "EC":
        prompt_es = (
            "Eres un transcriptor de segunda pasada que recibe la transcripción de primera pasada de un enunciado acústico. "
            "Por favor corrige cualquier error de transcripción de la primera pasada para minimizar la distancia de edición con la "
            "transcripción de referencia desconocida. Ten en cuenta que el original fue hablado, así que no corrijas las disfluencias "
            f"en el texto y {'en su lugar concéntrate en corregir' if not names_focus else 'corrige únicamente'} los nombres propios. "
            "Escribe solo la oración actualizada sin comentarios adicionales. "
            "Escribe únicamente palabras en minúsculas y sin puntuación\n\n"
        )
        prompt_en = (
                        "You are second pass transcriber that is given the first pass transcription of an acoustic utterance. "
                        "Please fix any transcription errors of the first pass transcriber to minimize the edit distance to the "
                        "unknown reference transcription. Note that the original was spoken, so please do not correct disfluencies "
                        f"in the text and {'rather focus on correcting' if not names_focus else 'only'} proper names. Write only the updated sentence without any additional comments. "
                        "Write only lowercased words without punctuation\n\n"
                    )
        if few_shot:
            return (prompt_en if lang == "EN" else prompt_es) + get_example(lang=lang, json=JSON_OUT, nbest=STRATEGY=='nbest_reason_rewrite')
        else:
            return (prompt_en if lang == "EN" else prompt_es)
    elif task == "Nbest_expand":
        expand_nbest_system_en = """
        You generate alternative hypotheses for an ASR segment.
        Your goal is to enrich and diversify the n-best list without drifting away from the original meaning or acoustic plausibility.

        Rules:
        - Use the input hypothesis as the main anchor.
        - Produce N alternative hypotheses that could realistically appear in an ASR n-best list.
        - Each hypothesis must differ slightly from the others (wording, small substitutions, reorderings).
        - Preserve plausible phonetic similarity; avoid adding new semantic content.
        - Keep everything in lowercase.
        - No punctuation.
        - No explanations.

        Output format:
        Return your output using Python dict syntax:

        {"expanded_nbest": ["hypothesis 1","hypothesis 2","hypothesis 3"]}
        """.strip()
        expand_nbest_system_es = """
        Eres un generador de hipótesis alternativas para un segmento de ASR.
        Tu objetivo es ampliar y diversificar la lista n-best sin alejarte demasiado del contenido ni de la plausibilidad acústica.

        Reglas:
        - Usa la hipótesis dada como base.
        - Genera N hipótesis alternativas que puedan aparecer razonablemente en un n-best real.
        - Cada hipótesis debe diferir ligeramente de las demás (sustituciones mínimas, pequeños cambios de orden, variantes fonéticas plausibles).
        - No añadas información nueva.
        - Mantén todo en minúsculas.
        - Sin puntuación.
        - No des explicaciones.

        Formato de salida:
        Devuelve un JSON:

        {"expanded_nbest": ["hipótesis 1","hipótesis 2","hipótesis 3"]}
        """.strip()
        if few_shot:
            return (expand_nbest_system_en if lang == "EN" else expand_nbest_system_es) + get_nbest_expand_examples(lang=lang, json=JSON_OUT, top_k=N_expand)
        else:
            return (expand_nbest_system_en if lang == "EN" else expand_nbest_system_es)
    else:
        raise NotImplementedError(task)

SPANISH_SYSTEM_PROMPT = f"""
Eres un corrector de errores de reconocimiento automático del habla (ASR) en español.
Tu tarea es transformar una hipótesis de ASR en una versión corregida, realizando
solo modificaciones mínimas y estrictamente necesarias.

{get_example(lang='ES', json=JSON_OUT, nbest=STRATEGY=='nbest_reason_rewrite') if FEW_SHOT and not USE_TAP else ''}
"""

SPANISH_SYSTEM_PROMPT += """
Instrucciones de corrección:
- corrige únicamente errores evidentes; no reformules ni parafrasees
- prioriza sustituciones por palabras que suenen de forma muy similar
- no borres palabras salvo que resulte imprescindible para la coherencia mínima
- no inventes palabras ni completes fragmentos dudosos

Formato de salida:
- devuelve solo la frase corregida
- todo en minúsculas
- sin signos de puntuación, paréntesis, guiones, comillas ni símbolos
- usa únicamente espacios simples entre palabras
- no añadas comentarios ni explicaciones
""" if not USE_TAP else """
Formato de salida:
- devuelve solo la frase corregida
- todo en minúsculas
- sin signos de puntuación, paréntesis, guiones, comillas ni símbolos
- usa únicamente espacios simples entre palabras
- no añadas comentarios ni explicaciones
"""

ENGLISH_SYSTEM_PROMPT = f"""
You are a corrector of automatic speech recognition (ASR) errors in Spanish.
Your task is to transform an ASR hypothesis into a corrected version, making
only minimal and strictly necessary modifications.
{get_example(lang='EN', json=JSON_OUT, nbest=STRATEGY=='nbest_reason_rewrite') if FEW_SHOT and not USE_TAP else ''}
"""
ENGLISH_SYSTEM_PROMPT += """
Correction guidelines:
- correct only obvious errors; do not rephrase or paraphrase
- prioritize substitutions with words that sound very similar
- do not delete words unless it is essential for minimal coherence
- do not invent words or complete uncertain fragments

Output format:
- return only the corrected sentence
- all in lowercase
- no punctuation marks, parentheses, dashes, quotation marks, or symbols
- use only single spaces between words
- do not add comments or explanations
""" if not USE_TAP else """
Output format:
- return only the corrected sentence
- all in lowercase
- no punctuation marks, parentheses, dashes, quotation marks, or symbols
- use only single spaces between words
- do not add comments or explanations
"""
#- Respond only with the output. Do not include Human:, Assistant:, or any role labels.
# -----------------Search config-----------------
trans_only_LM = False

CHEAT_N_BEST = False and seg_key == "ref"
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
CHEAT_CTX = True and LLM_PREV_ONE_CTX and seg_key == "ref"
CTX_LEN_LIMIT = 100

TUNE_LLM_SCALE_EVERY_PASS = LLM_PREV_ONE_CTX and not CHEAT_CTX

DIAGNOSE = True and not TUNE_LLM_SCALE_EVERY_PASS

# --- Helpers for ctc_exp ---

def get_decoding_config(lmname: str, lm, vocab: str, encoder: str, nbest: int =50, beam_size: int=80, real_vocab: VocabConfig = None) -> Tuple[dict, dict, dict, dict, bool, Optional[int]]:
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

    alias_name = f"{seg_key}_apptek-ctc-baseline_{encoder}_decodingWith_1st-{lmname}_{lm_hyperparamters_str}_{'LMTune' if not TUNE_ON_GREEDY_N_LIST else ''}_2rd{lm2_hyperparamters_str}"

    EC_config = decoding_config.get("EC_config",None)
    EC_model_name = EC_config.model_name if EC_config else None
    alias_name += f"_{TASK_instruct}_with_{os.path.basename(EC_model_name)}" if EC_model_name else ""
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

    decoding_config["tune_with_orig_Nbest"] = TUNE_WITH_ORIG_NBEST
    if EC_config:
        decoding_config["EC_config"] = EC_config
    if USE_GIVEN_NBEST:
        from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import NbestListDataset
        assert vocab_config is not None
        decoding_config["Nbest_dataset"] = NbestListDataset(spm=vocab_config, replace_list=[("[unknown]","<unk>"), ("[noise]", "<sep>"), ("[music]", "▁mes")])
        decoding_config["combine_with_given_Nlist"] = COMBINE_NLIST
        decoding_config["diagnose"] = DIAGNOSE and not decoding_config["tune_with_orig_Nbest"]
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
            dev_dataset_keys_for_first_pass_tune = DEV_DATASET_KEYS_FIRST_PASS,
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
        "meta-llama/Meta-Llama-3-8B-Instruct": ((100 if TASK_instruct == "EC" else 80) if DEFAULT_PROMPT else 80) - reduce_offset,
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
                system_prompt=(
                    SPANISH_SYSTEM_PROMPT if PROMPT_LANG == "ES" else ENGLISH_SYSTEM_PROMPT) if not DEFAULT_PROMPT else get_system_prompt1(names_focus=NAME_FOCUSED, lang=PROMPT_LANG, few_shot=FEW_SHOT, task=TASK_instruct),
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
                hf_batch_size=EC_LLMs_Batch_size[model_id],
                max_new_tokens=128 if TASK_instruct == "ES" else 1024,
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
                system_prompt=(SPANISH_SYSTEM_PROMPT if PROMPT_LANG == "ES" else ENGLISH_SYSTEM_PROMPT) if not DEFAULT_PROMPT else None,
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
    seg_key: str,
    aggregated: bool,
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

    ppl_list = [e.ppl for e in entries]
    wer_result_paths = [e.wer_result_path for e in entries]
    miscs = [e.miscs for e in entries]
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
        aggregated=aggregated,
        eval_dataset_keys=eval_dataset_keys,
        misc=miscs,
    )

    # Suffix for LLM/context etc, still centralised
    llm_suffix = build_llm_name_suffix()

    # Now: one alias that covers *all* EC configs together.
    alias_prefix = (
        f"ES_wer_ppl/"
        f"{seg_key}_"
        f"1st_pass_{model_name}"
        f"{'givenNbest' if USE_GIVEN_NBEST else ''}"
        f"{'combined' if COMBINE_NLIST else ''}"
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
    model,
    model_config: dict,
    vocab_config,
    vocab: str,
    encoder: str,
    train: bool,
    lms: Dict[str, object],
    rescor_lms: Dict[str, object],
    ppl_results: Dict[str, dict],
    ppl_results_2: Dict[str, dict],
    ec_configs: List[Tuple[str, Optional[LLMECConfig]]],
    i6_models,
    seg_key: str,
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
            model=model,
            model_config=model_config,
            vocab_config=vocab_config,
            prior_file=PRIOR_PATH[(vocab, "ctc", encoder)],
            i6_models=i6_models,
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
            model=model,
            model_config=model_config,
            vocab_config=vocab_config,
            prior_file=PRIOR_PATH[(vocab, "ctc", encoder)],
            i6_models=i6_models,
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
                    model=model,
                    model_config=model_config,
                    vocab_config=vocab_config,
                    prior_file=PRIOR_PATH[(vocab, "ctc", encoder)],
                    i6_models=i6_models,
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
            seg_key=seg_key,
            aggregated=AGGREGATED_WER,
            eval_dataset_keys=EVAL_DATASET_KEYS,
        )


def py():
    available = [("spm10k", "ctc", "conformer")]
    models = {"ctc": ctc_exp}
    encoder = "conformer"
    train = False

    global CUTS
    CUTS = {"conformer": 65, "blstm": 37}

    from i6_experiments.users.zhang.experiments.apptek.am.ctc_spm10k_16khz_mbw import (
        get_model_and_vocab,
        NETWORK_CONFIG_KWARGS,
    )

    model, spm, i6_models = get_model_and_vocab(fine_tuned_model=FINE_TUNED_MODEL)
    model_config = {
        "network_config_kwargs": NETWORK_CONFIG_KWARGS,
        "preload_from_files": {
            "am": {
                "prefix": "AM.",
                "filename": model.get_epoch(CKPT_EPOCH).checkpoint,
            },
        },
        "allow_random_model_init": True,
    }

    for k, v in spm["vocabulary"].items():
        print(f"{k}: {v}")

    vocab_config = SentencePieceModel(
        dim=spm["vocabulary"]["vocabulary_size"], model_file=spm["spm"]
    )

    # one vocab for now
    for vocab in ["spm10k"]:
        word_ppl = True
        lm_kinds = {"trafo"}
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
            vocab_config,
            lm_kinds=lm_kinds,
            lm_kinds_2=lm_kinds_2,
            word_ppl=word_ppl,
            trans_only_LM=trans_only_LM,
            task_name="ES",
        )

        ec_configs = build_ec_configs()  # includes "NoLM" EC baseline etc.

        for model_name, exp in models.items():
            if (vocab, model_name, encoder) not in available:
                train = True

            run_first_and_second_pass(
                exp=exp,
                model_name=model_name,
                model=model,
                model_config=model_config,
                vocab_config=vocab_config,
                vocab=vocab,
                encoder=encoder,
                train=train,
                lms=lms,
                rescor_lms=rescor_lms,
                ppl_results=ppl_results,
                ppl_results_2=ppl_results_2,
                ec_configs=ec_configs,
                i6_models=i6_models,        # <--- fixed
                seg_key=seg_key,
            )