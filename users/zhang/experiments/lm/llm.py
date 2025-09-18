from __future__ import annotations
import functools
from typing import Optional, Any, Dict, List, Sequence, TYPE_CHECKING, Iterable
import re
import math

from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import get_lm_eval_text
from i6_experiments.users.zhang.experiments.exp_wer_ppl import LLM_WITH_PROMPT, LLM_WITH_PROMPT_EXAMPLE


from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
from sisyphus import Job, Task, tk

from i6_experiments.common.datasets.librispeech.language_model import (
    get_librispeech_normalized_lm_data,
)
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    get_content_dir_from_hub_cache_dir,
)

from i6_experiments.users.zhang.datasets.librispeech import (
    get_train_corpus_text,
    _get_test_corpus_text,
    _get_corpus_text_dict,
    get_test_corpus_text,
)

from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import get_corpus_text_dict as ES_get_corpus_text_dict

import torch

if TYPE_CHECKING:
    pass
from i6_core.util import uopen

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_fix_context_file(size: int):
    if size < 1:
        return None
    path = f"llm_fixed_ctx/fix_ctx_forLLM_LBS_{size}.txt" #/u/haoran.zhang/setups/2024-12-16--lm-ppl/
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(path)

USE_LOWER_CASE = True
CTX_10_STR = ["THEY SAID IT TO THE END VERSE ANSWERING VERSE AND THE PRAYER OF THE KING POET STILLED THE THROBBING OF HURTS TOO DEEP TO HEAL\nMARY DID YOU EVER THINK WHAT YOU WOULD DO IF YOU HAD TO LIVE ON JUST A FEW CENTS A DAY\nTRUE COAL HAD TO BE BROUGHT FROM SOME DISTANCE AND THERE WAS A GREAT NEED OF REALLY SKILLED LABOR\nWELL THEN WE'LL TALK ABOUT BEAUTIFUL WOMEN IF YOU PREFER\nWAS IT ON THE STAGE THAT YOU FOUND YOUR MOST INTENSE JOYS YOUR TRUE HAPPINESS\nIT WAS WELL KNOWN OF COURSE TO JEANJEAN THAT HIS PRISONER HAD BEEN GUILTY OF THE OFFENCE FOR WHICH HE HAD ARRESTED HIM AND THE COUP WAS QUITE EASY\nHE SNATCHED THE WHIP AND STRUCK THE CONDEMNED MAN WITH IT AS HIGH UP AS HE COULD REACH MAKING A GREAT WELT ACROSS HIS BARE STOMACH\nWHAT DID THAT CREEP WANT\nWE HAVE COME DOWN HERE TO DO A LITTLE PROSPECTING AND WERE JUST RIDING AROUND A BIT TO TAKE A LOOK AT THE COUNTRY\nBUT BECAUSE MANY OF THE SIMPLE IDEAS THAT MAKE UP OUR SPECIFIC IDEAS OF SUBSTANCES ARE POWERS WHICH LIE NOT OBVIOUS TO OUR SENSES IN THE THINGS AS THEY ORDINARILY APPEAR THEREFORE IN THE SIGNIFICATION OF OUR NAMES OF SUBSTANCES SOME PART OF THE SIGNIFICATION WILL BE BETTER MADE KNOWN BY ENUMERATING THOSE SIMPLE IDEAS THAN BY SHOWING THE SUBSTANCE ITSELF"]
PROMPT = ["THIS IS A TEXT DATA SOURCED FROM PUBLIC DOMAIN AUDIOBOOKS\nIT REPRESENTS THE DOMAIN OF READ, AND SCRIPTED SPEECH"]
EXAMPLE = ["I SAY ADVERSARIES FOR ON RECALLING SUCH PROUD MEMORIES WE SHOULD AVOID THE WORD ENEMIES WHOSE HOSTILE SOUND PERPETUATES THE ANTAGONISMS AND STRIFE OF NATIONS SO IRREMEDIABLE PERHAPS SO FATEFUL AND ALSO SO VAIN"]
LLM_Batch_size = {#"meta-llama/Llama-3.2-1B": 18*3,
                  "meta-llama/Llama-3.1-8B": 14*9, #10*9, # actual 15 batch size-> ~ 50GB peak usage#  be at least 3 times larger from 10*3
                  #"Qwen/Qwen3-0.6B-Base": 51,
                  "Qwen/Qwen3-1.7B-Base": 30*3,#15 has peak 19GB on 48G, so can be at least doubled
                  #"Qwen/Qwen3-4B-Base":24,
                  #"Qwen/Qwen3-8B-Base":5*3,
                  "microsoft/phi-4": 14*6, #Can be 24 on 80GB with 100 ctx(peak 40 with 14), so 20*6
                  #"mistralai/Mistral-7B-v0.3": 4,
                  } # Keys of this determines which LLM will be built by lm_getter

LLM_Batch_size_PPL = {"meta-llama/Llama-3.2-1B": 18*3,
                  "meta-llama/Llama-3.1-8B": 4*3,
                  "Qwen/Qwen3-0.6B-Base": 51,
                  "Qwen/Qwen3-1.7B-Base": 36,
                  "Qwen/Qwen3-4B-Base":24,
                  "Qwen/Qwen3-8B-Base":4*3,
                  "microsoft/phi-4": 4*3,
                  }
LLM_rqmt = {"meta-llama/Llama-3.2-1B": {"time": 3, "cpu": 3, "mem": 16, "gpu": 1, "gpu_mem": 48},
            "meta-llama/Llama-3.1-8B": {"time": 4, "cpu": 3, "mem": 40, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-0.6B-Base": {"time": 3, "cpu": 3, "mem": 12, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-1.7B-Base": {"time": 3, "cpu": 3, "mem": 20, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-4B-Base":{"time": 3, "cpu": 3, "mem": 25, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-8B-Base":{"time": 4, "cpu": 3, "mem": 40, "gpu": 1, "gpu_mem": 48},
                  #"mistralai/Mistral-7B-v0.3": 4,
            "microsoft/phi-4":{"time": 4, "cpu": 3, "mem": 65, "gpu": 1, "gpu_mem": 80},
            }




def get_raw_text_func_ES_spm(seq:list):
    # import sentencepiece as spm
    #
    # sp = spm.SentencePieceProcessor(model_file="/nas/models/asr/artefacts/subword_units/sentencepiece/ES/2025-04-spm_10240-mbw/10240-nmt_nfkc_cf.spm")
    # pieces = ["▁ac", "tion"]
    # return sp.decode_pieces(seq)  # -> "action"
    if len(seq) == 0:
        return " ".join(seq)
    if "▁" not in " ".join(seq): # Should not further do replacement
        print(f"Warning: Passed non spm sequence: \n{seq}\n to get_raw_text_func_ES_spm")
        return " ".join(seq)
    return " ".join(seq).replace("<sep>", "▁[noise]").replace(" ", "").replace("▁", " ")

def get_prompt(LLM_FXIED_CTX, LLM_FXIED_CTX_SIZE):
    prompt = None
    if LLM_FXIED_CTX:
        prompt = tk.Path(get_fix_context_file(LLM_FXIED_CTX_SIZE))
    elif LLM_WITH_PROMPT:
        prompt = PROMPT
        if LLM_WITH_PROMPT_EXAMPLE:
            prompt += [f"This is one sentence as an example: {EXAMPLE[0]}"]
        prompt = prompt
    return prompt

#@functools.cache
def get_llm(model_ids: List[str], batch_sizes: List[int] = None, word_ppl: bool = False, task_name: str = "LBS") -> tuple[
    dict[Any, dict[str, int | str | Any]], dict[Any, Any]]:
    if task_name == "LBS":
        from i6_experiments.users.zhang.experiments.exp_wer_ppl import CTX_LEN_LIMIT, LLM_FXIED_CTX, LLM_FXIED_CTX_SIZE, LLM_PREV_ONE_CTX#, CHEAT_CTX
    elif task_name == "ES":
        from i6_experiments.users.zhang.experiments.LLM_apptek_exp import CTX_LEN_LIMIT, LLM_FXIED_CTX, LLM_FXIED_CTX_SIZE, LLM_PREV_ONE_CTX, CHEAT_CTX
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    ds_names = ["test-other", "dev-other"] if task_name=="LBS" else list(set(DEV_KEYS + TEST_KEYS))
    #main_ppl_mesure_name = ["test-other"] if task_name=="LBS" else (DEV_KEYS + TEST_KEYS)[0]
    prompt = get_prompt(LLM_FXIED_CTX, LLM_FXIED_CTX_SIZE)
    llms = dict()
    ppls = dict()
    if not batch_sizes:
        batch_sizes = [2 for _ in model_ids]
    prompt = [prompt for _ in model_ids]
    assert len(model_ids) == len(batch_sizes)
    name_ext = f"{LLM_FXIED_CTX_SIZE}" if LLM_FXIED_CTX else ""
    name_ext += f"prev_{CTX_LEN_LIMIT}" if LLM_PREV_ONE_CTX else ""
    name_ext += f"{'prompt' if LLM_WITH_PROMPT else ''}{'_example' if LLM_WITH_PROMPT_EXAMPLE else ''}{'_low' if USE_LOWER_CASE else ''}"

    for model_id, prompt, batch_size in zip(model_ids, prompt, batch_sizes):
        ppl_batch_size = LLM_Batch_size_PPL[model_id]
        if LLM_FXIED_CTX:
            ppl_batch_size = LLM_Batch_size_PPL[model_id] // 3
            batch_size = LLM_Batch_size[model_id] // 3
        if LLM_FXIED_CTX_SIZE > 10:
            batch_size = 1
        if LLM_PREV_ONE_CTX:
            batch_size = LLM_Batch_size[model_id] // 6
            ppl_batch_size = LLM_Batch_size_PPL[model_id] // 6
        name = os.path.basename(model_id)
        model = DownloadHuggingFaceRepoJob(model_id=model_id)
        tk.register_output(model_id, model.out_hub_cache_dir)
        lm_cfg = {"model_dir": model.out_hub_cache_dir, "batch_size": batch_size,
                  "name": name, "prompt": prompt, "lm_type": "HuggingFaceLm"}
        if LLM_FXIED_CTX:
            lm_cfg.update({"eos_symbol": "\n"})
        if LLM_PREV_ONE_CTX:
            lm_cfg.update({"eos_symbol": "\n"})
            lm_cfg.update({"prev_one_ctx": LLM_PREV_ONE_CTX})
            lm_cfg.update({"ctx_len_limit": CTX_LEN_LIMIT})
            if CHEAT_CTX:
                lm_cfg.update({"cheat_prev_ctx": True})
        if USE_LOWER_CASE:
            lm_cfg.update({"lower_case": True})
        if task_name=="ES":
            lm_cfg.update({"get_raw_text_func": get_raw_text_func_ES_spm})
        llms.update({name: lm_cfg})
        ppls[name] = dict()
        for ds_name in ds_names:
            if task_name=="LBS":
                text_file = _get_corpus_text_dict(key=ds_name)
            elif task_name=="ES":
                text_file = ES_get_corpus_text_dict(key=ds_name)
            else:
                raise ValueError(f"Unknown task name: {task_name}")
            ppl_job = HuggingFaceLmPerplexityJobV2(
                model_dir=model.out_hub_cache_dir,
                text_file=[text_file], # get_test_corpus_text(keys=[ds_name])
                llm_name=model_id,
                batch_size=max(ppl_batch_size//2,1),
                lower_case=USE_LOWER_CASE,
                word_ppl=word_ppl,
                prompt=prompt,
                eos_symbol="\n",
                use_prev_context=LLM_PREV_ONE_CTX,
                context_len_limit=CTX_LEN_LIMIT if LLM_PREV_ONE_CTX else None,
            )
            ppl_job.rqmt.update(LLM_rqmt[model_id])
            ppl_job_name = f"ppl/{name}/{ds_name}" + (f"_ctx{LLM_FXIED_CTX_SIZE}" if LLM_FXIED_CTX else "") + f"{'low' if USE_LOWER_CASE else ''}{f'_prev{str(CTX_LEN_LIMIT)}' if LLM_PREV_ONE_CTX else ''}"
            ppl_job.add_alias(ppl_job_name)
            ppls[name].update({ds_name: ppl_job.out_ppl})
            # if main_ppl_mesure_name == ds_name:
            #     ppls.update({name: ppl_job.out_ppl})
            #print(lm_cfg)
        # ppl_job.add_alias("lm/" + llm_name + "/ppl/librispeech_" + ds_name + ("low" if lower_case else "") + eos_name)
            tk.register_output(
                "ppl/" + name + f"/{task_name}-" + ds_name + name_ext + "-ppl",
                ppl_job.out_ppl)
    return llms, ppls

"""Compute perplexity for a HuggingFace Transformer LLM model on given text corpus."""
class HuggingFaceLmPerplexityJob(Job):
    """Compute perplexity of a HuggingFace LM over a text corpus."""
    __sis_hash_exclude__ = {"batch_size": None}
    def __init__(self, *, model_dir: tk.Path, text_file: List[tk.Path], batch_size: int = None,
                 llm_name: str, lower_case:bool = False, eos_symbol: str = "", word_ppl: bool = False, version:int = 3):
        super().__init__()
        #self.name = f"HFLM-PPL-{llm_name}-{self.text_file[0].basename()}"
        self.model_dir = model_dir
        self.text_file = text_file
        self.batch_size = {"Llama-3.2-1B": 8, "Llama-3.1-8B": 3}.get(llm_name,1) if batch_size is None else batch_size
        self.lower_case = lower_case
        self.eos_symbol = eos_symbol
        self.word_ppl = word_ppl
        self.out_ppl = self.output_path("ppl")
        self.rqmt = {"time": 4, "cpu": 3, "mem": 8 + self.batch_size//2, "gpu": 1, "gpu_mem": {"Llama-3.2-1B": 10, "Llama-3.1-8B": 36}.get(llm_name,10)}
        self.rqmt.update({"mem": {"Llama-3.2-1B": 15, "Llama-3.1-8B": 40}.get(llm_name,25),
                        "time": {"Llama-3.2-1B": 4, "Llama-3.1-8B": 6}.get(llm_name,4)})

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import math
        import time
        import returnn.util.basic as util # noqa

        def _score_batch(batch_lines, tokenizer, model, device):
            """
            Tokenize a batch of lines, run through the model, and compute negative log-likelihood,
            token count, and total BPE tokens (sum of sequence lengths).
            Returns (neg_log_likelihood, token_count, total_bpe_tokens).
            """
            encoding = tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Mask out padding tokens in the labels
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            with torch.no_grad():
                out = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Total non-padded tokens across the batch
            token_count = int(attention_mask.sum().item())
            # Sum of BPE tokens per sequence
            bpe_seq_lengths = attention_mask.sum(dim=1).tolist()
            total_bpe = sum(bpe_seq_lengths)
            # out.loss is average over non-ignored tokens; multiply by token count
            neg_log_likelihood = out.loss.item() * token_count
            return neg_log_likelihood, token_count, total_bpe

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        def _report_dev_memory_stats():
            dev = torch.device(device_str)
            if dev.type == "cuda":
                stats = [
                    f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                    f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                    f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                    f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
                ]
                print(f"Memory usage ({device_str}):", " ".join(stats))

        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_time = time.time()
        print("Loading model...")

        model_path = get_content_dir_from_hub_cache_dir(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"\nTokenizer_max_length:{tokenizer.model_max_length}\n")

        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        model.eval()
        device = torch.device(device_str)
        model.to(device)

        print(f"({time.time() - start_time} secs)")
        _report_dev_memory_stats()

        for s in ["A BUSINESS. ", "a business \n"]:
            enc = tokenizer(s, return_tensors="pt")
            print(s, "→ token IDs:", enc.input_ids.tolist())
        print(f"eos_token id:{tokenizer.eos_token}")

        total_logprob = 0.0
        total_tokens = 0
        total_word_tokens = 0

        lines_seen = 0
        bpe_length = 0
        total_lines = 0
        batch_count = 0
        log_every = 1000  # print a message every 1k lines

        for text_file in self.text_file:
            if text_file.get_path().endswith(".gz"):
                import gzip

                open_func = gzip.open
            else:
                open_func = open

            with open_func(text_file.get_path(), "rt") as f:
                for line in f:
                    total_lines += 1
                    total_word_tokens += len(line.strip().split())

        for text_file in self.text_file:
        # Open the file and iterate line by line
            if text_file.get_path().endswith(".gz"):
                import gzip

                open_func = gzip.open
            else:
                open_func = open
            with open_func(text_file.get_path(), "rt") as f:
                batch_lines = []
                for raw_line in f:
                    line = raw_line.strip().lower() if self.lower_case else raw_line.strip()
                    if not line:
                        continue

                    eos_symbol = (" "+tokenizer.eos_token) if self.eos_symbol == "eos" else self.eos_symbol #(" "+tokenizer.eos_token) makes ppl worse
                    batch_lines.append(line + eos_symbol)

                    lines_seen += 1

                    # Log after every `log_every` lines
                    if lines_seen % log_every == 0:
                        print(f"[Line {lines_seen:,}/{total_lines}] {100*lines_seen/total_lines:.2f}% processed…")
                        print(f"current lines:{batch_lines}")
                        _report_dev_memory_stats()
                    # Once we have `batch_size` lines, tokenize & process them
                    if len(batch_lines) == self.batch_size:
                        # ➞ tokenize + move to device
                        # batch_count += 1
                        # encoding = tokenizer(
                        #     batch_lines,
                        #     return_tensors="pt",
                        #     padding=True,
                        #     truncation=True,
                        #     max_length=tokenizer.model_max_length,
                        # )
                        # input_ids = encoding["input_ids"].to(device)
                        # bpe_length += len(encoding["input_ids"][0].tolist())
                        # attention_mask = encoding["attention_mask"].to(device)
                        #
                        # # Mask out padding in the labels
                        # labels = input_ids.clone()
                        # labels[attention_mask == 0] = -100
                        #
                        # with torch.no_grad():
                        #     out = model(input_ids, attention_mask=attention_mask, labels=labels)
                        # # out.loss is average over non‐ignored tokens; multiply by token count
                        # batch_tok_count = int(attention_mask.sum().item())
                        # neg_log_likelihood = out.loss.item() * batch_tok_count
                        batch_count += 1
                        nll, tok_count, bpe_sum = _score_batch(batch_lines, tokenizer, model, device)
                        total_logprob -= nll
                        total_tokens += tok_count
                        bpe_length += bpe_sum

                        if batch_count % 1000 == 0:
                            print(f"  → Completed {batch_count:,} batches; Tokens so far: {total_tokens:,}")

                        batch_lines = []  # clear for next batch



            # Process any leftover lines (if total lines % batch_size != 0)
            if batch_lines:
                nll, tok_count, bpe_sum = _score_batch(batch_lines, tokenizer, model, device)
                total_logprob -= nll
                total_tokens += tok_count
                bpe_length += bpe_sum

        print(f"(Assumed batch size 1)Average bpe seq length:{bpe_length/total_lines:.2f}")
        print(f"Average bpe/word length ratio:{total_tokens}/{total_word_tokens}->{total_tokens / total_word_tokens:.2f}")
        # Explicit cleanup to avoid stuck CG state
        del model
        del tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Finished and cleaned up.")

        # Finally compute PPL
        ppl = math.exp(-total_logprob / total_word_tokens) if self.word_ppl else math.exp(-total_logprob / total_tokens)
        with open(self.out_ppl.get_path(), "w") as out_f:
            out_f.write(f"Average bpe/word length ratio:: {total_tokens / total_word_tokens:.2f}\n")
            out_f.write(f"Perplexity: {ppl}\n")

"""Compute perplexity for a HuggingFace Transformer LLM model on given text corpus."""
class HuggingFaceLmPerplexityJobV2(Job):
    """Compute perplexity of a HuggingFace LM over a text corpus.
        Using a fixed context from training set
    """
    __sis_hash_exclude__ = {"batch_size" : None}
    def __init__(self, *, model_dir: tk.Path, prompt: [List[str] | tk.Path] = None, text_file: List[tk.Path], batch_size: int = None,
                 llm_name: str, lower_case:bool = False, context_len_limit: int = None, eos_symbol: str = "", word_ppl: bool = False, add_eos_to_completion: bool = False, use_prev_context: bool = False, version:int = 5):
        super().__init__()
        #self.name = f"HFLM-PPL-{llm_name}-{self.text_file[0].basename()}"
        self.model_dir = model_dir
        self.text_file = text_file
        self.batch_size = batch_size or 4
        self.lower_case = lower_case
        self.add_eos_to_completion = add_eos_to_completion
        self.eos_symbol = eos_symbol
        self.delimiter = " " if not self.eos_symbol else (self.eos_symbol + " ") # Not sure
        self.use_prev_context = use_prev_context
        self.context_len_limit = context_len_limit
        self.context = prompt

        # These two are used for prev_ctx
        self.prompt = None #
        self.prompt_buffer = []

        # Legacy setting for use fixed context
        # if isinstance(prompt, tk.Path):
        #     with open(prompt.get_path(), "r", encoding="utf-8") as f:
        #         prompt = [line.strip() for line in f.readlines()]
        # if prompt:
        #     prompt +=  [""]  # +[""] So that for last prompt(or only one prompt) it also has eos
        #     self.prompt = self.delimiter.join(prompt)
        self.word_ppl = word_ppl
        self.out_ppl = self.output_path("ppl")
        self.rqmt = {"time": 4, "cpu": 3, "mem": 8 + self.batch_size//2, "gpu": 1, "gpu_mem": {"Llama-3.2-1B": 10, "Llama-3.1-8B": 36}.get(llm_name,10)}
        # self.rqmt.update({"mem": {"Llama-3.2-1B": 15, "Llama-3.1-8B": 40}.get(llm_name,25),
        #                 "time": {"Llama-3.2-1B": 4, "Llama-3.1-8B": 6}.get(llm_name,4)})

    def ctx_over_limit(self):
        if self.context_len_limit is None:
            return False
        return len(self.delimiter.join(self.prompt_buffer + [""]).split()) > self.context_len_limit

    def update_prompt(self, new_prompt: str):
        self.prompt_buffer += [new_prompt]
        while self.ctx_over_limit():
            self.prompt_buffer.pop(0)
        self.prompt = self.prompt_buffer.copy()
        self.prompt += [""]  # +[""] So that for last prompt(or only one prompt) it also has eos
        self.prompt = self.delimiter.join(self.prompt)

    def clear_prompt(self):
        torch.cuda.empty_cache()
        self.prompt_buffer = []
        self.prompt = ""

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import math
        import time
        import returnn.util.basic as util # noqa

        debug_flag = True
        def _score_batch(batch_lines, batch_prompt, tokenizer, model, device):
            """
            Tokenize a batch of lines, run through the model, and compute negative log-likelihood,
            token count, and total BPE tokens (sum of sequence lengths).
            Returns (neg_log_likelihood, token_count, total_bpe_tokens).
            """
            nonlocal debug_flag
            encoding = tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True if self.prompt is None else False,
                max_length=tokenizer.model_max_length,
            )
            hyp_input_ids = encoding["input_ids"].to(device)
            # Prepare inputs
            if self.prompt:
                enc_prompt = tokenizer(
                    batch_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                )
                start = enc_prompt["input_ids"].shape[1]  # M
                end = start + hyp_input_ids.shape[1]  # M + H
                input_ids = torch.cat([enc_prompt["input_ids"], encoding["input_ids"]], dim=1).to(device)
                input_ids = input_ids.long()
                attention_mask = torch.cat([enc_prompt["attention_mask"], encoding["attention_mask"]], dim=1).to(device)
                if debug_flag:
                    debug_flag = False
                    print(f"\n \n batch_prompt: {batch_prompt[0:2]}\n batch_prompt_token: {enc_prompt['input_ids'][0:2]}, shape:{enc_prompt['input_ids'].shape}")
                    print(f"\n \n encoding: {batch_lines[0:2]}\n encoding_token: {hyp_input_ids[0:2]}\n  shape:{hyp_input_ids.shape}")
                    print(f"\n \n Cat: {input_ids[0:2]}\n shape:{input_ids.shape}")
                    print(f"\n \n Slice: {input_ids[0][start:end]}\n shape:{input_ids[0][start:end].shape} , with start{start}, end{end}")
                    print(f"\n \n Slice2: {input_ids[0][-hyp_input_ids.shape[1] - 1:-1]}\n shape:{input_ids[0][-hyp_input_ids.shape[1] - 1:-1].shape} , with start{-hyp_input_ids.shape[1] - 1}, end{-1}")
            else:
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

            # Mask out padding tokens in the labels, only need for loss
            # labels = input_ids.clone()
            # labels[attention_mask == 0] = -100

            # Compute logits and log-probs
            with torch.no_grad():
                logits = model(input_ids, attention_mask=attention_mask).logits
                gather_ids = hyp_input_ids[:, 1:].unsqueeze(-1)
                scores_mask = encoding["attention_mask"][..., 1:].to(device)
                if self.prompt:
                    gather_ids = hyp_input_ids.unsqueeze(-1) # No bos here
                    scores_mask = encoding["attention_mask"].to(device)
                    logits = logits[:, -hyp_input_ids.shape[1] - 1:-1, :]
                    #logits = logits[:, start :end, :]

                log_probs = torch.log_softmax(logits, dim=-1)
                nll = torch.gather(log_probs, dim=-1, index=gather_ids).squeeze()
                nll = nll * scores_mask
                nll = nll.sum()

            # Total non-padded tokens across the batch
            token_count = int(encoding["attention_mask"].sum().item())

            return nll, token_count

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        def _report_dev_memory_stats():
            dev = torch.device(device_str)
            if dev.type == "cuda":
                stats = [
                    f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                    f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                    f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                    f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
                ]
                print(f"Memory usage ({device_str}):", " ".join(stats))

        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_time = time.time()
        print("Loading model...")

        model_path = get_content_dir_from_hub_cache_dir(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"\nTokenizer_max_length:{tokenizer.model_max_length}\n")

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[
            0] >= 8 else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,  # load directly in low precision
            device_map={"": 0},  # put the whole model on cuda:0 (single GPU)
            low_cpu_mem_usage=True,  # stream weights, smaller CPU peak
            #attn_implementation="flash_attention_2",  # faster runtime; f
        )
        #model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        model.eval()
        device = torch.device(device_str)
        #model.to(device)

        print(f"({time.time() - start_time} secs)")
        _report_dev_memory_stats()

        for s in ["A BUSINESS. ", "a business \n"]:
            enc = tokenizer(s, return_tensors="pt")
            print(s, "→ token IDs:", enc.input_ids.tolist())
        print(f"bos_token id:{tokenizer.bos_token}")
        print(f"eos_token id:{tokenizer.eos_token}")

        total_logprob = 0.0
        total_tokens = 0
        total_word_tokens = 0

        lines_seen = 0
        total_lines = 0
        batch_count = 0
        log_every = 1000  # print a message every 1k lines
        d_rec = dict()
        import i6_core.util as cutil
        for text_file in self.text_file:
            d_rec.update(eval(cutil.uopen(text_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")}))
            # Iterate records
            lines_seen = 0
        total_lines = sum(len(n_best) for _, n_best in d_rec.items())
        from i6_experiments.users.zhang.datasets.utils import sort_dict_by_record, extract_record_id
        if self.use_prev_context:
            d_rec = sort_dict_by_record(d_rec)
        ctx_rec = None
        if self.context is not None:
            ctx_rec = eval(cutil.uopen(self.context, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        batch_lines, batch_prompt = [], []
        eos_symbol = (
                " " + tokenizer.eos_token) if self.eos_symbol == "eos" else self.eos_symbol  # (" "+tokenizer.eos_token) makes ppl worse
        eos_symbol = eos_symbol if self.add_eos_to_completion else ""
        last_record = None
        for seq_tag, raw_line in d_rec.items():
            line = raw_line.strip().lower() if self.lower_case else raw_line.strip()
            if not line:
                continue
            total_lines += 1
            total_word_tokens += len(line.split()) + (1 if self.eos_symbol else 0)
            batch_lines.append(line + eos_symbol)
            if self.use_prev_context:
                current_record = extract_record_id(seq_tag)
                if current_record != last_record:
                    self.clear_prompt()
                    print(f"Clear context for record {last_record}")
            if self.prompt or self.use_prev_context:
                batch_prompt.append(self.prompt.lower() if self.lower_case else self.prompt)
            lines_seen += 1

            # Log after every `log_every` lines
            if lines_seen % log_every == 0:
                print(f"[Line {lines_seen:,}/{total_lines}] {100 * lines_seen / total_lines:.2f}% processed…")
                print(f"current lines:{batch_lines}")
                _report_dev_memory_stats()
            # Once we have `batch_size` lines, tokenize & process them
            if len(batch_lines) == self.batch_size:
                batch_count += 1
                nll, tok_count = _score_batch(batch_lines, batch_prompt, tokenizer, model, device)
                total_logprob += nll
                total_tokens += tok_count

                if batch_count % 1000 == 0:
                    print(f"  → Completed {batch_count:,} batches; Tokens so far: {total_tokens:,}")

                batch_lines, batch_prompt = [], []  # clear for next batch

            if self.use_prev_context:
                if ctx_rec is not None:
                    ctx = ctx_rec[seq_tag]
                    print(f"Ctx for {seq_tag}: {ctx}")
                else:
                    ctx = line
                    print(f"Transcription for {seq_tag}: {ctx}")
                self.update_prompt(ctx)
                print(f"Current context len: {len(self.prompt.split())}")
                last_record = current_record

        # Process any leftover lines (if total lines % batch_size != 0)
        if batch_lines:
            nll, tok_count = _score_batch(batch_lines, batch_prompt, tokenizer, model, device)
            total_logprob += nll
            total_tokens += tok_count

        #print(f"(Assumed batch size 1)Average bpe seq length:{bpe_length/total_lines:.2f}")
        print(f"Average bpe/word length ratio:{total_tokens}/{total_word_tokens}->{total_tokens / total_word_tokens:.2f}")
        # Explicit cleanup to avoid stuck CG state
        del model
        del tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Finished and cleaned up.")

        # Finally compute PPL
        bpe_ppl = math.exp(-total_logprob / total_tokens)
        ppl = math.exp(-total_logprob / total_word_tokens) if self.word_ppl else bpe_ppl
        with open(self.out_ppl.get_path(), "w") as out_f:
            out_f.write(f"Average bpe/word length ratio:: {total_tokens / total_word_tokens:.2f}\n")
            out_f.write(f"bpe level: {bpe_ppl}\n")
            out_f.write(f"Perplexity: {ppl}\n")

"""Compute perplexity for a HuggingFace Transformer LLM model on LibriSpeech."""
class HuggingFaceLmPerplexityJobV3(Job):
    """
    Compute perplexity of a HuggingFace LM over a text corpus.
        Using a fixed context from training set with caching(TODO)
    """

    def __init__(self, *, model_dir: tk.Path, prompt: [List[str] | tk.Path] = None, text_file: List[tk.Path], batch_size: int = None,
                 llm_name: str, lower_case:bool = False, eos_symbol: str = "", word_ppl: bool = False, add_eos_to_completion: bool = False, version:int = 0):
        super().__init__()
        self.model_dir = model_dir
        self.text_file = text_file
        self.batch_size = {"Llama-3.2-1B": 8, "Llama-3.1-8B": 3}.get(llm_name,1) if batch_size is None else batch_size
        self.lower_case = lower_case
        self.add_eos_to_completion = add_eos_to_completion
        self.eos_symbol = eos_symbol
        delimiter = " " if not self.eos_symbol else (self.eos_symbol + " ") # Not sure
        if isinstance(prompt, tk.Path):
            with open(prompt.get_path(), "r", encoding="utf-8") as f:
                prompt = [line.strip() for line in f.readlines()]
        self.prompt = delimiter.join(prompt) if prompt else None
        self.word_ppl = word_ppl
        self.out_ppl = self.output_path("ppl")
        self.rqmt = {"time": 4, "cpu": 3, "mem": 8 + self.batch_size//2, "gpu": 1, "gpu_mem": {"Llama-3.2-1B": 10, "Llama-3.1-8B": 36}.get(llm_name,10)}
        self.rqmt.update({"mem": {"Llama-3.2-1B": 15, "Llama-3.1-8B": 40}.get(llm_name,25),
                        "time": {"Llama-3.2-1B": 4, "Llama-3.1-8B": 6}.get(llm_name,4)})

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import math
        import time
        import returnn.util.basic as util # noqa

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        def _report_dev_memory_stats():
            dev = torch.device(device_str)
            if dev.type == "cuda":
                stats = [
                    f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                    f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                    f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                    f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
                ]
                print(f"Memory usage ({device_str}):", " ".join(stats))

        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_time = time.time()
        print("Loading model...")

        model_path = get_content_dir_from_hub_cache_dir(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"\nTokenizer_max_length:{tokenizer.model_max_length}\n")

        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        model.eval()
        device = torch.device(device_str)
        model.to(device)

        print(f"({time.time() - start_time} secs)")
        _report_dev_memory_stats()

        for s in ["A BUSINESS. ", "a business \n"]:
            enc = tokenizer(s, return_tensors="pt")
            print(s, "→ token IDs:", enc.input_ids.tolist())
        print(f"bos_token id:{tokenizer.bos_token}")
        print(f"eos_token id:{tokenizer.eos_token}")

        total_logprob = 0.0
        total_tokens = 0
        total_word_tokens = 0

        lines_seen = 0
        total_lines = 0
        batch_count = 0
        log_every = 1000  # print a message every 1k lines

        past_kvs, ctx_mask = None, None
        if self.prompt:
            context = [self.prompt.strip().lower() if self.lower_case else self.prompt.strip() for _ in range(self.batch_size)]
            ctx = tokenizer(context, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**ctx, use_cache=True)
                past_kvs = out.past_key_values  # cache for all layers
                ctx_mask = ctx["attention_mask"].to(device)

        debug_flag = True
        def _score_batch(batch_lines, ctx_mask, tokenizer, model, device):
            """
            Tokenize a batch of lines, run through the model, and compute negative log-likelihood,
            token count, and total BPE tokens (sum of sequence lengths).
            Returns (neg_log_likelihood, token_count, total_bpe_tokens).
            """
            nonlocal debug_flag, past_kvs
            past_kvs_copy = past_kvs.copy() # This reset the kvs back for each batch
            encoding = tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True if self.prompt is None else False,
                max_length=tokenizer.model_max_length,
            )
            hyp_input_ids = encoding["input_ids"].to(device)
            # Prepare inputs

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            if ctx_mask is not None:
                attention_mask = torch.cat([ctx_mask, attention_mask], dim=1)

            # Mask out padding tokens in the labels, only need for loss
            # labels = input_ids.clone()
            # labels[attention_mask == 0] = -100

            # Compute logits and log-probs
            with torch.no_grad():
                if self.prompt:
                    print(f"hyp shape{hyp_input_ids.shape}")
                    print(f"ctx mask shape{ctx_mask.shape}")
                    print(f"hyp mask shape{encoding['attention_mask'].shape}")
                    logits = model(input_ids, attention_mask=attention_mask, past_key_values=past_kvs_copy, use_cache=False).logits
                else:
                    logits = model(input_ids, attention_mask=attention_mask).logits
                gather_ids = hyp_input_ids.unsqueeze(-1)
                scores_mask = encoding["attention_mask"].to(device)

                log_probs = torch.log_softmax(logits, dim=-1)
                nll = torch.gather(log_probs, dim=-1, index=gather_ids).squeeze()
                nll = nll * scores_mask
                nll = nll.sum()

            # Total non-padded tokens across the batch
            token_count = int(encoding["attention_mask"].sum().item())

            return nll, token_count

        for text_file in self.text_file:
            if text_file.get_path().endswith(".gz"):
                import gzip

                open_func = gzip.open
            else:
                open_func = open

            with open_func(text_file.get_path(), "rt") as f:
                for line in f:
                    total_lines += 1
                    total_word_tokens += len(line.strip().split()) + (1 if self.eos_symbol else 0)

        for text_file in self.text_file:
        # Open the file and iterate line by line
            if text_file.get_path().endswith(".gz"):
                import gzip

                open_func = gzip.open
            else:
                open_func = open
            with open_func(text_file.get_path(), "rt") as f:
                batch_lines = []
                eos_symbol = (
                            " " + tokenizer.eos_token) if self.eos_symbol == "eos" else self.eos_symbol  # (" "+tokenizer.eos_token) makes ppl worse
                eos_symbol = eos_symbol if self.add_eos_to_completion else ""
                for raw_line in f:
                    line = raw_line.strip().lower() if self.lower_case else raw_line.strip()
                    if not line:
                        continue
                    batch_lines.append(line + eos_symbol)

                    lines_seen += 1

                    # Log after every `log_every` lines
                    if lines_seen % log_every == 0:
                        print(f"[Line {lines_seen:,}/{total_lines}] {100*lines_seen/total_lines:.2f}% processed…")
                        print(f"current lines:{batch_lines}")
                        _report_dev_memory_stats()
                    # Once we have `batch_size` lines, tokenize & process them
                    if len(batch_lines) == self.batch_size:
                        batch_count += 1
                        nll, tok_count = _score_batch(batch_lines, ctx_mask, tokenizer, model, device)
                        total_logprob += nll
                        total_tokens += tok_count

                        if batch_count % 1000 == 0:
                            print(f"  → Completed {batch_count:,} batches; Tokens so far: {total_tokens:,}")

                        batch_lines = []  # clear for next batch



            # Process any leftover lines (if total lines % batch_size != 0)
            if batch_lines:
                nll, tok_count = _score_batch(batch_lines, ctx_mask, tokenizer, model, device)
                total_logprob += nll
                total_tokens += tok_count

        #print(f"(Assumed batch size 1)Average bpe seq length:{bpe_length/total_lines:.2f}")
        print(f"Average bpe/word length ratio:{total_tokens}/{total_word_tokens}->{total_tokens / total_word_tokens:.2f}")
        # Explicit cleanup to avoid stuck CG state
        del model
        del tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Finished and cleaned up.")

        # Finally compute PPL
        bpe_ppl = math.exp(-total_logprob / total_tokens)
        ppl = math.exp(-total_logprob / total_word_tokens) if self.word_ppl else bpe_ppl
        with open(self.out_ppl.get_path(), "w") as out_f:
            out_f.write(f"Average bpe/word length ratio:: {total_tokens / total_word_tokens:.2f}\n")
            out_f.write(f"bpe level: {bpe_ppl}\n")
            out_f.write(f"Perplexity: {ppl}\n")

def raw_text_from_bpe_seq(seq:list):
    return " ".join(seq).replace("@@ ","").replace(" <s>", "")

_v2_forward_out_filename = "output.py.gz"
_v2_forward_ext_out_filename = "output_ext.py.gz"
class HuggingFaceLmRescoringJob(Job):
    """Rescoring the n-best list from ASR beam search"""

    def __init__(self, *, model_dir: tk.Path, weight: float, prompt: str = None, recog_out_file: tk.Path, batch_size: int = None,
                 llm_name: str, lower_case:bool = True, eos_symbol: str = "", version:int = 4):
        super().__init__()
        #self.name = f"HFLM-RESCO-{llm_name}"
        self.model_dir = model_dir
        self.scale = weight
        self.prompt = prompt # TODO:
        self.n_best_file = recog_out_file
        self.batch_size = {"Llama-3.2-1B": 8, "Llama-3.1-8B": 3}.get(llm_name,1) if batch_size is None else batch_size
        self.lower_case = lower_case
        self.eos_symbol = eos_symbol
        self.out_file = self.output_path(_v2_forward_out_filename)
        self.rqmt = {"time": 4, "cpu": 3, "mem": 8 + self.batch_size//2, "gpu": 1, "gpu_mem": {"Llama-3.2-1B": 10, "Llama-3.1-8B": 36}.get(llm_name,10)}
        self.rqmt.update({"mem": {"Llama-3.2-1B": 15, "Llama-3.1-8B": 40}.get(llm_name,12),
                        "time": {"Llama-3.2-1B": 4, "Llama-3.1-8B": 6}.get(llm_name,4)})

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import math
        import time
        import returnn.util.basic as util

        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        def _report_dev_memory_stats():
            dev = torch.device(device_str)
            if dev.type == "cuda":
                stats = [
                    f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                    f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                    f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                    f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
                ]
                print(f"Memory usage ({device_str}):", " ".join(stats))

        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_time = time.time()
        print("Loading model...")

        model_path = get_content_dir_from_hub_cache_dir(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"\nTokenizer_max_length:{tokenizer.model_max_length}\n")

        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        model.eval()
        device = torch.device(device_str)
        model.to(device)

        print(f"({time.time() - start_time} secs)")
        _report_dev_memory_stats()

        import gzip
        import i6_core.util as cutil
        d_rec = eval(cutil.uopen(self.n_best_file, "rt").read())
        out_file = gzip.open(self.out_file.get_path(), "wt")
        out_file.write("{\n")

        total_tokens = 0
        lines_seen = 0
        bpe_length = 0
        total_lines = sum(len(n_best) for _, n_best in d_rec.items())
        batch_count = 0
        log_every = 1000  # print a message every 1k lines

        # Helper to process a batch of lines
        def _process_batch(batch_lines, batch_prompt, scores_buffer):
            nonlocal batch_count, total_tokens, bpe_length
            batch_count += 1
            enc_hyp = tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                add_special_tokens=False if self.prompt else True,
            )
            hyp_input_ids = enc_hyp["input_ids"].to(device)
            bpe_length += len(enc_hyp["input_ids"][0].tolist())

            # Prepare inputs
            if self.prompt:
                enc_prompt = tokenizer(
                    batch_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                )
                input_ids = torch.cat([enc_prompt["input_ids"], enc_hyp["input_ids"]], dim=1).to(device)
                attention_mask = torch.cat([enc_prompt["attention_mask"], enc_hyp["attention_mask"]], dim=1).to(device)
            else:
                input_ids = enc_hyp["input_ids"].to(device)
                attention_mask = enc_hyp["attention_mask"].to(device)

            # Compute logits and log-probs
            with torch.no_grad():
                logits = model(input_ids, attention_mask=attention_mask).logits
                gather_ids = hyp_input_ids[:, 1:].unsqueeze(-1)
                scores_mask = enc_hyp["attention_mask"][..., 1:].to(device)
                if self.prompt:
                    gather_ids = hyp_input_ids.unsqueeze(-1)
                    scores_mask = enc_hyp["attention_mask"].to(device)
                    logits = logits[:, -hyp_input_ids.shape[1] - 1:-1, :]

                log_probs = torch.log_softmax(logits, dim=-1)
                llm_scores = torch.gather(log_probs, dim=-1, index=gather_ids).squeeze()
                llm_scores = llm_scores * scores_mask
                llm_scores = llm_scores.sum(dim=1)

            scores_buffer.extend(llm_scores.tolist())
            total_tokens += int(attention_mask.sum().item())

            if batch_count % 1000 == 0:
                print(f"  → Completed {batch_count:,} batches; Tokens so far: {total_tokens:,}")

        # Iterate records
        for seq_tag, n_best in d_rec.items():
            out_file.write(f"{seq_tag!r}: [\n")
            batch_lines, batch_prompt, scores_buffer = [], [], []

            for original_score, hyp in n_best:
                if self.prompt:
                    batch_prompt.append(self.prompt.strip().lower() if self.lower_case else self.prompt.strip())
                hyp = raw_text_from_bpe_seq(hyp.split())
                line = hyp.strip().lower() if self.lower_case else hyp.strip()
                if not line: continue
                eos_symbol = (" " + tokenizer.eos_token) if self.eos_symbol == "eos" else self.eos_symbol
                batch_lines.append(line + eos_symbol)

                lines_seen += 1
                if lines_seen % log_every == 0:
                    print(f"[Line {lines_seen:,}/{total_lines}] {100 * lines_seen / total_lines:.2f}% processed…")
                    _report_dev_memory_stats()

                if len(batch_lines) == self.batch_size:
                    _process_batch(batch_lines, batch_prompt, scores_buffer)
                    batch_lines, batch_prompt = [], []

            # leftover
            if batch_lines:
                _process_batch(batch_lines, batch_prompt, scores_buffer)

            # reorder and select top
            hyps = [x[1] for x in n_best]
            ori_scores = [x[0] for x in n_best]
            reorder = list(zip(scores_buffer, ori_scores, hyps))
            reorder.sort(key=lambda x: x[1] + self.scale * x[0], reverse=True)
            if batch_count%1000 == 0:
                print(f"------Check reorder----------\n{reorder}\n\n")
            out_file.write(f"  ({reorder[0][1]!r}, {reorder[0][2]!r}),\n")
            out_file.write("],\n")

        # cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        import gc;gc.collect()
        print("Finished and cleaned up.")

        out_file.write("}\n")
        out_file.close()



class SummarizeLLMPPLJob(Job):
    """
    Summarize LLM PPLs into per-dataset tables.

    Inputs:
      - ppls_by_ds: Mapping[ds_name][model_name][ctx_len_limit] -> tk.Path to a PPL .txt file
        with lines like:
            Average bpe/word length ratio:: 1.23
            bpe level: 6.470073703341919
            Perplexity: 9.943061786014486

      - ctx_order: column order for context length limits (e.g., [0, 1000, 3000, 5000, None])
      - value_source: "perplexity" (word-level) or "bpe" (BPE-level)
      - float_fmt: format string for numbers
    """

    def __init__(
        self,
        ppls_by_ds: Dict[str, Dict[str, Dict[Optional[int], tk.Path]]],
        ctx_order: Iterable[Optional[int]] = (0, 1000, 3000, 5000, None),
        value_source: str = "perplexity",  # "perplexity" or "bpe"
        float_fmt: str = "{:.2f}",
        title: str = "LLM Perplexity Summary",
    ):
        self.ppls_by_ds = ppls_by_ds
        self.ctx_order = list(ctx_order)
        self.value_source = value_source.lower()
        assert self.value_source in {"perplexity", "bpe"}, "value_source must be 'perplexity' or 'bpe'"
        self.float_fmt = float_fmt
        self.title = title

        self.out_index_md = self.output_path("index.md")
        self.out_per_ds_md = {}
        self.out_per_ds_csv = {}
        for ds in sorted(self.ppls_by_ds.keys()):
            self.out_per_ds_md[ds] = self.output_path(f"{ds}.md")
            self.out_per_ds_csv[ds] = self.output_path(f"{ds}.csv")

    def tasks(self):
        yield Task("run", mini_task=True)

    def _safe_load_ppl_from_txt(self, path: tk.Path) -> Optional[float]:
        """Parse your txt format."""
        try:
            txt = uopen(path, "rt").read()
        except Exception:
            return None

        # Accept both "Perplexity:" and "bpe level:" (case-insensitive, tolerant of spaces)
        if self.value_source == "perplexity":
            m = re.search(r"(?i)^\s*Perplexity\s*:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*$", txt, re.M)
        else:  # "bpe"
            m = re.search(r"(?i)^\s*bpe\s*level\s*:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*$", txt, re.M)

        if not m:
            return None
        try:
            val = float(m.group(1))
            return val if math.isfinite(val) else None
        except Exception:
            return None

    def _ctx_label(self, ctx: Optional[int]) -> str:
        return "None" if ctx is None else str(ctx)

    def _fmt(self, v: Optional[float]) -> str:
        return self.float_fmt.format(v) if (v is not None and math.isfinite(v)) else ""

    def run(self):
        # Index
        with uopen(self.out_index_md, "wt") as fidx:
            label = "Word-level Perplexity" if self.value_source == "perplexity" else "BPE-level Perplexity"
            fidx.write(f"# {self.title} — {label}\n\n")
            fidx.write("Datasets summarized here:\n\n")
            for ds in sorted(self.ppls_by_ds.keys()):
                fidx.write(f"- [{ds}]({ds}.md)\n")
            fidx.write("\n")

        # Per-dataset outputs
        for ds in sorted(self.ppls_by_ds.keys()):
            models = sorted(self.ppls_by_ds[ds].keys())
            values = {m: {} for m in models}
            for m in models:
                for ctx in self.ctx_order:
                    p = self.ppls_by_ds[ds][m].get(ctx)
                    values[m][ctx] = self._safe_load_ppl_from_txt(p) if p is not None else None

            headers = ["Model"] + [self._ctx_label(c) for c in self.ctx_order]
            # Markdown
            with uopen(self.out_per_ds_md[ds], "wt") as fmd:
                label = "Word-level Perplexity" if self.value_source == "perplexity" else "BPE-level Perplexity"
                fmd.write(f"# {ds} — {label} by context length limit\n\n")
                fmd.write("| " + " | ".join(headers) + " |\n")
                fmd.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                for m in models:
                    row = [m] + [self._fmt(values[m].get(c)) for c in self.ctx_order]
                    fmd.write("| " + " | ".join(row) + " |\n")

            # CSV
            with uopen(self.out_per_ds_csv[ds], "wt") as fcsv:
                fcsv.write(",".join(headers) + "\n")
                for m in models:
                    row = [m] + [self._fmt(values[m].get(c)) for c in self.ctx_order]
                    fcsv.write(",".join(row) + "\n")

# def general_test1():
#     from i6_experiments.users.zhang.experiments.decoding.lm_rescoring import LmRescoringJob
#     # lm_text = [#get_librispeech_normalized_lm_data(),
#     #            get_train_corpus_text(),
#     #             _get_test_corpus_text()]
#     for llm_name in ["Llama-3.2-1B",
#                      #"Llama-3.1-8B",
#                      ]:
#         dl_model = DownloadHuggingFaceRepoJob(model_id="meta-llama/" + llm_name)
#         tk.register_output(f"llm/{llm_name}", dl_model.out_hub_cache_dir)
#         # lower_case = False
#         # llm_scale = 0.2
#         # resco_job = HuggingFaceLmRescoringJob(
#         #     model_dir=dl_model.out_hub_cache_dir,
#         #     weight=llm_scale,
#         #     recog_out_file=tk.Path("work/i6_core/returnn/search/SearchOutputRawReplaceJob.a5gb36CLt4N6/output/search_results.py.gz"),
#         #     llm_name=llm_name,
#         #     lower_case=lower_case,
#         # )
#         # resco_job.add_alias("lm/" + llm_name + "/rescoring_"  + f"w{llm_scale}".replace(".","") + ("low" if lower_case else ""))
#         # tk.register_output(
#         #     llm_name + "/librispeech-rescoring_test" + f"w{llm_scale}".replace(".","") + ("low" if lower_case else ""),
#         #     resco_job.out_file)
#         llm_scale = 0.2
#         resco_job = LmRescoringJob(
#             recog_out_file=tk.Path(
#                 "work/i6_core/returnn/search/SearchRemoveLabelJob.HxKTed4GQc38/output/search_results.py.gz"),
#             lm_cfg={
#                 "lm_type": "HuggingFaceLm",
#                 "model_dir": dl_model.out_hub_cache_dir,
#                 "batch_size": 8,
#                 "weight": llm_scale,
#             }
#         )
#         resco_job.rqmt["gpu_mem"] = 48
#         resco_job.add_alias("lm/" + llm_name + "/rescoring_"  + f"w{llm_scale}".replace(".",""))
#         tk.register_output(
#             llm_name + "/librispeech-rescoring_test" + f"w{llm_scale}".replace(".",""),
#             resco_job.out_file)
#         for ds_name in [#"dev-clean",
#                         # "dev-other",
#                         #"test-clean",
#                         "test-other",
#                         #"train",
#                         ]:
#             for lower_case in [#True,
#                                False,
#                                ]:
#                 # for eos_name, eos_symbol in {#"newline": " \n",
#                 #                             "period":".",
#                 #                              #"eos": "eos"
#                 #                              }.items():
#                 ppl_job = HuggingFaceLmPerplexityJob(
#                     model_dir=dl_model.out_hub_cache_dir,
#                     text_file=[get_test_corpus_text(keys = [ds_name])] if ds_name != "train" else [get_train_corpus_text()],
#                     llm_name=llm_name,
#                     lower_case=lower_case,
#                     batch_size=1,
#                 )
#                 ppl_job.add_alias("lm/" + llm_name + "/ppl/librispeech_" + ds_name + ("low" if lower_case else ""))# + eos_name)
#                 tk.register_output("ppl/" + llm_name + "/librispeech-" + ds_name + ("low" if lower_case else "") +  "-ppl", ppl_job.out_ppl)

def py():
    ds_names = ["test_set.ES_ES.f16kHz.eval_voice_call-v3"]
    ppls = dict()
    ppls_by_ds = {}
    for ds_name in ds_names:
        text_file = ES_get_corpus_text_dict(key=ds_name)
        for model_id in ["meta-llama/Llama-3.2-1B",
                      "meta-llama/Llama-3.1-8B",
                      "Qwen/Qwen3-1.7B-Base",
                    #"Qwen/Qwen3-4B-Base",
                      "Qwen/Qwen3-8B-Base",
                      "microsoft/phi-4",
                         ]:
            batch_size = LLM_Batch_size[model_id] // 6
            name = os.path.basename(model_id)
            model = DownloadHuggingFaceRepoJob(model_id=model_id)
            tk.register_output(model_id, model.out_hub_cache_dir)
            ppls[name] = dict()
            for ctx_len_limit in [0, 1000, 3000]:#, 5000, None]:
                ppl_job = HuggingFaceLmPerplexityJobV2(
                    model_dir=model.out_hub_cache_dir,
                    text_file=[text_file], # get_test_corpus_text(keys=[ds_name])
                    llm_name=model_id,
                    batch_size=max(batch_size//2,1),
                    lower_case=USE_LOWER_CASE,
                    word_ppl=True,
                    prompt=None,
                    eos_symbol="\n",
                    use_prev_context=True,
                    context_len_limit=ctx_len_limit,
                )
                ppl_job.rqmt.update(LLM_rqmt[model_id])
                ppl_job_name = f"ppl/{name}/{ds_name}" + f"{'low'}{f'_prev{str(ctx_len_limit)}'}"
                ppl_job.add_alias(ppl_job_name)
                ppls[name].update({ds_name: ppl_job.out_ppl})
                name_ext = ""
                name_ext += f"prev_{ctx_len_limit}"
                name_ext += f"{'_low'}"
                # tk.register_output(
                #     "ppl/" + name + f"/{task_name}-" + ds_name + name_ext + "-ppl",
                #     ppl_job.out_ppl)
                # Fill nested structure
                ppls_by_ds.setdefault(ds_name, {}).setdefault(name, {})[ctx_len_limit] = ppl_job.out_ppl

    summary_job = SummarizeLLMPPLJob(
        ppls_by_ds=ppls_by_ds,
        ctx_order=[0, 1000, 3000, 5000, None],
        value_source="perplexity",  # word-level
        float_fmt="{:.3f}",
        title="LLM Perplexity Summary",
    )
    summary_job.add_alias("ppl/LLM_ctx_PPL_summary_job")
    # from i6_experiments.users.zhang.experiments.decoding.lm_rescoring import LmRescoringJob
    # # lm_text = [#get_librispeech_normalized_lm_data(),
    # #            get_train_corpus_text(),
    # #             _get_test_corpus_text()]
    # for llm_name in [
    #                 # "Qwen/Qwen3-4B-Base",
    #                 # "Qwen/Qwen3-0.6B-Base",
    #                 "Qwen/Qwen3-1.7B-Base",
    #                 "meta-llama/Llama-3.2-1B",
    #                 #"meta-llama/Llama-3.1-8B",
    #                  ]:
    #     dl_model = DownloadHuggingFaceRepoJob(model_id=llm_name)
    #     for ds_name in [  # "dev-clean",
    #         "dev-other",
    #         # "test-clean",
    #         #"test-other",
    #         # "train",
    #     ]:
    #         lower_case = True
    #         for ctx_lines in [#10, 5, 3,
    #                           1, 0]:
    #             context = get_fix_context_file(size=ctx_lines)
    #             for eos_name, eos_symbol in {"newline": "\n",
    #                                         #"period":".",
    #                                          #"eos": "eos",
    #                                          #"" : "",
    #                                          }.items():
    #                 add_eos_to_completion = True
    #                 if eos_name == "period" and ctx_lines >=20 :
    #                     continue
    #                 if isinstance(context, list) and eos_name: # eos is joined
    #                     if eos_name != "newline":
    #                         continue
    #                 batch_size = {100: 1, 50: 1, 30: 2, 10:4, 1:LLM_Batch_size[llm_name], 0:LLM_Batch_size[llm_name]}[ctx_lines]
    #                 if eos_name == "eos" and ctx_lines > 10:
    #                     batch_size = 1
    #
    #                 if isinstance(context, str):
    #                     context = tk.Path(context)
    #                 ppl_job = HuggingFaceLmPerplexityJobV2(
    #                     model_dir=dl_model.out_hub_cache_dir,
    #                     prompt=context,
    #                     text_file=[get_test_corpus_text(keys=[ds_name])] if ds_name != "train" else [
    #                         get_train_corpus_text()],
    #                     llm_name=llm_name,
    #                     eos_symbol=eos_symbol,
    #                     add_eos_to_completion=add_eos_to_completion,
    #                     lower_case=lower_case,
    #                     word_ppl=True,
    #                     batch_size=batch_size,
    #                 )
    #                 ppl_job.rqmt.update(LLM_rqmt[llm_name])
    #                 alias_name = ds_name + "_" + eos_name + ("low" if lower_case else "") + f"_ctx{ctx_lines}" + ("str" if context == CTX_10_STR else "") + ("eos_comp" if add_eos_to_completion else "")
    #                 ppl_job.add_alias(
    #                     "lm/" + llm_name + "/ppl/librispeech_" + alias_name)  # + eos_name)
    #                 tk.register_output(
    #                     "ppl/" + llm_name + "/librispeech-" + alias_name + "-ppl_V2",
    #                     ppl_job.out_ppl)
