from __future__ import annotations
import functools
from typing import Optional, Any, Dict, List, Sequence, TYPE_CHECKING, Iterable
import re
import math

from i6_experiments.users.zhang.experiments.exp_wer_ppl import LLM_WITH_PROMPT, LLM_WITH_PROMPT_EXAMPLE



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



import torch

if TYPE_CHECKING:
    pass
from i6_core.util import uopen

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_fix_context_file(size: int, task_name: str = "LBS") -> Optional[tk.Path]:
    if size < 1:
        return None
    path = f"llm_fixed_ctx/fix_ctx_forLLM_{task_name}_{size}.txt" #/u/haoran.zhang/setups/2024-12-16--lm-ppl/
    if os.path.isfile(path):
        return tk.Path(path)
    else:
        raise FileNotFoundError(path)

USE_LOWER_CASE = True
CTX_10_STR = ["THEY SAID IT TO THE END VERSE ANSWERING VERSE AND THE PRAYER OF THE KING POET STILLED THE THROBBING OF HURTS TOO DEEP TO HEAL\nMARY DID YOU EVER THINK WHAT YOU WOULD DO IF YOU HAD TO LIVE ON JUST A FEW CENTS A DAY\nTRUE COAL HAD TO BE BROUGHT FROM SOME DISTANCE AND THERE WAS A GREAT NEED OF REALLY SKILLED LABOR\nWELL THEN WE'LL TALK ABOUT BEAUTIFUL WOMEN IF YOU PREFER\nWAS IT ON THE STAGE THAT YOU FOUND YOUR MOST INTENSE JOYS YOUR TRUE HAPPINESS\nIT WAS WELL KNOWN OF COURSE TO JEANJEAN THAT HIS PRISONER HAD BEEN GUILTY OF THE OFFENCE FOR WHICH HE HAD ARRESTED HIM AND THE COUP WAS QUITE EASY\nHE SNATCHED THE WHIP AND STRUCK THE CONDEMNED MAN WITH IT AS HIGH UP AS HE COULD REACH MAKING A GREAT WELT ACROSS HIS BARE STOMACH\nWHAT DID THAT CREEP WANT\nWE HAVE COME DOWN HERE TO DO A LITTLE PROSPECTING AND WERE JUST RIDING AROUND A BIT TO TAKE A LOOK AT THE COUNTRY\nBUT BECAUSE MANY OF THE SIMPLE IDEAS THAT MAKE UP OUR SPECIFIC IDEAS OF SUBSTANCES ARE POWERS WHICH LIE NOT OBVIOUS TO OUR SENSES IN THE THINGS AS THEY ORDINARILY APPEAR THEREFORE IN THE SIGNIFICATION OF OUR NAMES OF SUBSTANCES SOME PART OF THE SIGNIFICATION WILL BE BETTER MADE KNOWN BY ENUMERATING THOSE SIMPLE IDEAS THAN BY SHOWING THE SUBSTANCE ITSELF"]
PROMPT = ["THIS IS A TEXT DATA SOURCED FROM PUBLIC DOMAIN AUDIOBOOKS\nIT REPRESENTS THE DOMAIN OF READ, AND SCRIPTED SPEECH"]
EXAMPLE = ["I SAY ADVERSARIES FOR ON RECALLING SUCH PROUD MEMORIES WE SHOULD AVOID THE WORD ENEMIES WHOSE HOSTILE SOUND PERPETUATES THE ANTAGONISMS AND STRIFE OF NATIONS SO IRREMEDIABLE PERHAPS SO FATEFUL AND ALSO SO VAIN"]
LLM_Batch_size = {"meta-llama/Llama-3.2-1B": 1,#40,#40*6,
                  #"meta-llama/Llama-3.1-8B": 1,#50*6, #14*9, # actual 21 batch size-> ~ 40GB peak usage#  be at least 3 times larger from 10*3
                  #"Qwen/Qwen3-0.6B-Base": 51,
                  "Qwen/Qwen3-1.7B-Base": 1,#40,#40*6,#15 has peak 19GB on 48G, so can be at least doubled
                  #"Qwen/Qwen3-4B-Base":24,
                  #"Qwen/Qwen3-8B-Base":5*3,
                  #"microsoft/phi-4": 1,#30*6, #Can be 24 on 80GB with 100 ctx(peak 40 with 14， peaK 48 with 30), so even 50*6 should be fine
                  #"mistralai/Mistral-7B-v0.3": 4,
                  } # Keys of this determines which LLM will be built by lm_getter

LLM_Batch_size_PPL = {"meta-llama/Llama-3.2-1B": 50,
                  "meta-llama/Llama-3.1-8B": 10,
                  "Qwen/Qwen3-0.6B-Base": 51,
                  "Qwen/Qwen3-1.7B-Base": 30,
                  "Qwen/Qwen3-4B-Base":24,
                  "Qwen/Qwen3-8B-Base":20,
                  "microsoft/phi-4": 50,
                  }

#This rqmt is only used by ppl job
LLM_rqmt = {"meta-llama/Llama-3.2-1B": {"time": 2, "cpu": 1, "mem": 16, "gpu": 1, "gpu_mem": 48},
            "meta-llama/Llama-3.1-8B": {"time": 2, "cpu": 1, "mem": 40, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-0.6B-Base": {"time": 1, "cpu": 1, "mem": 12, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-1.7B-Base": {"time": 2, "cpu": 1, "mem": 20, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-4B-Base":{"time": 1, "cpu": 1, "mem": 25, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-8B-Base":{"time": 2, "cpu": 3, "mem": 40, "gpu": 1, "gpu_mem": 48},
                  #"mistralai/Mistral-7B-v0.3": 4,
            "Qwen/Qwen2-0.5B": {"time": 1, "cpu": 1, "mem": 25, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen2-1.5B": {"time": 1, "cpu": 1, "mem": 25, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen2-7B": {"time": 1, "cpu": 1, "mem": 40, "gpu": 1, "gpu_mem": 80},
            "microsoft/phi-4":{"time": 2, "cpu": 1, "mem": 65, "gpu": 1, "gpu_mem": 80},
            }




def get_raw_text_func_ES_spm(seq:list)-> str:
    # import sentencepiece as spm
    #
    # sp = spm.SentencePieceProcessor(model_file="/nas/models/asr/artefacts/subword_units/sentencepiece/ES/2025-04-spm_10240-mbw/10240-nmt_nfkc_cf.spm")
    # pieces = ["▁ac", "tion"]
    # return sp.decode_pieces(seq)  # -> "action"
    if len(seq) == 0:
        return " ".join(seq)
    if "▁" not in " ".join(seq): # Should further do replacement but just print a warning
        print(f"Warning: Passed likely a non spm sequence: \n{seq}\n to get_raw_text_func_ES_spm")
        # return " ".join(seq)
    return " ".join(seq).replace("<sep>", "▁[noise]").replace(" ", "").replace("▁", " ")


def get_raw_text_func_bpe(seq:list)-> str:
    # import sentencepiece as spm
    #
    # sp = spm.SentencePieceProcessor(model_file="/nas/models/asr/artefacts/subword_units/sentencepiece/ES/2025-04-spm_10240-mbw/10240-nmt_nfkc_cf.spm")
    # pieces = ["▁ac", "tion"]
    # return sp.decode_pieces(seq)  # -> "action"
    if len(seq) == 0:
        return " ".join(seq)
    if "@@" not in " ".join(seq): # Should further do replacement but just print a warning
        print(f"Warning: Passed likely a non bpe sequence: \n{seq}\n to get_raw_text_func_bpe")
        # return " ".join(seq)
    return " ".join(seq).replace("@@ ", "")


def get_prompt(LLM_FXIED_CTX, LLM_FXIED_CTX_SIZE, task_name:str = "LBS"):
    prompt = None
    if LLM_FXIED_CTX:
        prompt = tk.Path(get_fix_context_file(LLM_FXIED_CTX_SIZE, task_name=task_name))
    elif LLM_WITH_PROMPT:
        prompt = PROMPT
        if LLM_WITH_PROMPT_EXAMPLE:
            prompt += [f"This is one sentence as an example: {EXAMPLE[0]}"]
        prompt = prompt
    return prompt

# TODO: make this vocab dependent instead task specific
get_raw_text_func_dict = {"ES": get_raw_text_func_ES_spm, "LBS":get_raw_text_func_bpe}

#@functools.cache
def get_llm(model_ids: List[str], batch_sizes: List[int] = None, word_ppl: bool = False, task_name: str = "LBS") -> tuple[
    dict[Any, dict[str, int | str | Any]], dict[Any, Any]]:
    if task_name == "LBS":
        from i6_experiments.users.zhang.experiments.LLM_LBS_exp import CTX_LEN_LIMIT, LLM_FXIED_CTX, LLM_FXIED_CTX_SIZE, LLM_PREV_ONE_CTX, CHEAT_CTX
        ds_names = ["test-other", "dev-other","test-clean", "dev-clean"]
    elif task_name == "ES":
        from i6_experiments.users.zhang.experiments.LLM_apptek_exp import CTX_LEN_LIMIT, LLM_FXIED_CTX, LLM_FXIED_CTX_SIZE, LLM_PREV_ONE_CTX, CHEAT_CTX
        from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
        ds_names = list(set(DEV_KEYS + TEST_KEYS))
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    #main_ppl_mesure_name = ["test-other"] if task_name=="LBS" else (DEV_KEYS + TEST_KEYS)[0]
    prompt = get_prompt(LLM_FXIED_CTX, LLM_FXIED_CTX_SIZE,task_name=task_name)
    llms = dict()
    ppls = dict()
    if not batch_sizes:
        batch_sizes = [LLM_Batch_size[model_id] for model_id in model_ids]
    prompt = [prompt for _ in model_ids]
    assert len(model_ids) == len(batch_sizes)
    name_ext = f"{LLM_FXIED_CTX_SIZE}" if LLM_FXIED_CTX else ""
    name_ext += f"prev_{CTX_LEN_LIMIT}" if LLM_PREV_ONE_CTX else ""
    name_ext += f"{'prompt' if LLM_WITH_PROMPT else ''}{'_example' if LLM_WITH_PROMPT_EXAMPLE else ''}{'_low' if USE_LOWER_CASE else ''}"

    for model_id, prompt, batch_size in zip(model_ids, prompt, batch_sizes):
        # if LLM_FXIED_CTX:
        #     batch_size = batch_size // 3
        # if LLM_FXIED_CTX_SIZE > 10:
        #     batch_size = 1
        # if LLM_PREV_ONE_CTX:
        #     batch_size = batch_size // 6
        name = os.path.basename(model_id)
        model = DownloadHuggingFaceRepoJob(model_id=model_id)
        tk.register_output(model_id, model.out_hub_cache_dir)
        lm_cfg = {"model_dir": model.out_hub_cache_dir, "batch_size": max(batch_size,1),
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
        lm_cfg.update({"get_raw_text_func": get_raw_text_func_dict[task_name]})
        llms.update({name: lm_cfg})
        ppls[name] = dict()
        for ds_name in ds_names:
            if task_name=="LBS":
                text_file = _get_corpus_text_dict(key=ds_name)
            elif task_name=="ES":
                from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import \
                    get_corpus_text_dict as ES_get_corpus_text_dict
                text_file = ES_get_corpus_text_dict(key=ds_name)
            else:
                raise ValueError(f"Unknown task name: {task_name}")
            ppl_job = HuggingFaceLmPerplexityJobV2(
                model_dir=model.out_hub_cache_dir,
                text_file=[text_file], # get_test_corpus_text(keys=[ds_name])
                llm_name=model_id,
                batch_size=max(LLM_Batch_size_PPL[model_id],1), #max(ppl_batch_size//2,1), Currently the implementation is somehow not batch invariant
                lower_case=USE_LOWER_CASE,
                add_eos_to_completion=True,
                word_ppl=word_ppl,
                prompt=prompt,
                eos_symbol="\n",
                use_prev_context=LLM_PREV_ONE_CTX and "common_voice" not in ds_name,
                context_len_limit=(CTX_LEN_LIMIT if "common_voice" not in ds_name else 0)
                if LLM_PREV_ONE_CTX else None,
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
                "ppl/" + name + f"/{task_name}-" + ds_name + name_ext + f"-ppl_batch{max(LLM_Batch_size_PPL[model_id],1)}",
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


class HuggingFaceLmPerplexityJobV2(Job):
    """Compute perplexity for a HuggingFace Transformer LLM model on given text dict corpus."""
    
    def __init__(self, *, model_dir: tk.Path, prompt: [List[str] | tk.Path] = None, text_file: List[tk.Path], batch_size: int = None,
                 llm_name: str, lower_case:bool = False, context_len_limit: int = None, eos_symbol: str = "", word_ppl: bool = False,
                 add_eos_to_completion: bool = True, use_prev_context: bool = False, version:int = 9):
        super().__init__()
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

        # setting for use fixed context
        if not use_prev_context:
            if isinstance(prompt, tk.Path):
                with open(prompt.get_path(), "r", encoding="utf-8") as f:
                    prompt = [line.strip() for line in f.readlines()]
            if prompt:
                prompt +=  [""]  # +[""] So that for last prompt(or only one prompt) it also has eos
                self.prompt = self.delimiter.join(prompt)
        self.word_ppl = word_ppl
        self.out_ppl = self.output_path("ppl")
        self.rqmt = {"time": 4, "cpu": 1, "mem": 8 + self.batch_size//2, "gpu": 1}


    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _ctx_over_limit(self):
        if self.context_len_limit is None:
            return False
        return len(self.delimiter.join(self.prompt_buffer + [""]).split()) > self.context_len_limit

    def _update_prompt(self, new_prompt: str):
        assert "@@" not in new_prompt and "▁" not in new_prompt
        self.prompt_buffer += [new_prompt]
        while self._ctx_over_limit():
            if len(self.prompt_buffer) == 1: # Keep at least one sentence
                break
            self.prompt_buffer.pop(0)
        self.prompt = self.prompt_buffer.copy()
        self.prompt += [""]  # +[""] So that for last prompt(or only one prompt) it also has eos
        self.prompt = self.delimiter.join(self.prompt)

    def _clear_prompt(self):
        torch.cuda.empty_cache()
        self.prompt_buffer = []
        self.prompt = ""

    def _score_batch(self, batch_lines, batch_prompt, tokenizer, model, device):
        """A simpler version:
         1. will pad between prompt and completion
         2. will skip scoring of first token if there is no BOS added by default tokenizer behaviour"""

        use_prompt = bool(batch_prompt) and any(p.strip() for p in batch_prompt)
        if not all(p.strip() for p in batch_prompt):
            print(f"Warning: Not all prompt are non empty{batch_prompt}")
        enc_hyp = tokenizer(
            batch_lines,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            add_special_tokens=False if use_prompt else True,
        )
        hyp_input_ids = enc_hyp["input_ids"].to(device)

        # Prepare inputs
        if use_prompt:
            enc_prompt = tokenizer(
                batch_prompt,
                return_tensors="pt",
                padding=True, # No need to pad actually, since all prompts are same
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
            logits = model(input_ids, attention_mask=attention_mask).logits.to(torch.float64)
            gather_ids = hyp_input_ids[:, 1:].unsqueeze(-1)
            scores_mask = enc_hyp["attention_mask"][..., 1:].to(device)
            if use_prompt:
                gather_ids = hyp_input_ids.unsqueeze(-1)
                scores_mask = enc_hyp["attention_mask"].to(device)
                logits = logits[:, -hyp_input_ids.shape[1] - 1:-1, :]

            log_probs = torch.log_softmax(logits, dim=-1)
            llm_scores = torch.gather(log_probs, dim=-1, index=gather_ids).squeeze()
            llm_scores = llm_scores * scores_mask
            token_count = int(scores_mask.sum().item())
            llm_scores = llm_scores.to(torch.float64)
            nll = llm_scores.sum().cpu().item()
        return nll, token_count

    def run(self):
        import math
        import time
        import returnn.util.basic as util # noqa
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
        @torch.no_grad()
        def _score_batch(batch_lines, batch_prompt, tokenizer, model, device):
            #return self._score_batch(batch_lines, batch_prompt, tokenizer, model, device)
            """
            Manual log-softmax + gather scorer that:
              - Use EOS as BOS if there is no one exist,
              - When no prompt given, Add BOS anyway even if tokenizer does not add it.
              - concatenates prompt+hyp per example (no interior PADs),
              #- Optionally provide position id for each real token,
              - pads once across the batch,
              - indexes the correct time steps for each hypothesis token,
              - sums exact token log-probs and returns (nll, token_count, total_bpe_tokens).
            """
            # 0) Decide whether there is any *non-empty* prompt in this batch
            use_prompt = bool(batch_prompt) and any(p.strip() for p in batch_prompt)
            if not use_prompt:
                return self._score_batch(batch_lines, batch_prompt, tokenizer, model, device)
            if not all(p.strip() for p in batch_prompt):
                print(f"Warning: Not all prompt are non empty{batch_prompt}")
            if not use_prompt:
                # normalize to empty prompts list of same length (keeps code uniform)
                batch_prompt = [""] * len(batch_lines)

            # 1) Tokenize WITHOUT padding (get raw lists)
            enc_hyp = tokenizer(
                batch_lines,
                return_tensors=None,
                padding=False,
                truncation=True,
                add_special_tokens=False,  # never add bos here
                max_length=tokenizer.model_max_length,
            )
            enc_prm = tokenizer(
                batch_prompt,
                return_tensors=None,
                padding=False,
                truncation=True,
                add_special_tokens=True, # Always try to add special tokens for prompt
                max_length=tokenizer.model_max_length,
            )

            bos_id = getattr(tokenizer, "bos_token_id", None)
            assert bos_id is not None

            # 2) Concatenate per example; record prompt/hyp lengths
            examples = []
            prm_lens, hyp_lens = [], []
            for prm_ids, hyp_ids in zip(enc_prm["input_ids"], enc_hyp["input_ids"]):
                # If prompt is empty, make the *prompt part* = [BOS].
                if len(prm_ids) == 0 and bos_id is not None:
                    prm_ids = [bos_id]  # BOS lives in the prompt/context segment
                prm_lens.append(len(prm_ids))
                hyp_lens.append(len(hyp_ids))
                examples.append({"input_ids": prm_ids + hyp_ids})

            # 3) Pad once across the batch
            batch = tokenizer.pad(examples, return_tensors="pt")
            input_ids = batch["input_ids"].to(device)  # [B, T]
            attn_mask = batch["attention_mask"].to(device)  # [B, T]
            B, T = input_ids.shape

            # 4) Forward pass
            with torch.no_grad():
                # [B, T] attention mask, 1 = real token, 0 = PAD
                # position_ids = (attn_mask.cumsum(dim=1) - 1).clamp(min=0)
                #
                # for i, row in enumerate(position_ids):
                #     if row.sum() == 0:
                #         print(position_ids)
                #         print(batch_lines[max(i-2,0):i+1])
                #         print(input_ids[max(i-2,0):i+1])
                logits = model(input_ids=input_ids, attention_mask=attn_mask,
                               #position_ids=position_ids
                               ).logits  # [B, T, V]

            # 5) Build per-row time indices for hypothesis predictions
            # For causal LMs: logits[:, t] predicts token at t+1.
            # Hyp tokens live at absolute positions:    pos = prm_len + j, j=0..hyp_len-1
            # We need predicting steps (time indices):  t = pos - 1 = (prm_len - 1) + j
            pl = torch.tensor(prm_lens, device=logits.device)  # [B]
            hl = torch.tensor(hyp_lens, device=logits.device)  # [B]
            maxH = int(hl.max().item())
            steps = torch.arange(maxH, device=logits.device)  # [maxH]
            time_idx = (pl - 1)[:, None] + steps[None, :]  # [B, maxH]
            valid_mask = steps[None, :] < hl[:, None]  # [B, maxH]
            # keep inside bounds after truncation/padding (safety)
            time_idx = time_idx.clamp(min=0, max=T - 1)

            # 6) Gather the logits at those times: [B, maxH, V]
            logits_hyp = torch.take_along_dim(
                logits, time_idx.unsqueeze(-1).expand(-1, -1, logits.size(-1)), dim=1
            )

            # 7) Build a padded tensor of hyp token ids to gather
            hyp_only_examples = [{"input_ids": ids} for ids in enc_hyp["input_ids"]]
            hyp_pad = tokenizer.pad(hyp_only_examples, return_tensors="pt")
            hyp_ids = hyp_pad["input_ids"].to(logits.device)  # [B, maxH]

            # 8) Log-softmax and gather per-token log-probs
            log_probs = torch.log_softmax(logits_hyp, dim=-1)  # [B, maxH, V]
            gather_ids = hyp_ids.unsqueeze(-1)  # [B, maxH, 1]
            tok_logp = torch.take_along_dim(log_probs.float(), gather_ids, dim=-1).squeeze(-1)  # [B, maxH]

            # 9) Mask out padded hyp slots and any positions we clamped away
            hyp_mask = valid_mask & (hyp_ids != tokenizer.pad_token_id)
            tok_logp = tok_logp * hyp_mask

            # 10) Sum and count
            nll = tok_logp.to(torch.float64).sum().cpu().item()  # scalar (log-prob sum in nats)
            token_count = int(hyp_mask.sum().item())  # exact # of scored tokens
            total_bpe_tokens = int(sum(len(x) for x in enc_hyp["input_ids"]))  # diagnostic

            # Sanity guard: make sure dims align (helps catch future drift)
            assert logits_hyp.shape[:2] == hyp_ids.shape[:2], (logits_hyp.shape, hyp_ids.shape)

            return nll, token_count#, total_bpe_tokens

        def safe_score_batch(batch_lines, batch_prompt, tokenizer, model, device):
            """
            Returns (total_nll: torch.Tensor on device, total_tok_count: int).
            Falls back by halving the batch on CUDA OOM until batch size = 1.
            """
            import gc
            def _is_oom(e):
                # PyTorch 2.0+: torch.cuda.OutOfMemoryError exists; older versions raise RuntimeError
                return (hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError)) \
                    or ("out of memory" in str(e).lower())

            def _helper(lines, prompts):
                try:
                    return _score_batch(lines, prompts, tokenizer, model, device)
                except Exception as e:
                    if _is_oom(e):
                        # Free what we can and try smaller batches
                        del e
                        torch.cuda.empty_cache()
                        gc.collect()
                        if len(lines) == 1:
                            # Can't split further; re-raise a clearer error
                            raise RuntimeError("CUDA OOM even with batch size = 1")  # noqa: B904
                        mid = len(lines) // 2
                        print(f"\nOOM: Split to {mid}/{self.batch_size}")
                        _report_dev_memory_stats()
                        nll1, tok1 = _helper(lines[:mid], prompts[:mid])
                        print(f"\nOOM: Second half {mid}/{self.batch_size}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        nll2, tok2 = _helper(lines[mid:], prompts[mid:])
                        # Sum results; nll is a scalar tensor, tok_count is int or scalar tensor
                        total_nll = nll1 + nll2
                        total_tok = tok1 + tok2
                        return total_nll, total_tok
                    else:
                        # Not an OOM; bubble up
                        raise

            return _helper(batch_lines, batch_prompt)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"


        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_time = time.time()
        print("Loading model...")

        model_path = get_content_dir_from_hub_cache_dir(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "bos_token_id", None) is None:
            tokenizer.bos_token_id = tokenizer.eos_token_id
            print(f"Warning: Will use eos as bos if not using self.score_batch")

        print(f"\nTokenizer_max_length:{tokenizer.model_max_length}\n")

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        if dtype != torch.bfloat16:
            print(f"!!! Warning : Using {dtype} dtype!")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,  # load directly in low precision
            device_map={"": 0},  # put the whole model on cuda:0 (single GPU)
            low_cpu_mem_usage=True,  # stream weights, smaller CPU peak
            attn_implementation="flash_attention_2",#"flash_attention_2",  # faster runtime; f
        )

        #model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        model.eval()

        print(getattr(model, "_attn_implementation", None) or getattr(model.config, "_attn_implementation", None))
        print("Model loaded ✓")

        device = torch.device(device_str)
        #model.to(device)

        print(f"({time.time() - start_time} secs)")
        _report_dev_memory_stats()

        for s in ["A BUSINESS. ", "a business \n", ""]:
            enc = tokenizer(s, return_tensors="pt")
            print(s, "→ token IDs:", enc.input_ids.tolist())
        print(f"bos_token:{tokenizer.bos_token}, {tokenizer.bos_token_id}")
        print(f"eos_token id:{tokenizer.eos_token}, {tokenizer.eos_token_id}")

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
            # # Iterate records
            # lines_seen = 0
        total_lines = sum(len(n_best) for _, n_best in d_rec.items())
        from i6_experiments.users.zhang.datasets.utils import sort_dict_by_record, extract_record_id
        if self.use_prev_context:
            d_rec = sort_dict_by_record(d_rec)
        ctx_rec = None
        if self.context is not None and self.use_prev_context:
            ctx_rec = eval(cutil.uopen(self.context, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        batch_lines, batch_prompt = [], []
        eos_symbol = (
                " " + tokenizer.eos_token) if self.eos_symbol == "eos" else self.eos_symbol  # (" "+tokenizer.eos_token) makes ppl worse
        eos_symbol = eos_symbol if self.add_eos_to_completion else ""
        last_record = None

        for seq_tag, raw_line in d_rec.items():
            # if self.batch_size == 1: # and lines_seen % 40 == 0:
            #     print(f"Current summed scores for {len(scores_list)} seq: {torch.tensor(scores_list).sum()} total_log_prob {total_logprob}")
            #     scores_list = []
            boundary = False
            line = raw_line.strip().lower() if self.lower_case else raw_line.strip()
            if not line:
                continue
            total_word_tokens += len(line.split()) + (1 if self.add_eos_to_completion else 0)
            current_record = extract_record_id(seq_tag)
            batch_lines.append(line + eos_symbol)

            if self.use_prev_context and current_record != last_record:
                boundary = True # This will skip one time following prompt append
                self._clear_prompt() # makes self.prompt = '', double guard
                # Ensure there is no empty prompt inside a prompt batch
                # Process current batch, separately handle the boundary sequence
                if len(batch_lines)  == 1:
                    batch_count += 1
                    nll, tok_count = _score_batch(batch_lines, batch_prompt, tokenizer, model, device)
                    total_logprob += nll
                    total_tokens += tok_count
                else:
                    batch_count += 2
                    nll, tok_count = safe_score_batch(batch_lines[:-1], batch_prompt, tokenizer, model, device)
                    nll_boundary, tok_count_boundary = _score_batch([batch_lines[-1]], [], tokenizer, model, device)
                    total_logprob += nll + nll_boundary
                    total_tokens += tok_count + tok_count_boundary

                if batch_count % 1000 == 0:
                    pass
                    #print(f"  → Completed {batch_count:,} batches; Tokens so far: {total_tokens:,}")

                batch_lines, batch_prompt = [], []  # clear for next batch
                print(f"Clear context for record {last_record}")

            if self.prompt and not boundary: #Never append " "
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
                nll, tok_count = safe_score_batch(batch_lines, batch_prompt, tokenizer, model, device)
                # if self.batch_size > 1:
                #     print(f"current batching summed score {batch_count}*{self.batch_size}: {nll} total_log_prob {total_logprob}")
                # scores_list.append(nll)
                total_logprob += nll
                total_tokens += tok_count

                if batch_count % 1000 == 0:
                    pass
                    # print(f"  → Completed {batch_count:,} batches; Tokens so far: {total_tokens:,}")

                batch_lines, batch_prompt = [], []  # clear for next batch

            if self.use_prev_context:
                if ctx_rec is not None:
                    ctx = ctx_rec[seq_tag]
                    print(f"Ctx for {seq_tag}: {ctx}")
                else:
                    ctx = line
                    print(f"Transcription for {seq_tag}: {ctx}")
                self._update_prompt(ctx)
                print(f"Current context len: {len(self.prompt.split())}")
                last_record = current_record

        # Process any leftover lines (if total lines % batch_size != 0)
        if batch_lines:
            nll, tok_count = _score_batch(batch_lines, batch_prompt, tokenizer, model, device)
            total_logprob += nll
            total_tokens += tok_count

        print(f"nll:{total_logprob}")
        print(f"Subword ppl:{math.exp(-total_logprob / total_tokens)}")
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
            out_f.write(f"Total word tokens:: {total_word_tokens}\n")
            out_f.write(f"Total bpe tokens:: {total_tokens}\n")
            out_f.write(f"bpe level: {bpe_ppl}\n")
            out_f.write(f"Perplexity: {ppl}\n")

    @classmethod
    def hash(cls, parsed_args):
        """delete batch size from the hashing"""
        return super().hash({k: v for k, v in parsed_args.items() if k != "batch_size"})

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

      - compare_dim: dimension that need to compare (default  ctx limit)
      - value_source: "perplexity" (word-level) or "bpe" (BPE-level)
      - float_fmt: format string for numbers
    """

    def __init__(
        self,
        ppls_by_ds: Dict[str, Dict[str, Dict[Optional[int], tk.Path]]],
        compare_dim: Iterable[Optional[int]] = (0, 1000, 3000, 5000, None),
        dim_name: str = "CTX_limit",
        value_source: str = "perplexity",  # "perplexity" or "bpe"
        float_fmt: str = "{:.2f}",
        title: str = "LLM Perplexity Summary",
    ):
        self.ppls_by_ds = ppls_by_ds
        self.compare_dim = list(compare_dim)
        self.dim_name = dim_name
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
        self.out_avg_md = self.output_path("average.md")
        self.out_avg_csv = self.output_path(f"average.csv")

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
            fidx.write(f"In folder: {self.out_index_md.get_path()}")
            fidx.write("\n")

        from collections import defaultdict
        avg_values = defaultdict(lambda: defaultdict(float))
        # Per-dataset outputs
        for ds in sorted(self.ppls_by_ds.keys()):
            models = sorted(self.ppls_by_ds[ds].keys())
            values = {m: {} for m in models}
            for m in models:
                for ctx in self.compare_dim:
                    p = self.ppls_by_ds[ds][m].get(ctx)
                    values[m][ctx] = self._safe_load_ppl_from_txt(p) if p is not None else None
                    v = values[m][ctx]
                    if v is not None:
                        avg_values[m][ctx] += v

            headers = ["Model"] + [self._ctx_label(c) for c in self.compare_dim]
            # Markdown
            with uopen(self.out_per_ds_md[ds], "wt") as fmd:
                label = "Word-level Perplexity" if self.value_source == "perplexity" else "BPE-level Perplexity"
                fmd.write(f"# {ds} — {label} by {self.dim_name}\n\n")
                fmd.write("| " + " | ".join(headers) + " |\n")
                fmd.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                for m in models:
                    row = [m] + [self._fmt(values[m].get(c)) for c in self.compare_dim]
                    fmd.write("| " + " | ".join(row) + " |\n")

            # CSV
            with uopen(self.out_per_ds_csv[ds], "wt") as fcsv:
                fcsv.write(",".join(headers) + "\n")
                for m in models:
                    row = [m] + [self._fmt(values[m].get(c)) for c in self.compare_dim]
                    fcsv.write(",".join(row) + "\n")

        #AVG output
        num_dataset = len(self.ppls_by_ds.keys())
        if avg_values:
            headers = ["Model"] + [self._ctx_label(c) for c in self.compare_dim]
            # Markdown
            with uopen(self.out_avg_md, "wt") as fmd:
                label = "Word-level Perplexity" if self.value_source == "perplexity" else "BPE-level Perplexity"
                fmd.write(f"# {ds} — {label} by {self.dim_name}\n\n")
                fmd.write("| " + " | ".join(headers) + " |\n")
                fmd.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                for m in models:
                    row = [m] + [self._fmt(avg_values[m].get(c)/num_dataset) for c in self.compare_dim]
                    fmd.write("| " + " | ".join(row) + " |\n")

            # CSV
            with uopen(self.out_avg_csv, "wt") as fcsv:
                fcsv.write(",".join(headers) + "\n")
                for m in models:
                    row = [m] + [self._fmt(avg_values[m].get(c)/num_dataset) for c in self.compare_dim]
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
class DummyJob(Job):
    """
    Keep Sis running for debug
    """

    def __init__(
            self,
            *,
            version: int = 1,
    ):
        """
        :param model: modelwithcheckpoints, all fixed checkpoints + scoring file for potential other relevant checkpoints (see update())
        :param recog_and_score_func: epoch -> scores. called in graph proc
        """
        super(DummyJob, self).__init__()
        self.version = version
        self.output = self.output_var("run_time")

    def tasks(self) -> Iterator[Task]:
        """tasks"""
        yield Task("run", rqmt={"cpu": 1, "time": 2})  # mini_task=True)

    def run(self):
        """run"""
        import time
        start = time.time()
        while time.time() < start + 60 * 60:
            time.sleep(10)  # sleep for 10s to reduce CPU usage
        self.output.set(time.time() - start)

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

    from i6_experiments.users.zhang.datasets.utils import GetCorpusStatsJob
    from i6_experiments.users.zhang.utils.report import ReportDictJob
    #tk.register_output("/tools/debugDummy_time", DummyJob().output)

    from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
    LBS = False
    ES = True
    dev = False
    #ds_names = list(set(DEV_KEYS + TEST_KEYS))
    if dev:
        ds_names = list(set([key for key in DEV_KEYS if "callhome" not in key] + [key for key in TEST_KEYS if "mtp_eval-v2" in key])) if ES else []
        ds_names += [] if not LBS else [
                    "dev-clean",
                    "dev-other",
             ]
    else:
        ds_names = list(set([key for key in TEST_KEYS if "mtp_eval-v2" not in key])) if ES else []
        ds_names += [] if not LBS else [
                    "test-clean",
                    "test-other",
             ]
    check_models = [ "meta-llama/Llama-3.2-1B",
                      "meta-llama/Llama-3.1-8B",
                      "Qwen/Qwen3-1.7B-Base",
                      #"Qwen/Qwen3-0.6B-Base",
                      #"Qwen/Qwen3-8B-Base",
                     "Qwen/Qwen2-0.5B",
                     "Qwen/Qwen2-1.5B",
                     "Qwen/Qwen2-7B",
                     "microsoft/phi-4",
                         ]

    LLM_and_Batch_size = {"meta-llama/Llama-3.2-1B": 40,  # 40*6,
                          "meta-llama/Llama-3.1-8B": 50,
                          # 50*6, #14*9, # actual 21 batch size-> ~ 40GB peak usage#  be at least 3 times larger from 10*3
                          # "Qwen/Qwen3-0.6B-Base": 51,
                          "Qwen/Qwen3-1.7B-Base": 40,  # 40*6,#15 has peak 19GB on 48G, so can be at least doubled
                          "Qwen/Qwen2-0.5B":50,
                          "Qwen/Qwen2-1.5B":42,
                          "Qwen/Qwen2-7B":50,
                          # "Qwen/Qwen3-4B-Base":24,
                          "Qwen/Qwen3-8B-Base": 50,
                          "microsoft/phi-4": 42,
                          # 30*6, #Can be 24 on 80GB with 100 ctx(peak 40 with 14， peaK 48 with 30), so even 50*6 should be fine
                          # "mistralai/Mistral-7B-v0.3": 4,
                          }  # Keys of this determines which LLM will be built by lm_getter

    def get_corpus_text_dict_by_name(ds_name):
        from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import \
            get_corpus_text_dict as ES_get_corpus_text_dict
        from i6_experiments.users.zhang.datasets.librispeech import _get_corpus_text_dict
        get_corpus_text_dict = ES_get_corpus_text_dict if "ES" in ds_name else _get_corpus_text_dict
        return get_corpus_text_dict(key=ds_name)
    ppls = dict()
    ppls_by_ds = {}
    #ctx_order = [0, 100, 300, 500, 700, 1000, 2000, 3000, 5000, 'unlimited']
    ctx_order = [0, #64,
                 #128, 256, 512,
                 #1024, 2048, 4096,
                 ]
    batch_sizes = [1, 10, 40]
    fix_random_ctx_numbers = {"LBS": [1,3,10], "ES": [3,8,30]}
    FIX_CTX_TEST = False

    ppls_per_model = dict()
    seg_stats_per_ds = dict()
    rec_stats_per_ds = dict()
    for ds_name in ds_names:
        text_file = get_corpus_text_dict_by_name(ds_name)
        get_statsjob = GetCorpusStatsJob(text_file=text_file)
        seg_stats_per_ds[ds_name] = get_statsjob.out_seg_report
        rec_stats_per_ds[ds_name] = get_statsjob.out_rec_report
        for model_id in check_models:
            name = os.path.basename(model_id)
            model = DownloadHuggingFaceRepoJob(model_id=model_id)
            tk.register_output(model_id, model.out_hub_cache_dir)
            ppls[name] = dict()
            #for ctx_len_limit in [None]:#ctx_order:#, 5000, None]:
            #for fix_random_ctx_number in fix_random_ctx_numbers:
            range = ctx_order if not FIX_CTX_TEST else fix_random_ctx_numbers["ES" if "ES" in ds_name else "LBS"]
            for dim_name in range:
                prompt = None
                if not FIX_CTX_TEST:
                    if dim_name == 'unlimited':
                        dim_value = None
                    else:
                        dim_value = dim_name
                    batch_size = 30 if '8B' in name else 50
                    if "ES" in ds_name and '8B' in name:
                        batch_size = 20
                    if dim_value is None or dim_value > 4000:
                        batch_size = 1
                    elif dim_value > 2000:
                        batch_size = min(5, batch_size)
                    elif dim_value > 500:
                        batch_size = min(10, batch_size)
                    elif dim_value > 250:
                        batch_size = min(20, batch_size)
                    if "common_voice" in ds_name:
                        dim_value = 0
                        batch_size = 40
                else:
                    dim_value = dim_name
                    ctx_len_limit = 0
                    batch_size = LLM_Batch_size_PPL[model_id]
                    batch_size = max(1, batch_size - dim_value//3)
                    if dim_value > 80:
                        batch_size = 3
                    prompt = get_fix_context_file(dim_name, task_name="ES" if "ES" in ds_name else "LBS")


                ppl_job = HuggingFaceLmPerplexityJobV2(
                    model_dir=model.out_hub_cache_dir,
                    text_file=[text_file], # get_test_corpus_text(keys=[ds_name])
                    llm_name=model_id,
                    batch_size=batch_size,#max(batch_size//2,1) + 2,
                    lower_case=USE_LOWER_CASE,
                    word_ppl=True,
                    prompt=prompt,
                    eos_symbol="\n",
                    use_prev_context=not FIX_CTX_TEST and (dim_value is None or dim_value > 0),
                    context_len_limit=dim_value if not FIX_CTX_TEST else 0,
                    add_eos_to_completion=True
                )
                ppl_job.rqmt.update(LLM_rqmt[model_id])
                ppl_job_name = (f"ppl/{'ES/' if 'ES' in ds_name else ''}{name}/{ds_name}"
                                f"{'low'}{f'fixed_{dim_name}' if FIX_CTX_TEST else ''}{f'_prev{str(dim_value)}' if dim_value else ('' if dim_value == 0 else '_prevInf')}"
                                + f"batch_{batch_size}")
                ppl_job.add_alias(ppl_job_name)
                ppls[name].update({ds_name: ppl_job.out_ppl})
                ppls_per_model.setdefault(name, {})[ds_name] = ppl_job.out_ppl
                # name_ext = ""
                # name_ext += f"prev_{ctx_len_limit}" if ctx_len_limit is not None else ""
                # name_ext += f"{'_low'}" + f"batch_{batch_size}"
                # tk.register_output(
                #     "ppl/" + name + f"/{task_name}-" + ds_name + name_ext + "-ppl",
                #     ppl_job.out_ppl)
                # Fill nested structure
                ppls_by_ds.setdefault(ds_name, {}).setdefault(name, {})[dim_name] = ppl_job.out_ppl

    task_spec_name = "LBS_ES" if ES and LBS else ("ES" if ES else "LBS")
    # Use this only with fixed ctx_limit
    for model_id in check_models:
        name = os.path.basename(model_id)
        tk.register_output(f"test/ppl/{task_spec_name}/{name}_ppl_ctx{ctx_order[-1]}_report", ReportDictJob(outputs=ppls_per_model[name]).out_report_dict)

    tk.register_output(f"stats/corpus/seg_stats_{len(ds_names)}_{task_spec_name}",
                       ReportDictJob(outputs=seg_stats_per_ds).out_report_dict)
    tk.register_output(f"stats/corpus/rec_stats_{len(ds_names)}_{task_spec_name}",
                       ReportDictJob(outputs=rec_stats_per_ds).out_report_dict)
    if FIX_CTX_TEST:
        if ES:
            summary_job_ES = SummarizeLLMPPLJob(
                ppls_by_ds=ppls_by_ds,
                compare_dim=fix_random_ctx_numbers["ES"],
                dim_name="Context_Length_Random",
                value_source="perplexity",  # word-level
                float_fmt="{:.3f}",
                title="LLM Perplexity Summary",
            )
            tk.register_output(f"ppl/ES_LLM_PPL_summary_random_ctx{min(fix_random_ctx_numbers['ES'])}_{max(fix_random_ctx_numbers['ES'])}",
                           summary_job_ES.out_index_md)
        if LBS:
            summary_job_LBS = SummarizeLLMPPLJob(
                ppls_by_ds=ppls_by_ds,
                compare_dim=fix_random_ctx_numbers["LBS"],
                dim_name="Context_Length_Random",
                value_source="perplexity",  # word-level
                float_fmt="{:.3f}",
                title="LLM Perplexity Summary",
            )
            tk.register_output(f"ppl/LBS_LLM_PPL_summary_random_ctx{min(fix_random_ctx_numbers['LBS'])}_{max(fix_random_ctx_numbers['LBS'])}",
                           summary_job_LBS.out_index_md)
    else:
        summary_job = SummarizeLLMPPLJob(
            ppls_by_ds=ppls_by_ds,
            compare_dim=ctx_order,
            dim_name="Context_Length",
            value_source="perplexity",  # word-level
            float_fmt="{:.3f}",
            title="LLM Perplexity Summary",
        )
        range = ctx_order[:-1] if None in ctx_order else ctx_order
        tk.register_output(f"ppl/{task_spec_name}_LLM_PPL_summary_ctx{min(range)}_{max(range)}",
                           summary_job.out_index_md)

    #summary_job.add_alias("ppl/LLM_PPL_summary_job")
    # if summary_job.dim_name == "Batch_size":
    #     tk.register_output(f"ppl/{task_spec_name}_LLM_PPL_summary_batch{min(batch_sizes)}_{max(batch_sizes)}{('_ctx' + str(ctx_len_limit)) if ctx_len_limit else ''}", summary_job.out_index_md)
    # elif summary_job.dim_name == "Context_Length":
    #         range = ctx_order[:-1] if None in ctx_order else ctx_order
    #         tk.register_output(f"ppl/{task_spec_name}_LLM_PPL_summary_ctx{min(range)}_{max(range)}", summary_job.out_index_md)
    # elif summary_job.dim_name == "Context_Length_Random":
    #     if ES:
    #         tk.register_output(f"ppl/ES_LLM_PPL_summary_random_ctx{min(fix_random_ctx_numbers['ES'])}_{max(fix_random_ctx_numbers['ES'])}",
    #                        summary_job_ES.out_index_md)
    #     if LBS:
    #         tk.register_output(f"ppl/LBS_LLM_PPL_summary_random_ctx{min(fix_random_ctx_numbers['LBS'])}_{max(fix_random_ctx_numbers['LBS'])}",
    #                        summary_job_LBS.out_index_md)
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
