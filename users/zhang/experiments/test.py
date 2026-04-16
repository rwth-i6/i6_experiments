from __future__ import annotations

from typing import Tuple, Dict, Set, List, Optional, Union, Iterator, Any
from sisyphus import Job, Task, tk, gs

import os, io, gzip, json, ast
from contextlib import ExitStack
import torch
from i6_experiments.users.zhang.datasets.librispeech import (
    get_train_corpus_text,
    _get_test_corpus_text,
    _get_corpus_text_dict,
    get_test_corpus_text,
)

from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    get_content_dir_from_hub_cache_dir,
)

#This rqmt is only used by ppl job
LLM_rqmt = {"meta-llama/Llama-3.2-1B": {"time": 2, "cpu": 1, "mem": 16, "gpu": 1, "gpu_mem": 48},
            "meta-llama/Llama-3.1-8B": {"time": 4, "cpu": 3, "mem": 40, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-0.6B-Base": {"time": 1, "cpu": 1, "mem": 12, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-1.7B-Base": {"time": 2, "cpu": 1, "mem": 20, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-4B-Base":{"time": 1, "cpu": 1, "mem": 25, "gpu": 1, "gpu_mem": 48},
            "Qwen/Qwen3-8B-Base":{"time": 4, "cpu": 3, "mem": 40, "gpu": 1, "gpu_mem": 48},
                  #"mistralai/Mistral-7B-v0.3": 4,
            "microsoft/phi-4":{"time": 4, "cpu": 3, "mem": 65, "gpu": 1, "gpu_mem": 80},
            }
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




class HuggingFaceLmPerplexityJobV2(Job):
    """Compute perplexity of a HuggingFace LM over a text corpus.
        Using a fixed context from training set
    """
    def __init__(self, *, model_dir: tk.Path, prompt: [List[str] | tk.Path] = None, text_file: List[tk.Path], batch_size: int = None,
                 llm_name: str, lower_case:bool = False, context_len_limit: int = None, eos_symbol: str = "", word_ppl: bool = False, add_eos_to_completion: bool = True, use_prev_context: bool = False, version:int = 8):
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
            if len(self.prompt_buffer) == 1: # Keep at least one sentence
                break
            self.prompt_buffer.pop(0)
        self.prompt = self.prompt_buffer.copy()
        self.prompt += [""]  # +[""] So that for last prompt(or only one prompt) it also has eos
        self.prompt = self.delimiter.join(self.prompt)

    def clear_prompt(self):
        torch.cuda.empty_cache()
        self.prompt_buffer = []
        self.prompt = ""

    def _score_batch(self, batch_lines, batch_prompt, tokenizer, model, device):
        #import pdb;pdb.set_trace()
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

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import math
        import time
        import returnn.util.basic as util # noqa

        def _score_batch(batch_lines, batch_prompt, tokenizer, model, device):
            """
            Use same scoring as rescoring job does
            """
            #return self._score_batch(batch_lines, batch_prompt, tokenizer, model, device)
            """
            Manual log-softmax + gather scorer that:
              - concatenates prompt+hyp per example (no interior PADs),
              - provide position id for each real token,
              - pads once across the batch,
              - indexes the correct time steps for each hypothesis token,
              - sums exact token log-probs and returns (nll, token_count, total_bpe_tokens).
            """
            # 0) Decide whether there is any *non-empty* prompt in this batch
            use_prompt = bool(batch_prompt) and any(p.strip() for p in batch_prompt)
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
                add_special_tokens=not use_prompt,  # if we have prompt, hyp should NOT add BOS/EOS
                max_length=tokenizer.model_max_length,
            )
            enc_prm = tokenizer(
                batch_prompt,
                return_tensors=None,
                padding=False,
                truncation=True,
                add_special_tokens=True,  # let tokenizer add BOS/EOS for the prompt context
                max_length=tokenizer.model_max_length,
            )

            # 2) Concatenate per example; record prompt/hyp lengths
            examples = []
            prm_lens, hyp_lens = [], []
            for prm_ids, hyp_ids in zip(enc_prm["input_ids"], enc_hyp["input_ids"]):
                # treat empty prompt as zero-length
                prm_len = len(prm_ids)
                hyp_len = len(hyp_ids)
                prm_lens.append(prm_len)
                hyp_lens.append(hyp_len)
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
                               ).logits.to(torch.float64)  # [B, T, V]

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
            tok_logp = torch.take_along_dim(log_probs, gather_ids, dim=-1).squeeze(-1)  # [B, maxH]

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

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        if dtype != torch.bfloat16:
            print(f"!!! Warning : Using {dtype} dtype!")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,  # load directly in low precision
            device_map={"": 0},  # put the whole model on cuda:0 (single GPU)
            low_cpu_mem_usage=True,  # stream weights, smaller CPU peak
            attn_implementation="flash_attention_2",#"flash_attention_2", sdpa,  # faster runtime; f
        )
        #model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        model.eval()
        print(getattr(model, "_attn_implementation", None) or getattr(model.config, "_attn_implementation", None))
        print("Model loaded ✓")
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
            # # Iterate records
            # lines_seen = 0
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
                self.clear_prompt() # makes self.prompt = '', double guard
                # Ensure there is no empty prompt inside a prompt batch
                # Process current batch, separately handle the boundary sequence
                if len(batch_lines)  == 1:
                    batch_count += 1
                    nll, tok_count = _score_batch(batch_lines, batch_prompt, tokenizer, model, device)
                    total_logprob += nll
                    total_tokens += tok_count
                else:
                    batch_count += 2
                    nll, tok_count = _score_batch(batch_lines[:-1], batch_prompt, tokenizer, model, device)
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
                nll, tok_count = _score_batch(batch_lines, batch_prompt, tokenizer, model, device)
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
                self.update_prompt(ctx)
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


from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from i6_experiments.users.zhang.recog import clean_RecogOut
from i6_core.returnn.search import (
    ReturnnSearchJobV2,
    SearchRemoveLabelJob,
    SearchCollapseRepeatedLabelsJob,
    SearchTakeBestJob,
)
def _spm_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_core.returnn.search import SearchOutputRawReplaceJob

    words = SearchOutputRawReplaceJob(bpe.output, [(" ", ""), ("▁", " ")], output_gzip=True).out_search_results
    return RecogOutput(output=words)


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
    from i6_experiments.users.zhang.experiments.apptek.am.ctc_spm10k_16khz_mbw import get_model_and_vocab, \
        NETWORK_CONFIG_KWARGS
    from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
    from i6_experiments.users.zhang.experiments.lm_getter import build_all_lms
    model, spm, i6_models = get_model_and_vocab(fine_tuned_model=True)

    # for k, v in spm["vocabulary"].items():
    #     print(f"{k}: {v}")
    # print(f"vocab setting: {spm}")
    vocab_config = SentencePieceModel(dim=spm["vocabulary"]["vocabulary_size"], model_file=spm["spm"])
    lms, ppl_results, _ = build_all_lms(vocab_config, lm_kinds={"word_ngram_apptek"}, word_ppl=True, only_best=True,
                                        task_name="ES", only_transcript=False)

    # from i6_experiments.users.zhang.datasets.utils import GetCorpusStatsJob
    # from i6_experiments.users.zhang.utils.report import ReportDictJob
    # # tk.register_output("/tools/debugDummy_time", DummyJob().output)
    #
    # from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
    # LBS = True
    # ES = False
    # # ds_names = list(set(DEV_KEYS + TEST_KEYS))
    # ds_names = list(set(DEV_KEYS)) if ES else []
    # ds_names += [] if not LBS else [
    #     "dev-clean",
    #     "dev-other",
    #     # "test-clean",
    #     # "test-other",
    # ]
    # check_models = [  #"meta-llama/Llama-3.2-1B",
    #     "meta-llama/Llama-3.1-8B",
    #     # "Qwen/Qwen3-1.7B-Base",
    #     #"Qwen/Qwen3-0.6B-Base",
    #     #"Qwen/Qwen3-8B-Base",
    #     #"microsoft/phi-4",
    # ]
    #
    # LLM_and_Batch_size = {"meta-llama/Llama-3.2-1B": 40,  # 40*6,
    #                       "meta-llama/Llama-3.1-8B": 50,
    #                       # 50*6, #14*9, # actual 21 batch size-> ~ 40GB peak usage#  be at least 3 times larger from 10*3
    #                       # "Qwen/Qwen3-0.6B-Base": 51,
    #                       "Qwen/Qwen3-1.7B-Base": 40,  # 40*6,#15 has peak 19GB on 48G, so can be at least doubled
    #                       # "Qwen/Qwen3-4B-Base":24,
    #                       "Qwen/Qwen3-8B-Base": 50,
    #                       "microsoft/phi-4": 42,
    #                       # 30*6, #Can be 24 on 80GB with 100 ctx(peak 40 with 14， peaK 48 with 30), so even 50*6 should be fine
    #                       # "mistralai/Mistral-7B-v0.3": 4,
    #                       }  # Keys of this determines which LLM will be built by lm_getter
    #
    # def get_corpus_text_dict_by_name(ds_name):
    #     from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import \
    #         get_corpus_text_dict as ES_get_corpus_text_dict
    #     from i6_experiments.users.zhang.datasets.librispeech import _get_corpus_text_dict
    #     get_corpus_text_dict = ES_get_corpus_text_dict if "ES" in ds_name else _get_corpus_text_dict
    #     return get_corpus_text_dict(key=ds_name)
    #
    # ppls = dict()
    # ppls_by_ds = {}
    # # ctx_order = [0, 100, 300, 500, 700, 1000, 2000, 3000, 5000, 'unlimited']
    # ctx_order = [0,64]#[0, 64, 128, 256, 512, 1024, 2048, 4096]
    # batch_sizes = [1, 10, 40]
    # ppls_per_model = dict()
    # seg_stats_per_ds = dict()
    # rec_stats_per_ds = dict()
    # for ds_name in ds_names:
    #     text_file = get_corpus_text_dict_by_name(ds_name)
    #     get_statsjob = GetCorpusStatsJob(text_file=text_file)
    #     seg_stats_per_ds[ds_name] = get_statsjob.out_seg_report
    #     rec_stats_per_ds[ds_name] = get_statsjob.out_rec_report
    #     for model_id in check_models:
    #         name = os.path.basename(model_id)
    #         model = DownloadHuggingFaceRepoJob(model_id=model_id)
    #         tk.register_output(model_id, model.out_hub_cache_dir)
    #         ppls[name] = dict()
    #         # for ctx_len_limit in [None]:#ctx_order:#, 5000, None]:
    #         for ctx_len_limit_name in ctx_order:
    #             if ctx_len_limit_name == 'unlimited':
    #                 ctx_len_limit = None
    #             else:
    #                 ctx_len_limit = ctx_len_limit_name
    #             batch_size = 30 if '8B' in name else 50
    #             if "ES" in ds_name and '8B' in name:
    #                 batch_size = 20
    #             if ctx_len_limit is None or ctx_len_limit > 4000:
    #                 batch_size = 1
    #             elif ctx_len_limit > 2000:
    #                 batch_size = min(5, batch_size)
    #             elif ctx_len_limit > 500:
    #                 batch_size = min(10, batch_size)
    #             elif ctx_len_limit > 250:
    #                 batch_size = min(20, batch_size)
    #             if "common_voice" in ds_name:
    #                 ctx_len_limit = 0
    #                 batch_size = 40
    #             ppl_job = HuggingFaceLmPerplexityJobV2(
    #                 model_dir=model.out_hub_cache_dir,
    #                 text_file=[text_file],  # get_test_corpus_text(keys=[ds_name])
    #                 llm_name=model_id,
    #                 batch_size=30,  # max(batch_size//2,1) + 2,
    #                 lower_case=True,
    #                 word_ppl=True,
    #                 prompt=None,
    #                 eos_symbol="\n",
    #                 use_prev_context=True and (ctx_len_limit is None or ctx_len_limit > 0),
    #                 context_len_limit=ctx_len_limit,
    #                 add_eos_to_completion=True
    #             )
    #             ppl_job.rqmt.update(LLM_rqmt[model_id])
    #             ppl_job_name = (f"ppl/{'ES/' if 'ES' in ds_name else ''}{name}/{ds_name}" +
    #                             f"{'low'}{f'_prev{str(ctx_len_limit)}' if ctx_len_limit else ('' if ctx_len_limit == 0 else '_prevInf')}" + f"batch_{batch_size}")
    #             ppl_job.add_alias(ppl_job_name)
    #             ppls[name].update({ds_name: ppl_job.out_ppl})
    #             ppls_per_model.setdefault(name, {})[ds_name] = ppl_job.out_ppl
    #             name_ext = ""
    #             name_ext += f"prev_{ctx_len_limit}" if ctx_len_limit is not None else ""
    #             name_ext += f"{'_low'}" + f"batch_{batch_size}"
    #             # tk.register_output(
    #             #     "ppl/" + name + f"/{task_name}-" + ds_name + name_ext + "-ppl",
    #             #     ppl_job.out_ppl)
    #             # Fill nested structure
    #             ppls_by_ds.setdefault(ds_name, {}).setdefault(name, {})[ctx_len_limit_name] = ppl_job.out_ppl
    # # Use this only with fixed ctx_limit
    # for model_id in check_models:
    #     name = os.path.basename(model_id)
    #     tk.register_output(f"test/ppl/{name}_ppl_ctx{ctx_len_limit}_report",
    #                        ReportDictJob(outputs=ppls_per_model[name]).out_report_dict)
    #
    # task_spec_name = "LBS_ES" if ES and LBS else ("ES" if ES else "LBS")
    # tk.register_output(f"stats/corpus/seg_stats_{len(ds_names)}_{task_spec_name}",
    #                    ReportDictJob(outputs=seg_stats_per_ds).out_report_dict)
    # tk.register_output(f"stats/corpus/rec_stats_{len(ds_names)}_{task_spec_name}",
    #                    ReportDictJob(outputs=rec_stats_per_ds).out_report_dict)
    # summary_job = SummarizeLLMPPLJob(
    #     ppls_by_ds=ppls_by_ds,
    #     compare_dim=ctx_order,
    #     dim_name="Context_Length",
    #     value_source="perplexity",  # word-level
    #     float_fmt="{:.3f}",
    #     title="LLM Perplexity Summary",
    # )
    #
    # # summary_job.add_alias("ppl/LLM_PPL_summary_job")
    # if summary_job.dim_name == "Batch_size":
    #     tk.register_output(
    #         f"ppl/LLM_PPL_summary_batch{min(batch_sizes)}_{max(batch_sizes)}{('_ctx' + str(ctx_len_limit)) if ctx_len_limit else ''}",
    #         summary_job.out_index_md)
    # else:
    #     tk.register_output(f"ppl/LLM_PPL_summary_ctx{min(ctx_order[:-1])}_{max(ctx_order[:-1])}",
    #                        summary_job.out_index_md)


def llmppl_py():
    from i6_experiments.users.zhang.experiments.lm.llm import get_llm, LLM_Batch_size_PPL, HuggingFaceLmPerplexityJobV2, LLM_rqmt
    from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import \
        get_corpus_text_dict as ES_get_corpus_text_dict
    input_text_dict = "/nas/models/asr/hzhang/setups/2025-07-20--combined/work/i6_experiments/users/zhang/experiments/decoding/lm_rescoring/LmRescoringJob.1bdD1JpX7Xd0/output/output.py.gz"
    model_id = "microsoft/phi-4"
    llm_config, _ = get_llm(model_ids=[model_id],batch_sizes=[LLM_Batch_size_PPL[model_id]],word_ppl=True,task_name="ES")
    llm_config = llm_config["phi-4"]
    lm_rescore_res = _spm_to_words(clean_RecogOut(RecogOutput(output=tk.Path(input_text_dict)))).output
    ds_name = "test_set.ES.f8kHz.mtp_dev_heldout-v2.ref.ff_wer"
    lm_rescore_res = SearchTakeBestJob(lm_rescore_res).out_best_search_results

    ppl_job = HuggingFaceLmPerplexityJobV2(
        model_dir=llm_config["model_dir"],
        text_file=[tk.Path("/nas/models/asr/hzhang/setups/2025-07-20--combined/work/i6_core/corpus/convert/CorpusToTextDictJob.neCUj8m5VRif/output/text_dictionary.py.gz")],#[ES_get_corpus_text_dict(key=ds_name)],  # get_test_corpus_text(keys=[ds_name])
        batch_size=llm_config["batch_size"],
        lower_case=True,
        word_ppl=True,
        prompt=tk.Path("/nas/models/asr/hzhang/setups/2025-07-20--combined/work/i6_core/returnn/search/SearchTakeBestJob.jJiw86R2keDE/output/best_search_results.py.gz"),#lm_rescore_res,
        eos_symbol="\n",
        use_prev_context=True, # For now only check for this setting
        context_len_limit=llm_config["ctx_len_limit"],
        llm_name=model_id,
    )
    ppl_job.rqmt.update(LLM_rqmt[model_id])
    tk.register_output(f"test/phi_4_rescor_ppl", ppl_job.out_ppl)
