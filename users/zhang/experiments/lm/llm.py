from __future__ import annotations
import functools
from typing import Optional, Any, Dict, List, TYPE_CHECKING
from sisyphus import Job, Task, tk

from i6_experiments.common.datasets.librispeech.language_model import (
    get_librispeech_normalized_lm_data,
)
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    get_content_dir_from_hub_cache_dir,
)

from i6_experiments.users.zhang.datasets.librispeech import get_train_corpus_text, _get_test_corpus_text
if TYPE_CHECKING:
    pass
from i6_core.util import uopen


@functools.cache
def get_llm(model_id: str) -> tk.Path:
    model = DownloadHuggingFaceRepoJob(model_id=model_id)
    tk.register_output(model_id, model.out_hub_cache_dir)
    return model.out_hub_cache_dir


"""Compute perplexity for a HuggingFace Llama model on LibriSpeech."""
class HuggingFaceLmPerplexityJob(Job):
    """Compute perplexity of a HuggingFace LM over a text corpus."""

    def __init__(self, *, model_dir: tk.Path, text_file: List[tk.Path], batch_size: int = None, llm_name: str):
        super().__init__()
        self.model_dir = model_dir
        self.text_file = text_file
        self.batch_size = {"Llama-3.2-1B": 8, "Llama-3.1-8B": 3}.get(llm_name,1) if batch_size is None else batch_size
        self.out_ppl = self.output_path("ppl")
        self.rqmt = {"time": 4, "cpu": 3, "mem": 8 + self.batch_size//2, "gpu": 1, "gpu_mem": 36} #Just use gpu48gb
        self.rqmt.update({"mem": {"Llama-3.2-1B": 15, "Llama-3.1-8B": 50}.get(llm_name,12),
                        "time": {"Llama-3.2-1B": 4, "Llama-3.1-8B": 6}.get(llm_name,4)})

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import torch
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

        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        model.eval()
        device = torch.device(device_str)
        model.to(device)

        print(f"({time.time() - start_time} secs)")
        _report_dev_memory_stats()

        for s in ["A BUSINESS", "a business"]:
            enc = tokenizer(s, return_tensors="pt")
            print(s, "→ token IDs:", enc.input_ids.tolist())

        total_logprob = 0.0
        total_tokens = 0

        lines_seen = 0
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
                total_lines += sum(1 for _ in f)

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
                    line = raw_line.strip()
                    if not line:
                        continue
                    batch_lines.append(line)

                    lines_seen += 1

                    # Log after every `log_every` lines
                    if lines_seen % log_every == 0:
                        print(f"[Line {lines_seen:,}/{total_lines}] {100*lines_seen/total_lines:.2f}% processed…")
                        _report_dev_memory_stats()
                    # Once we have `batch_size` lines, tokenize & process them
                    if len(batch_lines) == self.batch_size:
                        # ➞ tokenize + move to device
                        batch_count += 1
                        encoding = tokenizer(
                            batch_lines,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=tokenizer.model_max_length,
                        )
                        input_ids = encoding["input_ids"].to(device)
                        attention_mask = encoding["attention_mask"].to(device)

                        # Mask out padding in the labels
                        labels = input_ids.clone()
                        labels[attention_mask == 0] = -100

                        with torch.no_grad():
                            out = model(input_ids, attention_mask=attention_mask, labels=labels)
                        # out.loss is average over non‐ignored tokens; multiply by token count
                        batch_tok_count = int(attention_mask.sum().item())
                        neg_log_likelihood = out.loss.item() * batch_tok_count

                        total_logprob -= neg_log_likelihood
                        total_tokens += batch_tok_count

                        # Optionally log every N batches:
                        if batch_count % 1000 == 0:
                            print(f"  → Completed {batch_count:,} batches; Tokens so far: {total_tokens:,}")

                        batch_lines = []  # clear for next batch



            # Process any leftover lines (if total lines % batch_size != 0)
            if batch_lines:
                encoding = tokenizer(
                    batch_lines,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                with torch.no_grad():
                    out = model(input_ids, attention_mask=attention_mask, labels=labels)
                batch_tok_count = int(attention_mask.sum().item())
                neg_log_likelihood = out.loss.item() * batch_tok_count

                total_logprob -= neg_log_likelihood
                total_tokens += batch_tok_count

        # Finally compute PPL
        ppl = math.exp(-total_logprob / total_tokens)
        with open(self.out_ppl.get_path(), "w") as out_f:
            out_f.write(f"Perplexity: {ppl}\n")


def py():
    lm_text = [#get_librispeech_normalized_lm_data(),
               get_train_corpus_text(),
                _get_test_corpus_text()]
    for llm_name in ["Llama-3.2-1B", "Llama-3.1-8B"]:
        dl_model = DownloadHuggingFaceRepoJob(model_id="meta-llama/" + llm_name)

        tk.register_output(f"llm/{llm_name}", dl_model.out_hub_cache_dir)

        ppl_job = HuggingFaceLmPerplexityJob(
            model_dir=dl_model.out_hub_cache_dir,
            text_file=lm_text,
            llm_name=llm_name,
        )
        ppl_job.add_alias("lm/" + llm_name + "/ppl/librispeech")
        tk.register_output(llm_name + "-librispeech-ppl", ppl_job.out_ppl)