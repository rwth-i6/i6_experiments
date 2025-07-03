from __future__ import annotations
import functools
from typing import Optional, Any, Dict, List, Sequence, TYPE_CHECKING
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
    get_test_corpus_text,
)

if TYPE_CHECKING:
    pass
from i6_core.util import uopen

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#@functools.cache
def get_llm(model_ids: List[str], prompt: Any, batch_sizes: List[int] = None, word_ppl: bool = False) -> Task:
    llms = dict()
    ppls = dict()
    if not batch_sizes:
        batch_sizes = [2 for _ in model_ids]
    if isinstance(prompt, Sequence):
        assert len(prompt) == len(model_ids)
    else:
        prompt = [prompt for _ in model_ids]
    assert len(model_ids) == len(batch_sizes)
    for model_id, prompt, batch_size in zip(model_ids, prompt, batch_sizes):
        model = DownloadHuggingFaceRepoJob(model_id="meta-llama/" + model_id)
        tk.register_output(model_id, model.out_hub_cache_dir)
        ppl_job = HuggingFaceLmPerplexityJob(
            model_dir=model.out_hub_cache_dir,
            text_file=[get_test_corpus_text(keys=["test-other"])],
            llm_name=model_id,
            batch_size=batch_size,
            word_ppl=word_ppl,
        )
        llms.update({model_id: {"model_dir": model.out_hub_cache_dir, "batch_size": batch_size,
            "name": model_id, "prompt": prompt, "lm_type" : "HuggingFaceLm"}})
        ppls.update({model_id: ppl_job.out_ppl})

    # ppl_job.add_alias("lm/" + llm_name + "/ppl/librispeech_" + ds_name + ("low" if lower_case else "") + eos_name)
    # tk.register_output(
    #     "ppl/" + llm_name + "/librispeech-" + ds_name + ("low" if lower_case else "") + eos_name + "-ppl",
    #     ppl_job.out_ppl)
    return llms, ppls

"""Compute perplexity for a HuggingFace Llama model on LibriSpeech."""
class HuggingFaceLmPerplexityJob(Job):
    """Compute perplexity of a HuggingFace LM over a text corpus."""

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
        self.rqmt.update({"mem": {"Llama-3.2-1B": 15, "Llama-3.1-8B": 40}.get(llm_name,12),
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
                total_lines += sum(1 for _ in f)
                total_word_tokens += sum(len(line.strip().split()) for line in f)

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
                        batch_count += 1
                        encoding = tokenizer(
                            batch_lines,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=tokenizer.model_max_length,
                        )
                        input_ids = encoding["input_ids"].to(device)
                        bpe_length += len(encoding["input_ids"][0].tolist())
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


def py():
    from i6_experiments.users.zhang.experiments.decoding.lm_rescoring import LmRescoringJob
    # lm_text = [#get_librispeech_normalized_lm_data(),
    #            get_train_corpus_text(),
    #             _get_test_corpus_text()]
    for llm_name in ["Llama-3.2-1B",
                     #"Llama-3.1-8B",
                     ]:
        dl_model = DownloadHuggingFaceRepoJob(model_id="meta-llama/" + llm_name)
        tk.register_output(f"llm/{llm_name}", dl_model.out_hub_cache_dir)
        # lower_case = False
        # llm_scale = 0.2
        # resco_job = HuggingFaceLmRescoringJob(
        #     model_dir=dl_model.out_hub_cache_dir,
        #     weight=llm_scale,
        #     recog_out_file=tk.Path("work/i6_core/returnn/search/SearchOutputRawReplaceJob.a5gb36CLt4N6/output/search_results.py.gz"),
        #     llm_name=llm_name,
        #     lower_case=lower_case,
        # )
        # resco_job.add_alias("lm/" + llm_name + "/rescoring_"  + f"w{llm_scale}".replace(".","") + ("low" if lower_case else ""))
        # tk.register_output(
        #     llm_name + "/librispeech-rescoring_test" + f"w{llm_scale}".replace(".","") + ("low" if lower_case else ""),
        #     resco_job.out_file)
        llm_scale = 0.2
        resco_job = LmRescoringJob(
            recog_out_file=tk.Path(
                "work/i6_core/returnn/search/SearchRemoveLabelJob.HxKTed4GQc38/output/search_results.py.gz"),
            lm_cfg={
                "lm_type": "HuggingFaceLm",
                "model_dir": dl_model.out_hub_cache_dir,
                "batch_size": 8,
                "weight": llm_scale,
            }
        )
        resco_job.rqmt["gpu_mem"] = 48
        resco_job.add_alias("lm/" + llm_name + "/rescoring_"  + f"w{llm_scale}".replace(".",""))
        tk.register_output(
            llm_name + "/librispeech-rescoring_test" + f"w{llm_scale}".replace(".",""),
            resco_job.out_file)
        for ds_name in [#"dev-clean", "dev-other",
                        #"test-clean",
                        "test-other",
                        #"train",
                        ]:
            for lower_case in [#True,
                               False,
                               ]:
                for eos_name, eos_symbol in {#"newline": " \n",
                                            "period":".",
                                             #"eos": "eos"
                                             }.items():
                    ppl_job = HuggingFaceLmPerplexityJob(
                        model_dir=dl_model.out_hub_cache_dir,
                        text_file=[get_test_corpus_text(keys = [ds_name])] if ds_name != "train" else [get_train_corpus_text()],
                        llm_name=llm_name,
                        lower_case=lower_case,
                        eos_symbol=eos_symbol,
                        batch_size=1,
                    )
                    ppl_job.add_alias("lm/" + llm_name + "/ppl/librispeech_" + ds_name + ("low" if lower_case else "") + eos_name)
                    tk.register_output("ppl/" + llm_name + "/librispeech-" + ds_name + ("low" if lower_case else "") + eos_name +  "-ppl", ppl_job.out_ppl)