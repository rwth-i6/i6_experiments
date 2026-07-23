import gzip
import json
import math
import time
from pathlib import Path

from sisyphus import Job, Task


def extract_sample_records(data):
    texts = []
    entropies = []
    for seq_tag in sorted(data):
        beams = data[seq_tag]
        if len(beams) != 1:
            raise ValueError(f"expected one sample for {seq_tag}, got {len(beams)}")
        entropy, text = beams[0]
        texts.append(text)
        entropies.append(float(entropy))
    return texts, entropies


def perplexity_from_totals(nll_sum, num_tokens):
    if num_tokens <= 0:
        raise ValueError("no tokens to score")
    return math.exp(nll_sum / num_tokens)


def _format_time(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _format_progress(*, step, ppl, step_duration, elapsed, complete):
    remaining = elapsed / complete - elapsed
    return (
        f"gen ppl step {step}, ppl {ppl:.3f}, {step_duration:.3f} sec/step, "
        f"elapsed {_format_time(elapsed)}, exp. remaining {_format_time(remaining)}, "
        f"complete {complete * 100:.2f}%"
    )


def _load_search_output(filename):
    filename = str(filename)
    open_func = gzip.open if filename.endswith(".gz") else open
    with open_func(filename, "rt") as f:
        return eval(f.read(), {"nan": float("nan"), "inf": float("inf")})


class GenerativePerplexityJob(Job):
    def __init__(
        self,
        search_output,
        *,
        model_name="gpt2-large",
        batch_size=8,
        expected_num_samples=800,
        max_length=None,
        protocol=None,
    ):
        self.search_output = search_output
        self.model_name = model_name
        self.batch_size = batch_size
        self.expected_num_samples = expected_num_samples
        self.max_length = max_length
        self.protocol = protocol or {}
        self.out_results = self.output_path("result.json")
        self.out_ppl = self.output_var("ppl")
        self.out_entropy = self.output_var("entropy")
        self.out_num_samples = self.output_var("num_samples")
        self.out_num_tokens = self.output_var("num_tokens")

    def tasks(self):
        yield Task("run", rqmt={"time": 4, "mem": 24, "cpu": 4, "gpu": 1})

    def run(self):
        import torch
        import torch.nn.functional as f
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # load generated sampels
        data = _load_search_output(self.search_output.get_path())
        texts, entropies = extract_sample_records(data)
        if len(texts) != self.expected_num_samples:
            raise ValueError(f"expected {self.expected_num_samples} samples, got {len(texts)}")

        num_steps = math.ceil(len(texts) / self.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"gen ppl num_seqs: {len(texts)}, num_batches: {num_steps}", flush=True)
        print(f"Using device: {device}", flush=True)
        print(f"Load evaluation model: {self.model_name}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # load eval model
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.to(device).eval()
        max_length = self.max_length or tokenizer.model_max_length
        nll_sum = 0.0
        num_tokens = 0
        start_time = time.monotonic()

        with torch.inference_mode():
            for step, start in enumerate(range(0, len(texts), self.batch_size)):
                step_start_time = time.monotonic()
                batch = tokenizer(
                    texts[start : start + self.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                token_nll = f.cross_entropy(
                    logits[:, :-1].transpose(1, 2),
                    input_ids[:, 1:],
                    reduction="none",
                )
                valid = attention_mask[:, 1:].bool()
                nll_sum += token_nll.masked_select(valid).sum().item()
                num_tokens += valid.sum().item()
                step_end_time = time.monotonic()
                print(
                    _format_progress(
                        step=step,
                        ppl=perplexity_from_totals(nll_sum, num_tokens),
                        step_duration=step_end_time - step_start_time,
                        elapsed=step_end_time - start_time,
                        complete=min(start + self.batch_size, len(texts)) / len(texts),
                    ),
                    flush=True,
                )

        print(
            f"Generative perplexity {num_steps} steps, {_format_time(time.monotonic() - start_time)} elapsed",
            flush=True,
        )
        ppl = perplexity_from_totals(nll_sum, num_tokens)
        entropy = sum(entropies) / len(entropies)
        print(f"ppl {ppl:.3f}, entropy {entropy:.3f}, num_tokens {num_tokens}", flush=True)
        result = {
            "ppl": ppl,
            "entropy": entropy,
            "num_samples": len(texts),
            "num_tokens": num_tokens,
            "nll_sum": nll_sum,
            "eval_model": self.model_name,
            "eval_batch_size": self.batch_size,
            "max_length": max_length,
            "sampling": self.protocol,
        }
        Path(self.out_results.get_path()).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        self.out_ppl.set(ppl)
        self.out_entropy.set(entropy)
        self.out_num_samples.set(len(texts))
        self.out_num_tokens.set(num_tokens)
