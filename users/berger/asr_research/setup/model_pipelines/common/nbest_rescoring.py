"""Separate, dev-only n-best generation, full-string Qwen scoring and evaluation.

No references are visible to decoding, formatting or LM scoring. First-pass
costs are cumulative trace totals (already scaled), not sums of trace items.
Rescoring adds Qwen NLL to the first-pass objective; it does not replace an
existing 4gram or silently subtract OOV penalties. All LM tokens and EOS are
scored, without token-count averaging or a second normalization step.
"""

import json
import math
import time

from sisyphus import Job, Task

from .benchmark_scoring import read_jsonl
from .hf_scoring import normalize_compound_pair, normalize_hf_english


def lm_surface(text, policy):
    # Ordinary LibriSpeech lemmas are ALL-CAPS; fallback pieces are lowercase.
    # Remove that route-dependent casing artifact, not punctuation/contractions.
    text = text.lower()
    if policy == "lowercase":
        return text
    if policy not in ("initial-capital", "initial-capital-period"):
        raise ValueError(policy)
    # Deliberately only a first-letter diagnostic, not general truecasing.
    chars = list(text)
    for i, char in enumerate(chars):
        if char.isalpha():
            chars[i] = char.upper()
            break
    result = "".join(chars)
    if policy.endswith("-period") and result and result[-1] not in ".!?":
        result += "."
    return result


def select_candidate(candidates, n, scale, policy):
    if not candidates or n < 1 or scale < 0:
        raise ValueError("Invalid rescoring input")
    return min(
        candidates[:n],
        key=lambda c: c["first_pass_cost"] + scale * c["qwen"][policy]["nll"],
    )


class GraniteNbestJob(Job):
    def __init__(self, dataset, repo, rasr_config, binary, *, n=50, code_version):
        self.dataset, self.repo = dataset, repo
        self.rasr_config, self.binary, self.n, self.code_version = rasr_config, binary, n, code_version
        if n < 1:
            raise ValueError("n must be positive")
        self.out_candidates = self.output_path("candidates.jsonl")
        self.out_first_pass = self.output_path("first_pass.jsonl")
        self.out_report = self.output_path("generation.json")
        self.out_log = self.output_path("rasr.recog.log")
        self.rqmt = dict(cpu=2, mem=64, gpu=1, gpu_mem=24, time=24)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sys
        import numpy as np
        import torch
        import transformers
        from datasets import load_from_disk
        from .hf_benchmark import HfBenchmarkAdapter, audio_array, traceback_result

        if transformers.__version__ != "5.16.1":
            raise RuntimeError("Use the benchmark environment: transformers==5.16.1")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        adapter = HfBenchmarkAdapter(self.repo.get_path(), "ctc", "float32")
        sys.path.insert(0, self.binary.get_path())
        import librasr

        cfg = librasr.Configuration()
        if not cfg.set_from_file(self.rasr_config.get_path()):
            raise ValueError("Failed to load RASR config")
        cfg.set("lib-rasr.search-algorithm.log-stepwise-statistics", "false")
        for channel in ("log", "warning", "error", "statistics"):
            cfg.set(f"*.{channel}.channel", self.out_log.get_path())
        librasr.init_logging(cfg)
        search = librasr.SearchAlgorithm(config=cfg)
        if not hasattr(search, "recognize_segment_n_best"):
            raise RuntimeError("Pinned librasr lacks recognize_segment_n_best")
        dataset = load_from_disk(self.dataset.get_path())
        sizes, audio_seconds = [], 0.0
        started = time.perf_counter()
        with open(self.out_candidates.get_path(), "w") as dst, open(self.out_first_pass.get_path(), "w") as best:
            for i, sample in enumerate(dataset):
                waveform = audio_array(sample["audio"])
                with torch.inference_mode():
                    features = adapter.encode_inputs(adapter.inputs(waveform))
                features = np.ascontiguousarray(features, dtype=np.float32)
                traces = search.recognize_segment_n_best(features, self.n)
                # Native one-best is an independent parity check on the same scores.
                first = traceback_result(search.recognize_segment(features), adapter, {}, False)
                candidates = {}
                for rank, trace in enumerate(traces):
                    result = traceback_result(trace, adapter, {}, False)
                    am = float(trace[-1].am_score) if trace else 0.0
                    lm = float(trace[-1].lm_score) if trace else 0.0
                    cost = am + lm
                    if not math.isfinite(cost):
                        raise ValueError("Non-finite first-pass cost")
                    row = dict(result, native_rank=rank, am_cost=am, lm_cost=lm, first_pass_cost=cost)
                    old = candidates.get(row["text"])
                    if old is None or cost < old["first_pass_cost"]:
                        candidates[row["text"]] = row
                ordered = sorted(candidates.values(), key=lambda c: c["first_pass_cost"])
                if not ordered or ordered[0]["text"] != first["text"]:
                    raise ValueError(f"N-best/one-best mismatch for {sample['id']}; inspect lattice scores")
                first_trace = first["traceback"]
                first_cost = first_trace[-1]["am_score"] + first_trace[-1]["lm_score"] if first_trace else 0.0
                if abs(ordered[0]["first_pass_cost"] - first_cost) > 0.01:
                    raise ValueError(
                        f"N-best/one-best cost mismatch for {sample['id']}: "
                        f"{ordered[0]['first_pass_cost']} vs {first_cost}; "
                        "check epsilon/final weights and LM scaling in the native n-best exporter"
                    )
                sizes.append(len(ordered))
                audio_seconds += len(waveform) / 16000
                dst.write(json.dumps(dict(id=str(sample["id"]), candidates=ordered), ensure_ascii=False) + "\n")
                best.write(json.dumps(dict(id=str(sample["id"]), text=first["text"]), ensure_ascii=False) + "\n")
                dst.flush()
                print(f"{i + 1}/{len(dataset)} {sample['id']}: {len(ordered)} unique candidates", flush=True)
        with open(self.out_report.get_path(), "w") as f:
            json.dump(
                dict(
                    requested_n=self.n,
                    num_segments=len(sizes),
                    mean_unique=sum(sizes) / len(sizes),
                    min_unique=min(sizes),
                    max_unique=max(sizes),
                    singleton_fraction=sizes.count(1) / len(sizes),
                    rtf=(time.perf_counter() - started) / audio_seconds,
                ),
                f,
                indent=2,
            )


class ScoreQwenNbestJob(Job):
    def __init__(self, candidates, repo, *, code_version, batch_size=8):
        self.candidates, self.repo, self.code_version, self.batch_size = candidates, repo, code_version, batch_size
        self.policies = ("lowercase", "initial-capital", "initial-capital-period")
        self.out_scores = self.output_path("qwen_candidates.jsonl")
        self.out_report = self.output_path("lm_scoring.json")
        self.rqmt = dict(cpu=2, mem=16, gpu=1, gpu_mem=24, time=12)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(self.repo.get_path())
        model = AutoModelForCausalLM.from_pretrained(self.repo.get_path(), dtype=torch.float32).to("cuda").eval()
        bos, eos = model.config.bos_token_id, model.config.eos_token_id
        if not isinstance(bos, int) or not isinstance(eos, int):
            raise ValueError("Expected singleton BOS/EOS")
        rows = read_jsonl(self.candidates.get_path())
        surfaces = sorted({lm_surface(c["text"], p) for r in rows for c in r["candidates"] for p in self.policies})
        encoded = {}
        for surface in surfaces:
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if set(ids) & set(tokenizer.all_special_ids):
                raise ValueError("Special-token injection in candidate")
            if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != surface:
                raise ValueError(f"Non-round-tripping candidate: {surface!r}")
            encoded[surface] = [bos] + ids + [eos]
        surfaces.sort(key=lambda s: len(encoded[s]))
        scores = {}
        started = time.perf_counter()
        for start in range(0, len(surfaces), self.batch_size):
            batch = surfaces[start : start + self.batch_size]
            length = max(len(encoded[s]) for s in batch)
            if length > model.config.max_position_embeddings:
                raise ValueError("Candidate exceeds Qwen context; no silent truncation")
            ids = torch.full((len(batch), length), eos, dtype=torch.long, device="cuda")
            mask = torch.zeros_like(ids)
            for i, s in enumerate(batch):
                ids[i, : len(encoded[s])] = torch.tensor(encoded[s], device="cuda")
                mask[i, : len(encoded[s])] = 1
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1].float()
                costs = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
            for i, s in enumerate(batch):
                token_costs = costs[i, : len(encoded[s]) - 1].cpu().tolist()
                nll = sum(token_costs)
                if not math.isfinite(nll):
                    raise ValueError("Non-finite Qwen score")
                scores[s] = dict(surface=s, token_ids=encoded[s], token_costs=token_costs, nll=nll)
            print(f"Qwen scored {min(start + self.batch_size, len(surfaces))}/{len(surfaces)} strings", flush=True)
        with open(self.out_scores.get_path(), "w") as dst:
            for row in rows:
                for c in row["candidates"]:
                    c["qwen"] = {p: scores[lm_surface(c["text"], p)] for p in self.policies}
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(self.out_report.get_path(), "w") as f:
            json.dump(
                dict(
                    unique_strings=len(surfaces),
                    elapsed_seconds=time.perf_counter() - started,
                    dtype="float32",
                    bos=bos,
                    eos=eos,
                    policies=self.policies,
                ),
                f,
                indent=2,
            )


class EvaluateQwenNbestJob(Job):
    def __init__(self, references, candidates, *, code_version):
        self.references, self.candidates, self.code_version = references, candidates, code_version
        self.out_summary = self.output_path("summary.json")
        self.out_hypotheses = self.output_path("hypotheses", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import os
        from kaldialign import edit_distance

        refs = read_jsonl(self.references.get_path())
        rows = read_jsonl(self.candidates.get_path())
        reference = {r["id"]: r for r in refs}
        if len(reference) != len(refs) or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate IDs")
        if reference.keys() != {r["id"] for r in rows}:
            raise ValueError("Reference/candidate coverage mismatch")
        if any(r["group_id"] != r["id"] or r["chunk_index"] != 0 for r in refs):
            raise ValueError("Initial rescoring experiment requires independent utterances")

        def counts(row, candidate):
            ref = normalize_hf_english(reference[row["id"]]["reference"])
            hyp = normalize_hf_english(candidate["text"])
            ref, hyp = normalize_compound_pair(ref, hyp)
            return edit_distance(ref.split(), hyp.split())

        os.makedirs(self.out_hypotheses.get_path(), exist_ok=True)
        result = {}
        for n in (1, 10, 50):
            for policy in ("lowercase", "initial-capital", "initial-capital-period", "oracle"):
                for scale in (0.0,) if policy == "oracle" else (0.0, 0.05, 0.1, 0.2, 0.3):
                    name = f"n{n}_{policy}_scale{scale:g}"
                    totals = dict(ins=0, **{"del": 0}, sub=0, total=0, ref_len=0)
                    with open(os.path.join(self.out_hypotheses.get_path(), name + ".jsonl"), "w") as dst:
                        for row in rows:
                            if policy == "oracle":
                                candidate = min(row["candidates"][:n], key=lambda c: counts(row, c)["total"])
                            else:
                                candidate = select_candidate(row["candidates"], n, scale, policy)
                            errors = counts(row, candidate)
                            for key in totals:
                                totals[key] += errors[key]
                            dst.write(
                                json.dumps(
                                    dict(id=row["id"], text=candidate["text"], counts=errors), ensure_ascii=False
                                )
                                + "\n"
                            )
                    result[name] = dict(totals, wer_percent=100 * totals["total"] / totals["ref_len"])
        with open(self.out_summary.get_path(), "w") as f:
            json.dump(result, f, indent=2)
