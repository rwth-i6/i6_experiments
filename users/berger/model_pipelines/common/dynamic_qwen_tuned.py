"""First-pass transfer of rescoring biases, isolated from completed baselines."""

import json
import math
import time

from sisyphus import Job, Task

from .dynamic_llm import QwenContinuationScorer, casing_surfaces
from .dynamic_llm_jobs import DynamicGraniteQwenJob


def reward_costs(costs, tokens, *, reward, scale, excluded):
    """RASR scales these costs later: reward remains outside the LM scale."""
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(reward):
        raise ValueError("Token reward requires a finite positive LM scale")
    if len(costs) != len(tokens):
        raise ValueError("Token/cost length mismatch")
    return [cost - (reward / scale if token not in excluded else 0) for cost, token in zip(costs, tokens)]


class TunedQwenScorer(QwenContinuationScorer):
    def __init__(self, repo, *, scale, reward=0, formatting="cases", terminal_period=False):
        super().__init__(repo, device="cuda", batch_size=8)
        reward_costs([], [], reward=reward, scale=scale, excluded=set())
        if formatting not in {"cases", "initial-capital", "original"}:
            raise ValueError(formatting)
        self.scale, self.reward, self.formatting = scale, reward, formatting
        self.excluded = set(self.initial + self.end)
        if terminal_period:
            period = self.tokenizer.encode(".", add_special_tokens=False)
            if self.tokenizer.decode(period, clean_up_tokenization_spaces=False) != ".":
                raise ValueError("Non-round-tripping period")
            self.end = period + self.end
            self.excluded.update(period)

    def tokenize(self, texts):
        if self.formatting == "cases":
            return super().tokenize(texts)
        result = []
        for text in texts:
            # RASR prepends its space separator except for the first word.
            casing = "original" if self.formatting == "original" else "lower" if text[:1].isspace() else "capitalized"
            surface = casing_surfaces(text, (casing,))[0]
            ids = self.tokenizer.encode(surface, add_special_tokens=False)
            if not ids or set(ids) & set(self.tokenizer.all_special_ids):
                raise ValueError("Empty/special-token word")
            if self.tokenizer.decode(ids, clean_up_tokenization_spaces=False) != surface:
                raise ValueError("Non-round-tripping word")
            result.append([ids])
        return result

    def score_continuations(self, prefixes, continuations):
        costs = super().score_continuations(prefixes, continuations)
        return [
            reward_costs(row, tokens, reward=self.reward, scale=self.scale, excluded=self.excluded)
            for row, tokens in zip(costs, continuations)
        ]


def register_tuned_scorer(librasr, instances):
    class Scorer(librasr.LlmScorer):
        def __init__(self, config):
            super().__init__(config)
            self.backend = TunedQwenScorer(
                config["model"],
                scale=float(config["outer-lm-scale"]),
                reward=float(config["token-reward"] or 0),
                formatting=config["formatting"] or "cases",
                terminal_period=config["terminal-period"] == "true",
            )
            instances.append(self)

        def reset(self):
            self.backend.reset()

        def initial_tokens(self):
            return self.backend.initial_tokens()

        def sentence_end_tokens(self):
            return self.backend.sentence_end_tokens()

        def tokenize(self, texts):
            return self.backend.tokenize(texts)

        def score_continuations(self, prefixes, continuations):
            return self.backend.score_continuations(prefixes, continuations)

    librasr.register_llm_scorer_type("qwen-dynamic-tuned", Scorer)


class TunedDynamicQwenJob(DynamicGraniteQwenJob):
    """Separate job class preserves the completed original jobs and hashes."""

    def run(self):
        import sys
        import numpy as np
        import torch
        import transformers
        from datasets import load_from_disk
        from .hf_benchmark import HfBenchmarkAdapter, audio_array, traceback_result

        with open(self.gate.get_path()) as src:
            if not json.load(src)["passed"]:
                raise ValueError("Original native Qwen numerical gate did not pass")
        if transformers.__version__ != "5.16.1":
            raise RuntimeError("Use benchmark transformers==5.16.1")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        sys.path.insert(0, self.binary.get_path())
        import librasr

        instances = []
        register_tuned_scorer(librasr, instances)
        cfg = librasr.Configuration()
        if not cfg.set_from_file(self.rasr_config.get_path()):
            raise ValueError("Failed to read config")
        cfg.set("lib-rasr.search-algorithm.log-stepwise-statistics", "false")
        for channel in ("log", "warning", "error", "statistics"):
            cfg.set(f"*.{channel}.channel", self.out_log.get_path())
        librasr.init_logging(cfg)
        search = librasr.SearchAlgorithm(config=cfg)
        adapter = HfBenchmarkAdapter(self.repo.get_path(), "ctc", "float32")
        with open(self.vocabulary.get_path()) as src:
            vocabulary = {p.rstrip("\n"): i for i, p in enumerate(src)}
        dataset = load_from_disk(self.dataset.get_path())
        audio_seconds = elapsed = 0
        with open(self.out_hypotheses.get_path(), "w") as dst:
            for i, sample in enumerate(dataset):
                waveform = audio_array(sample["audio"])
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    features = adapter.encode_inputs(adapter.inputs(waveform))
                trace = search.recognize_segment(np.ascontiguousarray(features, dtype=np.float32))
                result = traceback_result(trace, adapter, vocabulary, True)
                pending = longest = 0
                for piece in adapter.tokenizer.convert_ids_to_tokens(result["token_ids"]):
                    if piece.startswith("▁"):
                        pending = 0
                    if piece != "▁":
                        pending += 1
                        longest = max(longest, pending)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                elapsed += seconds
                audio_seconds += len(waveform) / 16000
                backend = instances[-1].backend
                result.update(
                    id=str(sample["id"]),
                    elapsed_seconds=seconds,
                    num_output_words=len(result["text"].split()),
                    max_word_pieces=longest,
                    llm_requests=backend.requests,
                    llm_forward_batches=backend.forward_batches,
                    llm_max_request_length=backend.max_request_length,
                )
                dst.write(json.dumps(result, ensure_ascii=False) + "\n")
                dst.flush()
                print(f"{i + 1}/{len(dataset)}: RTF={elapsed / audio_seconds:.3f}", flush=True)
        with open(self.out_runtime.get_path(), "w") as dst:
            json.dump(
                dict(
                    num_segments=len(dataset),
                    audio_seconds=audio_seconds,
                    elapsed_seconds=elapsed,
                    rtf=elapsed / audio_seconds,
                    dtype="float32",
                    zero=False,
                ),
                dst,
                indent=2,
            )


class WordwiseQwenTokenizationJob(Job):
    """Check the approximation on finished hypotheses, without references."""

    def __init__(self, hypotheses, repo, *, code_version):
        self.hypotheses, self.repo, self.code_version = hypotheses, repo, code_version
        self.out_report = self.output_path("tokenization.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=6, time=1))

    def run(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.repo.get_path())
        result = {}
        for period in (False, True):
            total = mismatches = 0
            examples = []
            with open(self.hypotheses.get_path()) as src:
                for line in src:
                    row = json.loads(line)
                    words = row["text"].split()
                    pieces = [
                        casing_surfaces((" " if i else "") + word, ("lower" if i else "capitalized",))[0]
                        for i, word in enumerate(words)
                    ]
                    suffix = "." if period else ""
                    surface = "".join(pieces) + suffix
                    whole = tokenizer.encode(surface, add_special_tokens=False)
                    wordwise = [
                        token
                        for text in pieces + ([suffix] if suffix else [])
                        for token in tokenizer.encode(text, add_special_tokens=False)
                    ]
                    total += 1
                    if whole != wordwise:
                        mismatches += 1
                        if len(examples) < 10:
                            examples.append(dict(id=row["id"], surface=surface, whole=whole, wordwise=wordwise))
            result[f"period{int(period)}"] = dict(total=total, mismatches=mismatches, examples=examples)
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(result, dst, indent=2)
