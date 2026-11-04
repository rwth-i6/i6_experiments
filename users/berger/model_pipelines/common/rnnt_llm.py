"""Pinned standard RNN-T dynamic-word fusion jobs and topology gate."""

import json
import time

import numpy as np
import torch
from sisyphus import Job, Task


def register_rnnt_llm_scorer(librasr, name, backend_factory, *, case_variants=("lower", "capitalized")):
    """Adapt our single-variant backends to d000386c's request-based LLM API.

    RASR enumerates spelling variants and carries the cheapest one's history.
    Backends must preserve each supplied spelling (formatting="original").
    Older CTC binaries expect tokenize(texts) -> token variants and
    score_continuations(prefixes, tokens). This binary expects flat tokenizations
    and score(requests). Keep the compatibility boundary local to RNN-T jobs.
    """

    class Scorer(librasr.LlmScorer):
        def __init__(self, config):
            super().__init__(config)
            self.backend = backend_factory(config)

        def reset(self):
            self.backend.reset()

        def initial_tokens(self):
            return self.backend.initial_tokens()

        def sentence_end_tokens(self):
            return self.backend.sentence_end_tokens()

        def spelling_variants(self, words):
            from .dynamic_llm import casing_surfaces

            return [casing_surfaces(word, case_variants) for word in words]

        def tokenize(self, texts):
            variants = self.backend.tokenize(texts)
            if len(variants) != len(texts) or any(len(row) != 1 or not row[0] for row in variants):
                raise ValueError("RNN-T adapter requires exactly one nonempty tokenization per text")
            return [row[0] for row in variants]

        def score(self, requests):
            return self.backend.score_continuations([r.prefix for r in requests], [r.tokens for r in requests])

    librasr.register_llm_scorer_type(name, Scorer)
    return Scorer


class ZeroRnntLlmBackend:
    def reset(self):
        pass

    def initial_tokens(self):
        return [0]

    def sentence_end_tokens(self):
        return []

    def tokenize(self, texts):
        return [[[1]] for _ in texts]

    def score_continuations(self, prefixes, continuations):
        return [[0.0] * len(c) for c in continuations]


class RnntRawPiecesJob(Job):
    def __init__(self, repo, *, code_version):
        self.repo, self.code_version = repo, code_version
        self.out_vocab = self.output_path("pieces.txt")
        self.out_blank = self.output_var("blank")
        self.out_report = self.output_path("inventory.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=8, time=2))

    def run(self):
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(self.repo.get_path())
        tokenizer = AutoTokenizer.from_pretrained(self.repo.get_path())
        labels = tokenizer.convert_ids_to_tokens(list(range(config.vocab_size)))
        blank = config.blank_token_id
        labels[blank] = "[BLANK]"
        if len(set(labels)) != len(labels) or any(not p or "\n" in p or "\r" in p for p in labels):
            raise ValueError("RNN-T acoustic inventory must be dense and injective")
        for i, piece in enumerate(labels):
            if i == blank:
                continue
            if piece.count("▁") > 1 or ("▁" in piece and not piece.startswith("▁")) or any(c.isspace() for c in piece):
                raise ValueError(f"Unsupported word-piece boundary: {piece!r}")
        for text in ("hello world", "thank you", "commission's proposal", "European Union"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if blank in ids:
                raise ValueError("Tokenizer emits the acoustic blank")
            surface = "".join(labels[i] for i in ids).replace("▁", " ").strip()
            if tokenizer.decode(ids, group_tokens=False, skip_special_tokens=True).strip() != surface:
                raise ValueError("Raw pieces disagree with RNN-T detokenization")
        with open(self.out_vocab.get_path(), "w") as dst:
            dst.write("\n".join(labels) + "\n")
        self.out_blank.set(blank)
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(dict(num_labels=len(labels), blank=blank, marker="▁", model_output_order=True), dst)


class RnntLlmGateJob(Job):
    """CPU native gate: multiple labels per frame, repeats and blank-only time advancement."""

    def __init__(self, binary, *, code_version):
        self.binary, self.code_version = binary, code_version
        self.out_report = self.output_path("validation.json")
        self.out_log = self.output_path("rasr.log")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=1))

    def run(self):
        import sys

        sys.path.insert(0, self.binary.get_path())
        import librasr
        from .hf_benchmark import register_hf_scorer

        vocab = self.output_path("gate-pieces.txt").get_path()
        with open(vocab, "w") as dst:
            dst.write("▁he\nl\no\n▁world\n[BLANK]\n")

        class Adapter:
            family, blank = "rnnt", 4

            def score_contexts(self, contexts):
                answers = []
                for t, prefix in contexts:
                    if t >= 2:
                        answers.append(None)
                        continue
                    target = (
                        [0, 1, 1, 2][len(prefix)]
                        if t == 0 and len(prefix) < 4
                        else 3
                        if t == 1 and len(prefix) == 4
                        else 4
                    )
                    costs = [float("inf")] * 5
                    costs[target] = 0.0
                    answers.append((costs, t))
                return answers

        register_hf_scorer(librasr, Adapter())

        class Llm(ZeroRnntLlmBackend):
            def sentence_end_tokens(self):
                return [2]

            def tokenize(self, texts):
                tokens = {"hello": 1, "Hello": 3, "world": 4, "World": 5}
                return [[[tokens[text.strip()]]] for text in texts]

            def score_continuations(self, prefixes, continuations):
                rows = []
                for prefix, continuation in zip(prefixes, continuations):
                    history, costs = list(prefix), []
                    for token in continuation:
                        costs.append(
                            0.25 if token == 3 or (token == 4 and 3 in history) else 0.5 if token == 2 else 1.0
                        )
                        history.append(token)
                    rows.append(costs)
                return rows

        register_rnnt_llm_scorer(librasr, "rnnt-gate-llm", lambda config: Llm())
        cfg = librasr.Configuration()
        cfg.set("lib-rasr.lexicon.file", f"vocab-text:{vocab}")
        cfg.set("lib-rasr.label-scorer.type", "hf-benchmark")
        for key, value in {
            "type": "llm-rnnt-timesync-beam-search",
            "blank-label-index": 4,
            "max-beam-size": 1,
            "pre-llm-max-beam-size": 1,
            "score-threshold": 100,
            "max-labels-per-timeframe": 10,
            "collapse-repeated-labels": "false",
            "recombination-mode": "viterbi",
            "llm-scale": 0.3,
            "llm.type": "rnnt-gate-llm",
            "log-stepwise-statistics": "false",
        }.items():
            cfg.set(f"lib-rasr.search-algorithm.{key}", str(value))
        for channel in ("log", "warning", "error", "statistics"):
            cfg.set(f"*.{channel}.channel", self.out_log.get_path())
        librasr.init_logging(cfg)
        search = librasr.SearchAlgorithm(config=cfg)
        trace = search.recognize_segment(np.zeros((2, 1), dtype=np.float32))
        pieces = [x.lemma for x in trace if x.lemma and x.lemma != "[BLANK]"]
        if pieces != ["▁he", "l", "l", "o", "▁world"]:
            raise ValueError(f"RNN-T topology gate failed: {pieces!r}")
        cost = float(trace[-1].am_score + trace[-1].lm_score)
        if abs(cost - 0.3) > 1e-4:
            raise ValueError(f"Expected cheapest casing/history (.25 + .25 + .5) at scale .3; got {cost}")
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(
                dict(
                    passed=True,
                    multiple_labels_per_frame=True,
                    repeated_labels_preserved=True,
                    blank_advances_time=True,
                    word_and_eos_cost=cost,
                    casing_variant_history=True,
                ),
                dst,
            )


class RnntQwenRecogJob(Job):
    def __init__(
        self,
        dataset,
        repo,
        auxiliary_repo,
        vocabulary,
        binary,
        config,
        gate,
        *,
        kind,
        code_version,
        speech_baseline=None,
    ):
        if kind not in {"zero", "text", "speech"} or (kind == "speech" and speech_baseline is None):
            raise ValueError("Unknown fusion kind or missing standalone Speech-LLM baseline")
        self.dataset, self.repo, self.auxiliary_repo = dataset, repo, auxiliary_repo
        self.vocabulary, self.binary, self.config, self.gate = vocabulary, binary, config, gate
        self.kind, self.code_version, self.speech_baseline = kind, code_version, speech_baseline
        self.out_hypotheses = self.output_path("hypotheses.jsonl")
        self.out_runtime = self.output_path("runtime.json")
        self.out_validation = self.output_path("validation.json")
        self.out_log = self.output_path("rasr.log")
        self.rqmt = dict(cpu=2, mem=64, gpu=1, gpu_mem=24, time=48)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sys
        import transformers
        from datasets import load_from_disk
        from .hf_benchmark import HfBenchmarkAdapter, audio_array, traceback_result, register_hf_scorer
        from .dynamic_qwen_tuned import TunedQwenScorer
        from .qwen_speech import QwenSpeechScorer

        with open(self.gate.get_path()) as src:
            if not json.load(src)["passed"]:
                raise ValueError("RNN-T topology gate did not pass")
        if transformers.__version__ != "5.16.1":
            raise RuntimeError("Use benchmark transformers==5.16.1")
        if self.speech_baseline is not None:
            with open(self.speech_baseline.get_path()) as src:
                if not json.load(src)["ref_len"]:
                    raise ValueError("Standalone Speech-LLM baseline must be scored first")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        sys.path.insert(0, self.binary.get_path())
        import librasr

        adapter = HfBenchmarkAdapter(self.repo.get_path(), "rnnt", "float32")
        register_hf_scorer(librasr, adapter)
        backend = None
        if self.kind == "zero":
            register_rnnt_llm_scorer(
                librasr, "qwen-dynamic-zero", lambda config: ZeroRnntLlmBackend(), case_variants=("original",)
            )
        elif self.kind == "text":
            register_rnnt_llm_scorer(
                librasr,
                "qwen-dynamic-tuned",
                lambda config: TunedQwenScorer(
                    config["model"],
                    scale=float(config["outer-lm-scale"]),
                    reward=float(config["token-reward"] or 0),
                    formatting=config["formatting"] or "original",
                    terminal_period=config["terminal-period"] == "true",
                ),
            )
        else:
            backend = QwenSpeechScorer(self.auxiliary_repo.get_path(), batch_size=8)
            register_rnnt_llm_scorer(librasr, "qwen-speech", lambda config: backend)
        cfg = librasr.Configuration()
        if not cfg.set_from_file(self.config.get_path()):
            raise ValueError("Failed to read RNN-T LLM config")
        for channel in ("log", "warning", "error", "statistics"):
            cfg.set(f"*.{channel}.channel", self.out_log.get_path())
        librasr.init_logging(cfg)
        search = librasr.SearchAlgorithm(config=cfg)
        with open(self.vocabulary.get_path()) as src:
            vocabulary = {p.rstrip("\n"): i for i, p in enumerate(src)}
        dataset = load_from_disk(self.dataset.get_path())
        elapsed = audio_seconds = 0
        validation = dict(passed=True, topology_gate=True, kind=self.kind)
        with open(self.out_hypotheses.get_path(), "w") as dst:
            for i, sample in enumerate(dataset):
                waveform = audio_array(sample["audio"])
                torch.cuda.synchronize()
                started = time.perf_counter()
                if backend is not None:
                    backend.set_audio(backend.prepare(waveform))
                    if i == 0:
                        validation["speech_score_gate"] = backend.validate()
                with torch.inference_mode():
                    features = adapter.encode_inputs(adapter.inputs(waveform))
                trace = search.recognize_segment(np.ascontiguousarray(features, dtype=np.float32))
                result = traceback_result(trace, adapter, vocabulary, True)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                elapsed += seconds
                audio_seconds += len(waveform) / 16000
                result.update(id=str(sample["id"]), elapsed_seconds=seconds)
                dst.write(json.dumps(result, ensure_ascii=False) + "\n")
                dst.flush()
                print(f"{i + 1}/{len(dataset)}: RTF={elapsed / audio_seconds:.3f}", flush=True)
        with open(self.out_validation.get_path(), "w") as dst:
            json.dump(validation, dst)
        with open(self.out_runtime.get_path(), "w") as dst:
            json.dump(
                dict(
                    audio_seconds=audio_seconds,
                    elapsed_seconds=elapsed,
                    rtf=elapsed / audio_seconds,
                    num_segments=len(dataset),
                    kind=self.kind,
                    dtype="float32",
                    batch_size=8,
                ),
                dst,
            )
