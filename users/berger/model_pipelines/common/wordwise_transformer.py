"""Existing LibriSpeech Transformer LMs behind RASR's dynamic word-scoring API.

AM and LM vocabularies are independent. BPE words are retokenized using the LM's
codes; word-LM OOVs advance UNK while preserving the acoustic spelling. ONNX
exports return negative log probabilities for the next token, not logits.
"""

import ast
import json

import numpy as np
from sisyphus import Job, Task


def read_index_vocabulary(path, *, bpe=False):
    with open(path, encoding="utf-8") as src:
        text = src.read()
    if bpe:
        vocab = ast.literal_eval(text)
    else:
        vocab = {
            word: int(index) for word, index in (line.rsplit(None, 1) for line in text.splitlines() if line.strip())
        }
    if not isinstance(vocab, dict) or not vocab or any(not isinstance(i, int) or i < 0 for i in vocab.values()):
        raise ValueError("Invalid LM index vocabulary")
    if set(vocab.values()) != set(range(max(vocab.values()) + 1)):
        raise ValueError("LM output indices must be dense (sentence-boundary aliases may share index 0)")
    if vocab.get("</s>") != 0:
        raise ValueError("Expected training-time sentence boundary at index 0")
    return vocab


class WordwiseTransformerBackend:
    def __init__(self, config):
        import onnxruntime as ort

        self.kind = config["kind"]
        if self.kind not in {"bpe", "word"}:
            raise ValueError(f"Unsupported LM kind: {self.kind}")
        self.vocab = read_index_vocabulary(config["vocab"], bpe=self.kind == "bpe")
        unknown = "<unk>" if self.kind == "bpe" else "<UNK>"
        if unknown not in self.vocab:
            raise ValueError(f"LM vocabulary lacks {unknown!r}")
        self.unknown = self.vocab[unknown]
        self.batch_size = int(config["max-batch-size"] or 8)
        if self.batch_size < 1:
            raise ValueError("LM batch size must be positive")
        if self.kind == "bpe":
            from returnn.util.bpe import StandardBytePairEncoder

            self.encoder = StandardBytePairEncoder(config["bpe-codes"], labels=list(self.vocab))
        # if "CUDAExecutionProvider" not in ort.get_available_providers():
        #     raise RuntimeError("GPU LM experiments require ONNX Runtime CUDA support")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 2
        self.session = ort.InferenceSession(config["model"], sess_options=opts, providers=["CUDAExecutionProvider"])
        # if self.session.get_providers()[0] != "CUDAExecutionProvider":
        #     raise RuntimeError("LM silently fell back to CPU")
        self.checked = False

    def reset(self):
        pass  # RASR owns the per-segment score trie; no retained GPU KV states.

    def initial_tokens(self):
        return [0]

    def sentence_end_tokens(self):
        return [0]

    def tokenize(self, texts):
        result = []
        for text in texts:
            word = text.strip().upper()  # Both LibriSpeech LMs were trained on uppercase text.
            if not word or len(word.split()) != 1:
                raise ValueError(f"Expected one completed surface word, got {text!r}")
            pieces = self.encoder.segment_sentence(word) if self.kind == "bpe" else [word]
            result.append([self.vocab.get(piece, self.unknown) for piece in pieces])
        return result

    def score(self, requests):
        # Stateless export predicts only the last position. Expand a multi-token
        # continuation into teacher-forced prefixes; never sum logits or charge
        # the boundary twice. RASR deduplicates these token requests in its trie.
        rows, locations = [], []
        results = [[0.0] * len(r.tokens) for r in requests]
        for i, request in enumerate(requests):
            if not request.prefix:
                raise ValueError("LM request lacks its initial sentence-boundary token")
            for j, target in enumerate(request.tokens):
                rows.append(list(request.prefix) + list(request.tokens[:j]))
                locations.append((i, j, target))
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            lengths = np.array([len(row) for row in batch], dtype=np.int32)
            tokens = np.zeros((len(batch), int(lengths.max())), dtype=np.int32)
            for i, row in enumerate(batch):
                tokens[i, : len(row)] = row
            costs = self.session.run(["scores"], {"tokens": tokens, "tokens:size1": lengths})[0]
            if costs.shape != (len(batch), max(self.vocab.values()) + 1) or not np.isfinite(costs).all():
                raise ValueError("LM output shape/values disagree with its index vocabulary")
            if not self.checked:
                if not np.allclose(np.exp(-costs).sum(axis=-1), 1.0, atol=1e-3):
                    raise ValueError("LM export must return normalized negative log probabilities")
                self.checked = True
            for row, (i, j, target) in zip(costs, locations[start : start + len(batch)]):
                results[i][j] = float(row[target])
        return results


class RegisterWordwiseTransformer:
    """Pickle-safe registration hook, reconstructed in each RETURNN worker."""

    def __init__(self, gate_report):
        self.gate_report = gate_report

    def __call__(self):
        with open(self.gate_report.get_path()) as src:
            if not json.load(src)["passed"]:
                raise ValueError("Dynamic BPE topology gate failed")
        import librasr

        class Scorer(librasr.LlmScorer):
            def __init__(self, config):
                super().__init__(config)
                self.backend = WordwiseTransformerBackend(config)

            def reset(self):
                self.backend.reset()

            def initial_tokens(self):
                return self.backend.initial_tokens()

            def sentence_end_tokens(self):
                return self.backend.sentence_end_tokens()

            def tokenize(self, texts):
                return self.backend.tokenize(texts)

            def score(self, requests):
                return self.backend.score(requests)

        librasr.register_llm_scorer_type("wordwise-transformer", Scorer)
        from .dynamic_qwen_tuned import TunedQwenScorer
        from .rnnt_llm import register_rnnt_llm_scorer

        # Both acoustic BPE vocabularies are uppercase; preserve the best
        # lowercase/capitalized spelling in Qwen's own LM history.
        register_rnnt_llm_scorer(
            librasr,
            "wordwise-qwen",
            lambda config: TunedQwenScorer(
                config["model"],
                scale=float(config["outer-lm-scale"]),
                reward=float(config["token-reward"] or 0),
                formatting="original",
            ),
            case_variants=("lower", "capitalized"),
        )


class BpeWordwiseGateJob(Job):
    """Native CPU test of CTC/RNA repeats, immediate BPE boundaries and LM costs."""

    def __init__(self, binary, *, code_version):
        self.binary, self.code_version = binary, code_version
        self.out_report = self.output_path("validation.json")
        self.out_log = self.output_path("rasr.log")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=1))

    def run(self):
        # A subprocess isolates the global Flf lexicon singleton from Sisyphus.
        import subprocess
        import sys
        from pathlib import Path

        Path("pieces.txt").write_text("<blank>\nHE@@\nL@@\nO\nWORLD\n")
        script = r"""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
import librasr
class Toy(librasr.LlmScorer):
    def reset(self): pass
    def initial_tokens(self): return [0]
    def sentence_end_tokens(self): return [0]
    def tokenize(self, texts):
        return [[1] if t.strip() == "HELLO" else [2] if t.strip() == "WORLD" else (_ for _ in ()).throw(ValueError(t)) for t in texts]
    def score(self, requests): return [[{0:4., 1:2., 2:3.}[t] for t in r.tokens] for r in requests]
librasr.register_llm_scorer_type("toy", Toy)
cfg = librasr.Configuration()
for k, v in {
    "lib-rasr.lexicon.file":"vocab-text:pieces.txt",
    "lib-rasr.label-scorer.type":"no-op",
    "lib-rasr.search-algorithm.type":"llm-timesync-beam-search",
    "lib-rasr.search-algorithm.word-piece-convention":"continuation-marked",
    "lib-rasr.search-algorithm.word-piece-marker":"@@",
    "lib-rasr.search-algorithm.blank-label-index":"0",
    "lib-rasr.search-algorithm.max-beam-size":"16",
    "lib-rasr.search-algorithm.score-threshold":"12",
    "lib-rasr.search-algorithm.pre-llm-max-beam-size":"16",
    "lib-rasr.search-algorithm.llm-scale":"0.5",
    "lib-rasr.search-algorithm.llm.type":"toy",
    "*.log.channel":sys.argv[3], "*.error.channel":sys.argv[3], "*.statistics.channel":sys.argv[3],
    "lib-rasr.search-algorithm.log-stepwise-statistics":"false",
    "lib-rasr.search-algorithm.collapse-repeated-labels":sys.argv[2],
}.items(): cfg.set(k,v)
librasr.init_logging(cfg)
search = librasr.SearchAlgorithm(config=cfg)
path = [0,1,1,0,2,2,0,2,2,0,3,3,0,4,4,0] if sys.argv[2] == "true" else [0,1,2,2,3,4,0]
scores = np.full((len(path),5),10000,dtype=np.float32)
scores[np.arange(len(path)),path] = 0
trace = search.recognize_segment(scores)
pieces = [t.lemma for t in trace if t.lemma and t.lemma not in ("<blank>","[BLANK]")]
assert pieces == ["HE@@","L@@","L@@","O","WORLD"], pieces
cost = trace[-1].am_score + trace[-1].lm_score
assert abs(cost - 4.5) < 1e-5, cost
"""
        for collapse in ("true", "false"):
            subprocess.run(
                [sys.executable, "-c", script, self.binary.get_path(), collapse, self.out_log.get_path()], check=True
            )
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(dict(passed=True, ctc_and_rna=True, lm_cost=4.5), dst)
