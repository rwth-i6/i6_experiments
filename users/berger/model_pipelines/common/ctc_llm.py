"""Granite CTC + (Speech-)Qwen fusion with the request-based LLM search of d000386c.

The 9804f5aa Granite jobs use the older tokenize/score_continuations LLM API. These
jobs reuse the RNN-T job with a CTC adapter: the label scorer is "no-op" and the
features are the (optionally prior-corrected) Granite CTC costs, as in the public runs.
"""

import json

import numpy as np
from sisyphus import Job, Task

from .rnnt_llm import RnntQwenRecogJob, ZeroRnntLlmBackend, register_rnnt_llm_scorer


class CtcLlmGateJob(Job):
    """CPU native gate: repeat collapsing, blank separation, multi-piece words and scaled word/EOS costs."""

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

        vocab = self.output_path("gate-pieces.txt").get_path()
        labels = ["[BLANK]", "▁he", "l", "o", "▁world"]
        with open(vocab, "w") as dst:
            dst.write("\n".join(labels) + "\n")
        # "l" twice needs a blank in between; the repeated "o" collapses.
        path = ["▁he", "l", "[BLANK]", "l", "o", "o", "▁world", "[BLANK]"]
        features = np.full((len(path), len(labels)), 10.0, dtype=np.float32)
        for t, label in enumerate(path):
            features[t, labels.index(label)] = 0.0

        class Llm(ZeroRnntLlmBackend):
            def sentence_end_tokens(self):
                return [2]

            def tokenize(self, texts):
                tokens = {"hello": 1, "world": 4}
                return [[[tokens[text.strip()]]] for text in texts]

            def score_continuations(self, prefixes, continuations):
                return [[0.5 if token == 2 else 1.0 for token in continuation] for continuation in continuations]

        register_rnnt_llm_scorer(librasr, "ctc-gate-llm", lambda config: Llm(), case_variants=("original",))
        cfg = librasr.Configuration()
        cfg.set("lib-rasr.lexicon.file", f"vocab-text:{vocab}")
        cfg.set("lib-rasr.label-scorer.type", "no-op")
        for key, value in {
            "type": "llm-timesync-beam-search",
            "blank-label-index": 0,
            "max-beam-size": 1,
            "pre-llm-max-beam-size": 1,
            "score-threshold": 100,
            "collapse-repeated-labels": "true",
            "recombination-mode": "on",
            "word-piece-convention": "word-start-marked",
            "word-piece-marker": "▁",
            "llm-scale": 0.3,
            "llm.type": "ctc-gate-llm",
            "log-stepwise-statistics": "false",
        }.items():
            cfg.set(f"lib-rasr.search-algorithm.{key}", str(value))
        for channel in ("log", "warning", "error", "statistics"):
            cfg.set(f"*.{channel}.channel", self.out_log.get_path())
        librasr.init_logging(cfg)
        search = librasr.SearchAlgorithm(config=cfg)
        trace = search.recognize_segment(features)
        pieces = [x.lemma for x in trace if x.lemma and x.lemma != "[BLANK]"]
        if pieces != ["▁he", "l", "l", "o", "▁world"]:
            raise ValueError(f"CTC topology gate failed: {pieces!r}")
        cost = float(trace[-1].am_score + trace[-1].lm_score)
        if abs(cost - 0.75) > 1e-4:
            raise ValueError(f"Expected zero AM cost and LLM cost (1 + 1 + .5) at scale .3; got {cost}")
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(
                dict(
                    passed=True,
                    repeats_collapsed=True,
                    blank_separates_repeats=True,
                    multi_piece_words=True,
                    word_and_eos_cost=cost,
                ),
                dst,
            )


class CtcQwenRecogJob(RnntQwenRecogJob):
    """Granite CTC with zero, text or speech LLM; Granite's own surface casing is kept."""

    __sis_hash_exclude__ = {"resume": False, "transformers_version": None}
    family = "ctc"
    speech_case_variants = ("original",)

    def __init__(self, *, prior=None, prior_scale=0.0, resume=False, transformers_version=None, **kwargs):
        self.transformers_version = transformers_version
        self.prior, self.prior_scale = prior, prior_scale
        super().__init__(resume=resume, **kwargs)
        self.out_correction_validation = self.output_path("correction_validation.json")

    def run(self):
        from .pretrained_correction import _save_validation, corrected_adapter

        import transformers
        if self.transformers_version is not None and transformers.__version__ != self.transformers_version:
            raise RuntimeError(f"API parity requires transformers=={self.transformers_version}, found {transformers.__version__}")
        with corrected_adapter(self.prior, self.prior_scale) as adapters:
            super().run()
        _save_validation(self.out_correction_validation, adapters)
        with open(self.out_correction_validation.get_path()) as src:
            validation = json.load(src)
        validation["transformers_version"] = transformers.__version__
        with open(self.out_correction_validation.get_path(), "w") as dst:
            json.dump(validation, dst)
