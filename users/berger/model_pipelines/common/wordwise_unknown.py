"""Per-completed-word UNK bias, separate from normalized LM probabilities.

Keep this opt-in wrapper separate so existing zero-bias recognition hashes
and the shared Transformer backend remain unchanged.
"""

import json
import math

from .wordwise_transformer import WordwiseTransformerBackend


class UnknownPenaltyWordwiseBackend(WordwiseTransformerBackend):
    def __init__(self, config):
        super().__init__(config)
        if self.kind != "word":
            raise ValueError("Word-level UNK penalty requires a word LM, not a BPE LM")
        penalty = float(config["unknown-word-penalty"])
        scale = float(config["outer-lm-scale"])
        if not math.isfinite(penalty) or penalty < 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Expected a finite nonnegative UNK penalty and positive LM scale")
        # RASR scales every returned cost. Undo that scale for this bias so
        # its effective contribution is beta, not lambda * beta.
        self.unscaled_unknown_penalty = penalty / scale
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("UNK sweep requires GPU LM inference")

    def score(self, requests):
        costs = super().score(requests)
        for request, row in zip(requests, costs):
            for i, target in enumerate(request.tokens):
                if target == self.unknown:
                    row[i] += self.unscaled_unknown_penalty
        return costs


class RegisterWordwiseUnknownPenalty:
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
                self.backend = UnknownPenaltyWordwiseBackend(config)

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

        librasr.register_llm_scorer_type("wordwise-transformer-unk-penalty", Scorer)
