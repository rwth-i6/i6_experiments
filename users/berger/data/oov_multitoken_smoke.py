"""Multi-token word exits: native score/history checks on the merged decoder."""

from copy import deepcopy
import math

from i6_core.lexicon.modification import WriteLexiconJob
from i6_core.rasr import RasrConfig, WriteRasrConfigJob
from sisyphus import Job, Task

from .oov_decoder_smoke import smoke_lexicon
from .oov_length_smoke import LENGTH_RUNNER, OovLengthDecoderSmokeJob


def multi_token_arpa():
    """Normalized toy bigram: preferred successor .5, each other successor .1.

    KA T represents KAT; DOG tests the history returned by that multi-token
    exit. EOS after T differs from EOS after DOG, also testing finalization.
    """
    tokens = ("<s>", "</s>", "<UNK>", "KA", "T", "DOG")
    preferred = {"<s>": "KA", "<UNK>": "KA", "KA": "T", "T": "DOG", "DOG": "</s>", "</s>": "</s>"}
    lines = ["\\data\\", "ngram 1=6", "ngram 2=36", "", "\\1-grams:"]
    lines += [f"{math.log10(1 / 6):.12f}\t{t}\t0" for t in tokens]
    lines += ["", "\\2-grams:"]
    lines += [f"{math.log10(0.5 if preferred[h] == t else 0.1):.12f}\t{h} {t}" for h in tokens for t in tokens]
    return "\n".join(lines + ["", "\\end\\", ""])


class WriteMultiTokenArpaJob(Job):
    def __init__(self, content):
        self.content = content
        self.out_arpa = self.output_path("toy.arpa")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.out_arpa.get_path(), "w") as f:
            f.write(self.content)


class OovMultiTokenDecoderSmokeJob(OovLengthDecoderSmokeJob):
    """Retain the 34 accounting checks and add 12 nonzero-LM sequence tests."""

    def __init__(self, rasr_binary_path, python_exe, runner=LENGTH_RUNNER):
        super().__init__(rasr_binary_path, python_exe, runner)
        arpa = WriteMultiTokenArpaJob(multi_token_arpa()).out_arpa
        for kind in ("bpe", "granite"):
            template = next(c for c in self.cases if c["name"] == f"{kind}-known-exclusion")["config"].creator
            scenarios = [
                ("closed-sequence", "disabled", [1, 2, 4], "KAT DOG", 4 * math.log(2), 0),
                ("open-sequence", "known-excluding", [1, 2, 4], "KAT DOG", 4 * math.log(2), 0),
            ]
            if kind == "granite":
                # Unmarked x forces the fallback region, which cannot return to
                # the ordinary tree. KAT must resolve there with both LM tokens.
                scenarios += [
                    (
                        "fallback-sequence",
                        "known-excluding",
                        [3, 1, 2, 4],
                        "X KAT DOG",
                        math.log(10) + 4 * math.log(2),
                        3,
                    ),
                    ("fallback-final", "known-excluding", [3, 1, 2], "X KAT", 2 * math.log(10) + 2 * math.log(2), 3),
                ]
            for name, mode, labels, text, lm_cost, bias in scenarios:
                for scale in (1.0, 0.3):
                    lex, size = smoke_lexicon(kind)
                    next(l for l in lex.lemmata if l.orth == ["KAT"]).synt = ["KA", "T"]
                    if mode == "disabled":
                        lex.lemmata = [l for l in lex.lemmata if not (l.special or "").startswith("unknown-")]
                        next(l for l in lex.lemmata if l.special == "unknown").phon = []
                    cfg = deepcopy(template.config)
                    cfg.lib_rasr.lexicon.file = WriteLexiconJob(lex).out_bliss_lexicon
                    cfg.lib_rasr.lm = RasrConfig()
                    cfg.lib_rasr.lm.type = "ARPA"
                    cfg.lib_rasr.lm.file = arpa
                    cfg.lib_rasr.lm.scale = scale
                    search = cfg.lib_rasr.search_algorithm
                    search.lm_lookahead = False
                    search.unknown_word_fallback = mode
                    if mode == "disabled":
                        search.unknown_word_tokenization = "continuation-marked"
                    search.unknown_word_penalty = 3.0
                    search.unknown_piece_penalty = 2.0 if mode == "known-excluding" else 0.0
                    search.max_beam_size = search.max_word_end_beam_size = 4096
                    search.score_threshold = search.word_end_score_threshold = 200.0
                    self.cases.append(
                        dict(
                            name=f"{kind}-{name}-lm{scale:g}",
                            kind=kind,
                            mode=mode,
                            config=WriteRasrConfigJob(cfg, template.post_config).out_config,
                            labels=labels,
                            vocab_size=size,
                            expected_text=text,
                            expected_score=scale * lm_cost + bias,
                        )
                    )
