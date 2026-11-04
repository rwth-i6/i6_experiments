"""Real-decoder acceptance checks for 101ed636, not a reuse of old job hashes."""

from copy import deepcopy
import json
import os
import subprocess

from i6_core.lexicon.modification import WriteLexiconJob
from i6_core.lib.lexicon import Lemma
from i6_core.rasr import WriteRasrConfigJob
from sisyphus import Job, Task

from .oov_decoder_smoke import OovPrototypeDecoderSmokeJob, RUNNER, smoke_lexicon

LENGTH_RUNNER = RUNNER.replace(
    "path.extend([label, label, 0])", "path.extend([label, 0] if case.get('rna') else [label, label, 0])"
)


class OovLengthDecoderSmokeJob(OovPrototypeDecoderSmokeJob):
    """Force known/unknown paths to test beta-by-token and alpha settlement.

    Wide-beam, zero-LM checks establish score equivalence for attested words,
    including entries whose syntactic token is UNK. They do not claim exhaustive
    equivalence for arbitrary acoustic scores or convergence at a narrow beam.
    """

    def __init__(self, rasr_binary_path, python_exe, runner=LENGTH_RUNNER):
        super().__init__(rasr_binary_path, python_exe, runner)
        original_cases = list(self.cases)

        def add(kind, name, labels, text, score, beta, alpha, mode="known-excluding", extended=False):
            template = next(c for c in original_cases if c["name"] == f"{kind}-known-exclusion")
            cfg = deepcopy(template["config"].creator.config)
            lex, size = smoke_lexicon(kind)
            if extended:
                pieces = list(lex.phonemes)
                lex.add_lemma(Lemma(orth=["KAX"], phon=[f"{pieces[1]} x"], synt=["<UNK>"]))
            if mode == "disabled":
                lex.lemmata = [l for l in lex.lemmata if not (l.special or "").startswith("unknown-")]
                next(l for l in lex.lemmata if l.special == "unknown").phon = []
            cfg.lib_rasr.lexicon.file = WriteLexiconJob(lex).out_bliss_lexicon
            search = cfg.lib_rasr.search_algorithm
            search.unknown_word_fallback = mode
            # Word-start mode is only legal with known-excluding in this revision.
            if mode == "disabled":
                search.unknown_word_tokenization = "continuation-marked"
            search.unknown_word_penalty = beta
            search.unknown_piece_penalty = alpha
            search.max_beam_size = 4096
            search.score_threshold = 200.0
            search.max_word_end_beam_size = 4096
            self.cases.append(
                dict(
                    name=f"{kind}-{name}",
                    kind=kind,
                    mode=mode,
                    config=WriteRasrConfigJob(cfg, template["config"].creator.post_config).out_config,
                    labels=labels,
                    vocab_size=size,
                    expected_text=text,
                    expected_score=score,
                )
            )

        for kind in ("bpe", "granite"):
            add(kind, "zero-cost-iv", [1, 2, 4] * 3, "KAT DOG KAT DOG KAT DOG", 0, 0, 0)
            add(kind, "alpha-known-refund", [1, 2, 4], "KAT DOG", 0, 0, 3)
            add(kind, "alpha-unknown", [1, 3, 4], "KAX DOG", 5, 2, 3)
            add(kind, "alpha-one-piece", [3], "X", 2, 2, 3)
            add(kind, "alpha-two-unknowns", [1, 3, 1, 3], "KAX KAX", 10, 2, 3)
            add(kind, "extended-unk-refund", [1, 3, 4], "KAX DOG", 2, 2, 3, extended=True)
            add(kind, "extended-unk-ordinary", [1, 3, 4], "KAX DOG", 2, 2, 0, mode="disabled", extended=True)
        add("bpe", "legacy-beta", [1, 3], "KAX", 2, 2, 0, mode="legacy")
        add("granite", "alpha-separators", [5, 1, 3, 5, 5, 4, 5], "KAX DOG", 5, 2, 3)
        add("granite", "known-refund-after-oov", [1, 3, 1, 2, 4], "KAX KAT DOG", 5, 2, 3)
        add("granite", "extended-refund-after-oov", [3, 1, 3, 4], "X KAX DOG", 4, 2, 3, extended=True)
        for name in ("bpe-zero-cost-iv", "bpe-alpha-known-refund", "bpe-alpha-unknown", "bpe-legacy-beta"):
            case = dict(next(c for c in self.cases if c["name"] == name))
            cfg = deepcopy(case["config"].creator.config)
            cfg.lib_rasr.search_algorithm.tree_builder_type = "rna"
            cfg.lib_rasr.search_algorithm.collapse_repeated_labels = False
            cfg.lib_rasr.search_algorithm.force_blank_between_repeated_labels = False
            case.update(
                name=name.replace("bpe-", "rna-"),
                rna=True,
                config=WriteRasrConfigJob(cfg, case["config"].creator.post_config).out_config,
            )
            self.cases.append(case)


DELAYED_PENALTY_RUNNER = r"""
import sys,json,numpy as np
sys.path.insert(0,sys.argv[1])
import librasr
cfg=librasr.Configuration()
assert cfg.set_from_file(sys.argv[2])
librasr.init_logging(cfg)
search=librasr.SearchAlgorithm(config=cfg)
# DOG is a known final piece (column 4); ka@@ can remain pending (column 1).
scores=np.full((15,5),100.,dtype=np.float32)
scores[::2,0]=0.
for i in range(1,12,2):
    scores[i,1]=0.
    scores[i,4]=5.
scores[13,3]=0.
scores[13,4]=5.
scores[13,0]=10.
trace=search.recognize_segment(scores)
orths=[t.lemma for t in trace if t.lemma and not t.lemma.startswith(('[','<'))]
text=' '.join(orths).replace('&&','').replace('@@ ','').upper()
cost=float(trace[-1].am_score+trace[-1].lm_score) if trace else None
with open('result.json','w') as dst:
    json.dump(dict(text=text, cost=cost, correct=(text=='DOG DOG DOG DOG DOG DOG DOG'
                                               and cost is not None and abs(cost-35)<1e-4)),dst)
"""


class OovDelayedPenaltyProbeJob(Job):
    """Keep the high-beta pruning counterexample separate from acceptance tests.

    This reports whether the new revision still has the old limitation. The
    narrow open condition is not required to fail forever; a future lookahead
    fix should change this diagnostic without breaking the acceptance suite.
    """

    def __init__(self, rasr_binary_path, python_exe, runner=DELAYED_PENALTY_RUNNER):
        self.rasr_binary_path, self.python_exe, self.runner = rasr_binary_path, python_exe, runner
        template = OovPrototypeDecoderSmokeJob(rasr_binary_path, python_exe).cases[0]["config"].creator
        self.configs = {}
        for name, mode, threshold, beam in (
            ("closed", "disabled", 16.0, 256),
            ("open-narrow", "known-excluding", 16.0, 256),
            ("open-large-beam", "known-excluding", 16.0, 4096),
            ("open-wide-threshold", "known-excluding", 200.0, 4096),
        ):
            cfg = deepcopy(template.config)
            search = cfg.lib_rasr.search_algorithm
            search.unknown_word_fallback = mode
            search.unknown_word_penalty = 100.0
            search.unknown_piece_penalty = 0.0
            search.max_beam_size = beam
            search.max_word_end_beam_size = beam
            search.score_threshold = threshold
            self.configs[name] = WriteRasrConfigJob(cfg, template.post_config).out_config
        self.out_report = self.output_path("delayed-penalty-probe.json")
        self.out_logs = self.output_path("logs", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        results = {}
        for name, config in self.configs.items():
            directory = os.path.join(self.out_logs.get_path(), name)
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "stdout.log"), "w") as dst:
                subprocess.run(
                    [
                        self.python_exe.get_path(),
                        "-c",
                        self.runner,
                        self.rasr_binary_path.get_path(),
                        config.get_path(),
                    ],
                    cwd=directory,
                    stdout=dst,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
            with open(os.path.join(directory, "result.json")) as src:
                results[name] = json.load(src)
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(
                dict(results=results, delayed_beta_failure_observed=not results["open-narrow"]["correct"]),
                dst,
                indent=2,
            )
        if not results["closed"]["correct"] or not results["open-wide-threshold"]["correct"]:
            raise ValueError("Unexpected failure of delayed-penalty probe controls; inspect native logs")


FIRST_PIECE_RUNNER = RUNNER.replace(
    'scores = np.full((len(path), case["vocab_size"]), 100.0, dtype=np.float32)\n'
    "scores[np.arange(len(path)), path] = 0.0",
    "# Competing known DOG costs 20; fallback-only x costs 0 + beta=100.\n"
    'scores = np.full((3, case["vocab_size"]), 100.0, dtype=np.float32)\n'
    "scores[::2, 0] = 0.0\n"
    "scores[1, 3] = 0.0\n"
    "scores[1, 4] = 20.0",
)


class OovFirstPiecePruningProbeJob(OovPrototypeDecoderSmokeJob):
    """The first-piece counterexample, for both tokenizations and four budgets.

    All eight cases must return DOG at cost 20. On 3ed6d27b, the narrow open
    BPE case returns x at cost 100: its uncharged emission prunes DOG before
    the fallback exit adds beta. No GPU or AM forward pass is required.
    """

    def __init__(self, rasr_binary_path, python_exe, runner=FIRST_PIECE_RUNNER):
        super().__init__(rasr_binary_path, python_exe, runner=runner)
        templates = {case["kind"]: case for case in self.cases if case["name"].endswith("-known-closed")}
        self.cases = []
        for kind, template in templates.items():
            for name, mode, threshold, beam in (
                ("closed", "disabled", 16.0, 256),
                ("open-narrow", "known-excluding", 16.0, 256),
                ("open-large-beam", "known-excluding", 16.0, 4096),
                ("open-wide-threshold", "known-excluding", 200.0, 4096),
            ):
                cfg = deepcopy(template["config"].creator.config)
                search = cfg.lib_rasr.search_algorithm
                search.unknown_word_fallback = mode
                search.unknown_word_penalty = 100.0
                search.unknown_piece_penalty = 0.0
                search.max_beam_size = beam
                search.max_word_end_beam_size = beam
                search.score_threshold = threshold
                self.cases.append(
                    dict(
                        template,
                        name=f"{kind}-{name}",
                        mode=mode,
                        labels=[4],
                        expected_text="DOG",
                        expected_score=20.0,
                        config=WriteRasrConfigJob(cfg, template["config"].creator.post_config).out_config,
                    )
                )


class RequireOovPruningProbeJob(Job):
    """Turn the historical diagnostic into a hard gate, including narrow pruning.

    Separate from the original probe so its old job identity/results stay valid.
    A passing wide-threshold control alone is explicitly insufficient.
    """

    __sis_hash_exclude__ = {"first_piece_report": None}

    def __init__(self, probe_report, first_piece_report=None):
        self.probe_report = probe_report
        self.first_piece_report = first_piece_report
        self.out_report = self.output_path("pruning-acceptance.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.probe_report.get_path()) as src:
            report = json.load(src)
        expected = {"closed", "open-narrow", "open-large-beam", "open-wide-threshold"}
        results = report["results"]
        if set(results) != expected:
            raise ValueError(f"Incomplete pruning probe: {set(results)}")
        for name, result in results.items():
            if not result.get("correct") or result.get("cost") is None or abs(result["cost"] - 35.0) > 1e-4:
                raise ValueError(f"Pruning regression in {name}: {result}; do not launch fallback sweeps")
        if self.first_piece_report is not None:
            with open(self.first_piece_report.get_path()) as src:
                first_piece = json.load(src)
            expected_cases = {f"{kind}-{name}" for kind in ("bpe", "granite") for name in expected}
            cases = first_piece.get("results", [])
            if (
                not first_piece.get("passed")
                or first_piece.get("num_cases") != 8
                or len(cases) != 8
                or {case["case"]["name"] for case in cases} != expected_cases
            ):
                raise ValueError("Incomplete or failed first-piece pruning probe")
            for case in cases:
                if case.get("text") != "DOG" or case.get("score") is None or abs(case["score"] - 20.0) > 1e-4:
                    raise ValueError(f"First-piece pruning regression: {case}")
            report["first_piece"] = first_piece
        report["passed"] = True
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(report, dst, indent=2)
