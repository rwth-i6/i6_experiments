"""Small real-librasr checks, run on the pinned build before full recognition."""

import json
import os
import subprocess

from i6_core.lexicon.modification import WriteLexiconJob
from i6_core.lib.lexicon import Lemma, Lexicon
from sisyphus import Job, Task, tk


# Each case runs in its own process: never reuse another build's imported librasr
# or process-global logging channels. No acoustic model or GPU is required.
RUNNER = r"""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import librasr
config = librasr.Configuration()
assert config.set_from_file(sys.argv[2])
librasr.init_logging(config)
search = librasr.SearchAlgorithm(config=config)
case = json.loads(sys.argv[3])
# Repeat every nonblank frame, with separating blanks, to exercise CTC collapse.
path = [0]
for label in case["labels"]:
    path.extend([label, label, 0])
scores = np.full((len(path), case["vocab_size"]), 100.0, dtype=np.float32)
scores[np.arange(len(path)), path] = 0.0
trace = search.recognize_segment(scores)
assert trace, "Decoder returned no traceback"
score = float(trace[-1].am_score + trace[-1].lm_score)
assert abs(score - case["expected_score"]) < 1e-4, (score, case)
orths = [item.lemma for item in trace
         if item.lemma and not item.lemma.startswith(("[", "<"))]
if case["mode"] == "disabled":
    assert not any(orth.endswith(("&&", "@@")) for orth in orths), ("Closed tree used fallback", orths)
if case["kind"] == "bpe":
    text = " ".join(orths).replace("&&", "").replace("@@ ", "")
else:
    text = "".join(orth[:-2].replace("▁", " ") if orth.endswith("&&")
                   else " " + orth for orth in orths)
text = " ".join(text.upper().split())
assert text == case["expected_text"], (text, case, orths)
with open("result.json", "w") as out:
    json.dump(dict(case=case, score=score, text=text, traceback=orths), out, indent=2)
"""


def smoke_lexicon(kind):
    """Return a tiny inventory with the same special-lemma interface as real jobs."""
    lex = Lexicon()
    pieces = ["<blank>", "ka@@", "t", "x", "dog"] if kind == "bpe" else ["<blank>", "▁ka", "t", "x", "▁dog", "▁"]
    for label in pieces:
        lex.add_phoneme(label, variation="none")
    lex.add_lemma(Lemma(orth=["KAT"], phon=[f"{pieces[1]} t"], synt=["KAT"]))
    lex.add_lemma(Lemma(orth=["DOG"], phon=[pieces[4]], synt=["DOG"]))
    lex.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], synt=[], special="blank"))
    lex.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=[""], synt=["<UNK>"], special="unknown"))
    lex.add_lemma(Lemma(orth=["[BOS]"], phon=[""], synt=["<s>"], special="sentence-begin"))
    lex.add_lemma(Lemma(orth=["[EOS]"], phon=[""], synt=["</s>"], special="sentence-end"))
    for piece in pieces[1:]:
        if kind == "bpe":
            continuation = piece.endswith("@@")
            role = "unknown-continuation" if continuation else "unknown-final"
            orth = piece if continuation else piece + "&&"
            synt = [] if continuation else ["<UNK>"]
        else:
            role = "unknown-word-start" if piece.startswith("▁") else "unknown-word-internal"
            if piece == "▁":
                role += "+nonword"
            orth, synt = piece + "&&", []
        lex.add_lemma(Lemma(orth=[orth], phon=[piece], synt=synt, special=role))
    return lex, len(pieces)


class OovPrototypeDecoderSmokeJob(Job):
    """Check real decoding and fail before launching full-corpus tree jobs.

    Tests are finite-beam, forced-path examples, not an exhaustive correctness
    proof. Costs of alternative emissions are 100; expected paths cost <= 4.
    """

    def __init__(self, rasr_binary_path: tk.Path, python_exe: tk.Path, runner: str = RUNNER):
        from ..model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config
        from ..model_pipelines.common.recog_rasr_config import TreeTimesyncRecogParams, get_tree_timesync_recog_config

        self.rasr_binary_path, self.python_exe, self.runner = rasr_binary_path, python_exe, runner
        self.cases = []
        for kind in ("bpe", "granite"):
            lex, vocab_size = smoke_lexicon(kind)
            lexicon_file = WriteLexiconJob(lex).out_bliss_lexicon
            scenarios = [
                ("known-closed", "disabled", 0.0, [1, 2, 4], "KAT DOG", 0.0),
                ("known-exclusion", "known-excluding", -10.0, [1, 2, 4], "KAT DOG", 0.0),
                ("oov-penalty", "known-excluding", 2.0, [1, 3, 4], "KAX DOG", 2.0),
                ("oov-reward", "known-excluding", -2.0, [1, 3, 4], "KAX DOG", -2.0),
                ("two-oovs", "known-excluding", 2.0, [1, 3, 1, 3], "KAX KAX", 4.0),
            ]
            if kind == "granite":
                scenarios += [
                    ("separators", "known-excluding", 2.0, [5, 1, 3, 5, 5, 4, 5], "KAX DOG", 2.0),
                    ("separators-only", "known-excluding", 2.0, [5, 5], "", 0.0),
                ]
            for name, mode, beta, labels, text, score in scenarios:
                config = get_tree_timesync_recog_config(
                    lexicon_file,
                    [get_no_op_label_scorer_config()],
                    TreeTimesyncRecogParams(
                        collapse_repeated_labels=True,
                        max_beam_sizes=[1024],
                        score_thresholds=[30.0],
                        max_word_end_beam_size=1024,
                        word_end_score_threshold=30.0,
                        sentence_end_fallback=False,
                        recombination_mode="on",
                        unknown_word_fallback=mode,
                        unknown_word_tokenization="continuation-marked" if kind == "bpe" else "word-start-marked",
                        unknown_word_penalty=beta,
                    ),
                )
                self.cases.append(
                    dict(
                        name=f"{kind}-{name}",
                        kind=kind,
                        mode=mode,
                        config=config,
                        labels=labels,
                        vocab_size=vocab_size,
                        expected_text=text,
                        expected_score=score,
                    )
                )
        self.out_report = self.output_path("decoder-smoke.json")
        self.out_logs = self.output_path("logs", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        results = []
        for case in self.cases:
            directory = os.path.join(self.out_logs.get_path(), case["name"])
            os.makedirs(directory, exist_ok=True)
            config = case["config"].get_path()
            payload = {key: value for key, value in case.items() if key != "config"}
            with open(os.path.join(directory, "stdout.log"), "w") as out:
                subprocess.run(
                    [
                        self.python_exe.get_path(),
                        "-c",
                        self.runner,
                        self.rasr_binary_path.get_path(),
                        config,
                        json.dumps(payload),
                    ],
                    cwd=directory,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
            with open(os.path.join(directory, "result.json")) as src:
                results.append(json.load(src))
        with open(self.out_report.get_path(), "w") as out:
            json.dump(dict(passed=True, num_cases=len(results), results=results), out, indent=2)
