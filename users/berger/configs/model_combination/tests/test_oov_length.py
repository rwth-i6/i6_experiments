"""OOV graph checks and CPU regressions; no installed model/GPU required."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sisyphus import tk
from i6_core.util import write_xml

from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_decoder_smoke import smoke_lexicon
from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_diagnostics import (
    audit_config,
    ctc_viterbi_alignment,
    read_lexicon,
    read_tracebacks,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_extended_lexicon import (
    ExtendGraniteOovLexiconJob,
    RankOovLexiconCandidatesJob,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import oov_length
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.oov_guard import (
    GuardedOovTracebackFormatter,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog_rasr_config import (
    TreeTimesyncRecogParams,
    get_tree_timesync_recog_config,
)


class OovLengthTest(unittest.TestCase):

    def test_fallback_lexicons_depend_on_pruning_acceptance(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_extended_lexicon import PrepareOovLengthLexiconJob

        preflight = oov_length.run_preflight()
        _, lexicons = oov_length.setup_granite_lexicons(preflight, approaches=("closed", "A-100k", "B", "C-100k"))
        for name in ("B", "C-100k"):
            gate = lexicons[name].creator.lexicon.creator
            self.assertIsInstance(gate, PrepareOovLengthLexiconJob)
            self.assertEqual(gate.pruning_report, preflight["pruning"].out_report)
        self.assertIsNone(lexicons["closed"].creator.lexicon.creator.pruning_report)
        self.assertIsNone(lexicons["A-100k"].creator.lexicon.creator.lexicon.creator.pruning_report)

    def test_pruning_acceptance_requires_all_four_cases(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_length_smoke import RequireOovPruningProbeJob

        with tempfile.TemporaryDirectory() as d:
            report, out = Path(d) / "probe.json", Path(d) / "accepted.json"
            results = {
                name: dict(correct=True, cost=35.0)
                for name in ("closed", "open-narrow", "open-large-beam", "open-wide-threshold")
            }
            report.write_text(json.dumps(dict(results=results)))
            job = RequireOovPruningProbeJob(tk.Path(str(report)))
            job.out_report = tk.Path(str(out))
            job.run()
            self.assertTrue(json.loads(out.read_text())["passed"])
            for name in results:
                broken = {k: dict(v) for k, v in results.items()}
                broken[name] = dict(correct=False, cost=None)
                report.write_text(json.dumps(dict(results=broken)))
                with self.assertRaisesRegex(ValueError, "Pruning regression"):
                    job.run()

    def test_combined_pruning_acceptance_rejects_first_piece_failure(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_length_smoke import RequireOovPruningProbeJob

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            budgets = ("closed", "open-narrow", "open-large-beam", "open-wide-threshold")
            (p / "delayed.json").write_text(
                json.dumps(
                    dict(
                        results={name: dict(correct=True, cost=35.0) for name in budgets},
                    )
                )
            )
            first = dict(
                passed=True,
                num_cases=8,
                results=[
                    dict(case=dict(name=f"{kind}-{name}"), score=20.0, text="DOG")
                    for kind in ("bpe", "granite")
                    for name in budgets
                ],
            )
            (p / "first.json").write_text(json.dumps(first))
            job = RequireOovPruningProbeJob(tk.Path(str(p / "delayed.json")), tk.Path(str(p / "first.json")))
            old_job = RequireOovPruningProbeJob(tk.Path(str(p / "delayed.json")))
            self.assertNotEqual(job.out_report.get_path(), old_job.out_report.get_path())
            job.out_report = tk.Path(str(p / "accepted.json"))
            job.run()
            self.assertTrue(json.loads((p / "accepted.json").read_text())["passed"])
            for index in (1, 5):  # Narrow open, both tokenizations; old probe still passes.
                broken = json.loads(json.dumps(first))
                broken["results"][index].update(score=100.0, text="X")
                (p / "first.json").write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError, "First-piece pruning regression"):
                    job.run()
            first["results"].pop()
            (p / "first.json").write_text(json.dumps(first))
            with self.assertRaisesRegex(ValueError, "Incomplete or failed"):
                job.run()

    def test_explicit_alpha_grid_validation(self):
        kwargs = dict(
            kind="bpe",
            lexicons={"known-excluding": tk.Path("lex")},
            binary=tk.Path("binary"),
            lm_conditions={"lm": None},
            phase="sweep",
        )
        for alphas in ((), (-1,), (float("nan"),), (float("inf"),)):
            with self.assertRaises(ValueError):
                oov_length.length_variants(**kwargs, alphas=alphas)
        with self.assertRaises(ValueError):
            oov_length.length_variants(**kwargs, alphas=(0, 1), alpha0_by_lm={"lm": 1})

    def test_alpha_is_opt_in_and_hashed(self):
        paths = [
            get_tree_timesync_recog_config(
                tk.Path("lex"),
                [],
                TreeTimesyncRecogParams(
                    collapse_repeated_labels=True, max_beam_sizes=[256], unknown_piece_penalty=alpha
                ),
            )
            for alpha in (None, 0.0, 0.5)
        ]
        self.assertNotIn("unknown-piece-penalty", str(paths[0].creator.config))
        self.assertEqual(paths[2].creator.config.lib_rasr.search_algorithm.unknown_piece_penalty, 0.5)
        self.assertEqual(len(set(p.get_path() for p in paths)), 3)

    def test_ctc_alignment_repeated_labels(self):
        scores = np.full((5, 3), -20.0)
        scores[np.arange(5), [0, 1, 0, 1, 0]] = 0
        self.assertEqual(ctc_viterbi_alignment(scores, [1, 1]).tolist(), [0, 1, 2, 3, 4])
        with self.assertRaises(ValueError):
            ctc_viterbi_alignment(scores[:1], [1, 1])
        with self.assertRaises(ValueError):
            ctc_viterbi_alignment(scores, [0])

    def test_parser_apostrophes_and_no_alignment_mix(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log"
            path.write_text(
                "Recognize sequence 'dev/id/id'\n    Ground truth: \"IT'S\"\n"
                '    Recognized: "IT\'S"\n    Search traceback:\n'
                "<TracebackItem(lemma='IT'S', am_score=1, lm_score=2, start_time=0, end_time=1)>\n"
                "    Alignment traceback:\n"
                "<TracebackItem(lemma='WRONG', am_score=1, lm_score=2, start_time=0, end_time=1)>\n"
            )
            (seg,) = list(read_tracebacks(str(path)))
            self.assertEqual(seg["items"][0]["lemma"], "IT'S")
            self.assertEqual(len(seg["items"]), 1)

    def test_audit_dotted_scorer_and_costs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rasr.config"
            path.write_text(
                "[lib-rasr]\nlabel-scorer.type = no-op\n[lib-rasr.lm]\nscale=0.3\n"
                "[lib-rasr.search-algorithm]\nunknown-word-penalty=0\n"
            )
            report = audit_config(tk.Path(str(path)))
            self.assertEqual(report["word_exit_cost"], 0)
            self.assertEqual(report["lm_scale"], 0.3)
            path.write_text(path.read_text().replace("no-op", "transition"))
            with self.assertRaises(ValueError):
                audit_config(tk.Path(str(path)))

    def test_extension_candidates_and_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            lex, _ = smoke_lexicon("granite")
            lex.lemmata = [l for l in lex.lemmata if not (l.special or "").startswith("unknown-")]
            write_xml(str(p / "lex.xml"), lex.to_xml())
            (p / "train").write_text("KAT DOG KAX KAX X LMWORD\n")
            (p / "lm").write_text("\\data\\\nngram 1=3\n\\1-grams:\n-1 <UNK>\n-1 DOG\n-2 LMWORD\n\\end\\\n")
            rank = SimpleNamespace(
                training_text=tk.Path(str(p / "train")),
                recognition_lexicon=tk.Path(str(p / "lex.xml")),
                arpa_lm=tk.Path(str(p / "lm")),
                out_words=tk.Path(str(p / "words")),
                out_report=tk.Path(str(p / "rank.json")),
            )
            RankOovLexiconCandidatesJob.run(rank)
            self.assertEqual((p / "words").read_text(), "KAX\t2\nX\t1\n")
            table = [dict(token="▁ka", label="▁ka", role="start"), dict(token="x", label="x", role="internal")]
            (p / "table").write_text(json.dumps(table))
            extend = SimpleNamespace(
                lexicon=rank.recognition_lexicon,
                candidates=rank.out_words,
                tokenizer_table=tk.Path(str(p / "table")),
                repo=tk.Path("unused"),
                max_candidates=1,
                out_lexicon=tk.Path(str(p / "extended.xml")),
                out_report=tk.Path(str(p / "extend.json")),
                out_rejected=tk.Path(str(p / "rejected")),
            )
            tokenizer = SimpleNamespace(tokenize=lambda text: ["▁ka", "x"])
            with patch(
                "i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_extended_lexicon._load_tokenizer",
                return_value=tokenizer,
            ):
                ExtendGraniteOovLexiconJob.run(extend)
            root = read_lexicon(extend.out_lexicon)
            extra = root.findall("lemma")[-1]
            self.assertEqual(extra.findtext("orth"), "KAX")
            self.assertEqual(extra.findtext("synt/tok"), "<UNK>")
            self.assertIsNone(extra.get("special"))
            self.assertEqual(
                [x.findtext("symbol") for x in root.findall("phoneme-inventory/phoneme")], list(lex.phonemes)
            )

    def test_guard_detects_long_unknown_but_not_attested_word(self):
        with tempfile.TemporaryDirectory() as d:
            lex, _ = smoke_lexicon("bpe")
            path = Path(d) / "lex.xml"
            write_xml(str(path), lex.to_xml())
            guard = GuardedOovTracebackFormatter(tk.Path(str(path)), "continuation-marked", max_unknown_pieces=3)

            def item(orth):
                return SimpleNamespace(lemma=orth, start_time=0, end_time=1, confidence_score=None)

            with patch("builtins.print"):
                guard.validate_traceback([item("ka@@"), item("t&&")], "KAT", "known")
                with self.assertRaisesRegex(ValueError, "piece limit"):
                    guard.validate_traceback([item("ka@@")] * 4 + [item("x&&")], "KAX", "bad")
                with self.assertRaisesRegex(ValueError, "collapsed"):
                    guard.validate_traceback([item("x&&")], "WORD " * 12, "bad-boundaries")


if __name__ == "__main__":
    unittest.main()
