"""Graph and scoring-oracle checks for the separately pinned multi-token build."""

import math
import unittest

from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import llm_fusion, oov_length
from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_multitoken_smoke import multi_token_arpa


class OovMultiTokenTest(unittest.TestCase):
    def test_build_and_pruning_dependencies(self):
        jobs = llm_fusion.run_preflight()
        old = oov_length.run_preflight()
        clone = jobs["binary"].creator.source_folder.creator
        self.assertEqual(clone.commit, "8c1ac793675fb7aba0198cc165dca7a2e624375d")
        self.assertIn("unknown-word-start+nonword", clone.patches[0])
        self.assertNotEqual(jobs["binary"], old["binary"])
        for key in ("correctness", "delayed_penalty_probe", "first_piece_probe"):
            self.assertEqual(jobs[key].rasr_binary_path, jobs["binary"])
        self.assertEqual(jobs["pruning"].first_piece_report, jobs["first_piece_probe"].out_report)

    def test_cases_cover_multitoken_scores_history_and_finalization(self):
        job = llm_fusion.run_preflight()["correctness"]
        self.assertEqual(len(job.cases), 46)
        new = job.cases[34:]
        self.assertEqual(len({c["name"] for c in new}), 12)
        for case in new:
            cfg = case["config"].creator.config
            self.assertFalse(cfg.lib_rasr.search_algorithm.lm_lookahead)
            lex = cfg.lib_rasr.lexicon.file.creator.static_lexicon
            self.assertEqual(next(l for l in lex.lemmata if l.orth == ["KAT"]).synt, ["KA", "T"])
            scale = cfg.lib_rasr.lm.scale
            self.assertIn(scale, (0.3, 1.0))
            forced = "fallback-" in case["name"]
            bias = 3 if forced else 0
            raw_cost = (
                2 * math.log(10) + 2 * math.log(2)
                if "fallback-final" in case["name"]
                else 4 * math.log(2) + (math.log(10) if forced else 0)
            )
            self.assertAlmostEqual(case["expected_score"], scale * raw_cost + bias)
            if forced:
                self.assertEqual(case["kind"], "granite")
                self.assertEqual(case["labels"][0], 3)

    def test_toy_bigram_is_normalized_and_history_sensitive(self):
        section = multi_token_arpa().split("\\2-grams:\n")[1].split("\\end\\")[0]
        probs = {}
        for line in section.splitlines():
            if line.strip():
                logp, h, t = line.split()
                probs[h, t] = 10 ** float(logp)
        self.assertEqual(len(probs), 36)
        for h in {h for h, _ in probs}:
            self.assertAlmostEqual(sum(p for (history, _), p in probs.items() if history == h), 1)
        self.assertAlmostEqual(probs["T", "DOG"], 0.5)
        self.assertAlmostEqual(probs["KA", "DOG"], 0.1)
        self.assertAlmostEqual(probs["DOG", "</s>"], 0.5)
        self.assertAlmostEqual(probs["T", "</s>"], 0.1)
