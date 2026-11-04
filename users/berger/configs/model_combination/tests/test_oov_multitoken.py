"""Graph and scoring-oracle checks for the separately pinned multi-token build."""

import unittest

from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_multitoken_smoke import multi_token_arpa


class OovMultiTokenTest(unittest.TestCase):

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
