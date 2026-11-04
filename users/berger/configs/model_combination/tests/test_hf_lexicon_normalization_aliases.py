"""Exact token-path coverage for HF lexicon normalization aliases."""

import unittest

from i6_core.lib.lexicon import Lemma, Lexicon
from sisyphus import Job

from i6_experiments.example_setups.seq2seq_rasr_2025.data.huggingface import (
    HuggingFaceBpeBlissLexiconJob,
    _add_uncovered_normalization_aliases,
)


class NormalizationAliasTest(unittest.TestCase):
    @staticmethod
    def lexicon(*lemmas):
        lexicon = Lexicon()
        for lemma in lemmas:
            lexicon.add_lemma(lemma)
        return lexicon

    def test_expanded_contraction_uses_ordinary_words(self):
        do = Lemma(orth=["DO"], phon=["▁do", "▁Do", "▁DO"])
        not_ = Lemma(orth=["NOT"], phon=["▁not", "▁Not", "▁NOT"])
        dont = Lemma(orth=["DON'T"], phon=["▁don ' t"])
        lexicon = self.lexicon(do, not_, dont)
        self.assertEqual(
            _add_uncovered_normalization_aliases(lexicon, [(dont, ["▁do ▁not", "▁Do ▁Not", "▁DO ▁NOT"])]),
            (3, 0),
        )
        self.assertEqual(dont.phon, ["▁don ' t"])
        self.assertEqual(do.phon, ["▁do", "▁Do", "▁DO"])

    def test_possessive_and_spacing_aliases(self):
        possessive = Lemma(orth=["MURDERER'S"], phon=["▁murderer ' s"])
        oclock = Lemma(orth=["O'CLOCK"], phon=["▁o ' clock"])
        lexicon = self.lexicon(
            possessive,
            oclock,
            Lemma(orth=["MURDERER"], phon=["▁murderer"]),
            Lemma(orth=["IS"], phon=["▁is"]),
            Lemma(orth=["O"], phon=["▁o"]),
            Lemma(orth=["CLOCK"], phon=["▁clock"]),
        )
        self.assertEqual(
            _add_uncovered_normalization_aliases(lexicon, [(possessive, ["▁murderer ▁is"]), (oclock, ["▁o ▁clock"])]),
            (2, 0),
        )

    def test_missing_component_keeps_path(self):
        dont = Lemma(orth=["DON'T"], phon=["▁don ' t"])
        lexicon = self.lexicon(dont, Lemma(orth=["DO"], phon=["▁do"]))
        self.assertEqual(_add_uncovered_normalization_aliases(lexicon, [(dont, ["▁do ▁not"])]), (0, 1))
        self.assertIn("▁do ▁not", dont.phon)

    def test_different_tokenization_keeps_path(self):
        dont = Lemma(orth=["DON'T"], phon=["▁don ' t"])
        lexicon = self.lexicon(dont, Lemma(orth=["DO"], phon=["▁d o"]), Lemma(orth=["NOT"], phon=["▁not"]))
        self.assertEqual(_add_uncovered_normalization_aliases(lexicon, [(dont, ["▁do ▁not"])]), (0, 1))

    def test_numeric_and_punctuation_pronunciations_unchanged(self):
        twenty = Lemma(orth=["TWENTY"], phon=["▁twenty", "▁20"])
        punctuation = Lemma(orth=["."], phon=["_dot_"], synt=[])
        word = Lemma(orth=["WORD"], phon=["▁word"])
        lexicon = self.lexicon(twenty, punctuation, word)
        self.assertEqual(_add_uncovered_normalization_aliases(lexicon, [(word, ["▁word _dot_"])]), (1, 0))
        self.assertEqual(twenty.phon, ["▁twenty", "▁20"])
        self.assertEqual(punctuation.phon, ["_dot_"])
        self.assertEqual(punctuation.synt, [])

    def test_aliases_do_not_cover_each_other(self):
        a = Lemma(orth=["A"], phon=["a"])
        b = Lemma(orth=["B"], phon=["b"])
        lexicon = self.lexicon(a, b)
        self.assertEqual(_add_uncovered_normalization_aliases(lexicon, [(a, ["x"]), (b, ["x y"])]), (0, 2))

    def test_all_segmentations_are_considered(self):
        word = Lemma(orth=["WORD"], phon=["word"])
        lexicon = self.lexicon(word, Lemma(orth=["A"], phon=["a", "a b"]), Lemma(orth=["B"], phon=["b c"]))
        self.assertEqual(_add_uncovered_normalization_aliases(lexicon, [(word, ["a b c"])]), (1, 0))

    def test_job_hash_invalidates_old_lexicon(self):
        self.assertNotEqual(HuggingFaceBpeBlissLexiconJob.hash({}), Job.hash({}))


if __name__ == "__main__":
    unittest.main()
