"""Normalization and controlled experiment wiring for the Granite word LM."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from i6_core.lib.lexicon import Lemma, Lexicon
from i6_core.util import uopen, write_xml
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm_text_normalization import (
    NormalizeLexiconLmTokensJob,
    PrepareLmTrainingTextJob,
    BuildNormalizedWordSearchLexiconJob,
    _split_normalized_pronunciation,
    _has_normalized_word_path,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.hf_normalizer import EnglishTextNormalizer


class HfLmNormalizationTest(unittest.TestCase):
    def test_attached_suffix_split_and_unsplittable_contraction(self):
        normalize = EnglishTextNormalizer()
        decode = lambda tokens: "".join(tokens).replace("▁", " ").strip()
        self.assertEqual(
            _split_normalized_pronunciation(("▁z", "yl", "'", "s"), ("ZYL", "S"), decode, normalize),
            (("▁z", "yl"), ("'", "s")),
        )
        self.assertIsNone(_split_normalized_pronunciation(("▁don", "'", "t"), ("DO", "NOT"), decode, normalize))
        self.assertEqual(
            _split_normalized_pronunciation(("▁do", "▁not"), ("DO", "NOT"), decode, normalize),
            (("▁do",), ("▁not",)),
        )

    def test_exact_path_coverage_includes_neutral_punctuation(self):
        paths = {"ZYL": {("▁z", "yl")}, "S": {("s",)}}
        labels = ("▁z", "yl", "'", "s")
        self.assertFalse(_has_normalized_word_path(labels, ("ZYL", "S"), paths, set()))
        self.assertTrue(_has_normalized_word_path(labels, ("ZYL", "S"), paths, {("'",)}))
        self.assertFalse(_has_normalized_word_path(labels, ("ZYL",), paths, {("'",)}))

    def test_search_lexicon_modes(self):
        class Tokenizer:
            unk_token = "<unk>"

            def get_vocab(self):
                return {t: i for i, t in enumerate(["<unk>", "▁z", "yl", "'", "s", "▁s", "▁do", "▁not", "▁don", "t"])}

            def tokenize(self, text):
                return {"zyl": ["▁z", "yl"], "s": ["▁s"], "do": ["▁do"], "not": ["▁not"]}[text]

            def convert_tokens_to_string(self, tokens):
                return "".join(tokens).replace("▁", " ").strip()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Lexicon()
            for label in Tokenizer().get_vocab():
                source.add_phoneme(label, variation="none")
            source.add_lemma(Lemma(orth=["ZYL'S"], phon=["▁z yl ' s"], synt=["ZYL", "S"]))
            source.add_lemma(Lemma(orth=["DON'T"], phon=["▁don ' t"], synt=["DO", "NOT"]))
            source.add_lemma(Lemma(orth=["[BOS]"], phon=[""], synt=["<s>"], special="sentence-begin"))
            write_xml(str(root / "input.xml"), source.to_xml())
            results = {}
            for mode in ["canonical", "split"]:
                job = SimpleNamespace(
                    lexicon=tk.Path(str(root / "input.xml")),
                    huggingface_repo_dir=tk.Path("unused"),
                    mode=mode,
                    casings=("lower",),
                    out_lexicon=tk.Path(str(root / f"{mode}.xml")),
                    out_report=tk.Path(str(root / f"{mode}.json")),
                    out_uncovered=tk.Path(str(root / f"{mode}.jsonl.gz")),
                )
                with patch(
                    "i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm_text_normalization._load_tokenizer",
                    return_value=Tokenizer(),
                ):
                    BuildNormalizedWordSearchLexiconJob.run(job)
                lex = Lexicon()
                lex.load(job.out_lexicon.get_path())
                self.assertTrue(all(len(l.synt) <= 1 for l in lex.lemmata))
                results[mode] = {tuple(l.orth): l for l in lex.lemmata}
            self.assertEqual(results["canonical"].keys(), results["split"].keys())
            self.assertIn("' s", results["split"][("S",)].phon)
            self.assertNotIn("' s", results["canonical"][("S",)].phon)
            self.assertNotIn("▁don", results["split"][("DO",)].phon)
            self.assertNotIn(("DON'T",), results["split"])
            self.assertEqual(json.loads((root / "canonical.json").read_text())["covered_source_pronunciations"], 0)
            self.assertEqual(json.loads((root / "split.json").read_text())["covered_source_pronunciations"], 1)

    def test_lexicon_changes_only_syntactic_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lexicon = Lexicon()
            lexicon.add_phoneme("piece", variation="none")
            for lemma in [
                Lemma(orth=["DON'T"], phon=["piece"]),
                Lemma(orth=["TWENTY"], phon=["piece"]),
                Lemma(orth=["OH"], phon=["piece"]),
                Lemma(orth=["UM"], phon=["piece"]),
                Lemma(orth=["."], phon=["piece"], synt=[]),
                Lemma(orth=["[UNKNOWN]"], synt=["<UNK>"], special="unknown"),
            ]:
                lexicon.add_lemma(lemma)
            write_xml(str(root / "input.xml"), lexicon.to_xml())
            job = SimpleNamespace(
                lexicon=tk.Path(str(root / "input.xml")),
                out_lexicon=tk.Path(str(root / "output.xml.gz")),
                out_report=tk.Path(str(root / "report.json")),
            )
            NormalizeLexiconLmTokensJob.run(job)
            output = Lexicon()
            output.load(job.out_lexicon.get_path())
            self.assertEqual(output.phonemes, lexicon.phonemes)
            for before, after in zip(lexicon.lemmata, output.lemmata):
                self.assertEqual(before.orth, after.orth)
                self.assertEqual(before.phon, after.phon)
                self.assertEqual(before.special, after.special)
            self.assertEqual(
                [lemma.synt for lemma in output.lemmata],
                [["DO", "NOT"], ["20"], ["OH"], [], [], ["<UNK>"]],
            )

    def test_streamed_training_text_and_oov_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("I DON'T HAVE TWENTY CATS\nUM\nOH\nTWENTY ONE\n")
            (root / "vocab.txt").write_text("I\nDO\nNOT\nHAVE\n20\nCATS\nOH\n<unk>\n")
            job = SimpleNamespace(
                texts=[tk.Path(str(root / "input.txt"))],
                vocabulary=tk.Path(str(root / "vocab.txt")),
                normalize_hf=True,
                out_text=tk.Path(str(root / "text.gz")),
                out_report=tk.Path(str(root / "report.json")),
                out_oov_words=tk.Path(str(root / "oov.tsv")),
            )
            PrepareLmTrainingTextJob.run(job)
            with uopen(job.out_text, "rt") as src:
                self.assertEqual(src.read(), "I DO NOT HAVE 20 CATS\nOH\n21\n")
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["empty_sentences"], 1)
            self.assertEqual(report["tokens"], 8)
            self.assertEqual(report["oov_tokens"], 1)
            self.assertEqual((root / "oov.tsv").read_text(), "1\t21\n")
            job.normalize_hf = False
            PrepareLmTrainingTextJob.run(job)
            with uopen(job.out_text, "rt") as src:
                self.assertEqual(src.read(), (root / "input.txt").read_text())

    def test_matched_sweep(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.recognition.ctc_granite import (
            text_normalized_lm_variants,
        )

        target = (
            "i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech."
            "lm_text_normalization.get_text_normalization_4gram_lms"
        )
        paths = {"original": tk.Path("original.arpa.gz"), "normalized": tk.Path("normalized.arpa.gz")}
        with patch(target, return_value=paths):
            variants = text_normalized_lm_variants()
        self.assertEqual(len(variants), 20)
        for offset, condition in enumerate(["published", "original", "normalized-canonical", "normalized-split"]):
            arm = variants[offset * 5 : offset * 5 + 5]
            self.assertEqual([v.search_algorithm_params.word_lm_params.scale for v in arm], [0.0, 0.2, 0.4, 0.6, 0.8])
            for variant in arm:
                self.assertEqual(variant.normalize_lm_tokens, condition.startswith("normalized-"))
                self.assertEqual(
                    variant.search_algorithm_params.word_lm_params.lm_file,
                    paths.get("normalized" if condition.startswith("normalized-") else condition),
                )
                if condition.startswith("normalized-"):
                    self.assertEqual(variant.normalized_lexicon_mode, condition.split("-", 1)[1])
                self.assertEqual(variant.prior_scale, 0.2)
                self.assertEqual(variant.search_algorithm_params.max_beam_sizes, [256])

    def test_recognition_receives_single_token_search_lexicon(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.recognition import (
            common,
            ctc_granite,
        )

        for mode in ["canonical", "split"]:
            variant = ctc_granite.default_offline_tree_4gram_recog_variant()
            with (
                patch.object(common, "get_hf_ctc_vocab_file", return_value=tk.Path("vocab.txt")),
                patch.object(common, "get_hf_bpe_bliss_lexicon", return_value=tk.Path("acoustic.xml")),
                patch.object(common, "get_bliss_lexicon", return_value=tk.Path("base.xml")),
                patch.object(common.tk, "register_output"),
                patch.object(common, "_run_single_variant", return_value=[]) as decode,
            ):
                common.run_single_hf_bpe_variant(
                    model_descriptor="test",
                    huggingface_repo_dir=tk.Path("hf-repo"),
                    encoder_serializers=None,
                    label_scorer_configs=[],
                    blank_index=0,
                    variant=variant,
                    corpora=["dev-other"],
                    normalize_lm_tokens=True,
                    normalized_lexicon_mode=mode,
                )
            lexicon = decode.call_args.kwargs["lexicon_file"]
            self.assertIsInstance(lexicon.creator, BuildNormalizedWordSearchLexiconJob)
            self.assertEqual(lexicon.creator.mode, mode)
            self.assertIsInstance(lexicon.creator.lexicon.creator, NormalizeLexiconLmTokensJob)


if __name__ == "__main__":
    unittest.main()
