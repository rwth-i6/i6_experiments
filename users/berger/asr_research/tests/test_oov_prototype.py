"""Prototype graph, lexicon, boundary, and mixed traceback regression tests."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from i6_core.lib.lexicon import Lemma, Lexicon
from i6_core.util import uopen, write_xml
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_prototype import (
    BuildGraniteOovLexiconJob,
    RemoveEmptyUnknownPronunciationsJob,
    ValidateOovPrototypeInputsJob,
    classify_piece,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.oov_prototype import prototype_variants
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.oov_traceback import (
    GraniteOovTracebackFormatter,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog_rasr_config import (
    TreeTimesyncRecogParams,
    get_tree_timesync_recog_config,
)


class OovPrototypeTest(unittest.TestCase):
    def test_remove_only_empty_unknown_pronunciations(self):
        import xml.etree.ElementTree as ET

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = ET.fromstring(
                "<lexicon><phoneme-inventory><phoneme><symbol>x</symbol></phoneme></phoneme-inventory>"
                '<lemma special="unknown"><orth>[UNKNOWN]</orth><phon/><phon>  </phon>'
                "<synt><tok>&lt;UNK&gt;</tok></synt></lemma>"
                '<lemma special="sentence-end"><orth>[EOS]</orth><phon/></lemma>'
                "<lemma><orth>X</orth><phon>x</phon></lemma></lexicon>"
            )
            write_xml(str(root / "input.xml.gz"), source)
            job = SimpleNamespace(
                lexicon=tk.Path(str(root / "input.xml.gz")),
                out_lexicon=tk.Path(str(root / "output.xml.gz")),
            )
            RemoveEmptyUnknownPronunciationsJob.run(job)
            with uopen(job.out_lexicon, "rt") as src:
                result = ET.parse(src).getroot()
            unknown = result.find('lemma[@special="unknown"]')
            self.assertEqual(unknown.findall("phon"), [])
            self.assertEqual(unknown.findtext("synt/tok"), "<UNK>")
            self.assertEqual(unknown.findtext("orth"), "[UNKNOWN]")
            for expected, actual in zip(list(source), list(result)):
                if expected.get("special") != "unknown":
                    self.assertEqual(
                        ET.canonicalize(ET.tostring(expected), strip_text=True),
                        ET.canonicalize(ET.tostring(actual), strip_text=True),
                    )
            # Existing nonempty unknown pronunciations must remain intact.
            ET.SubElement(source.find('lemma[@special="unknown"]'), "phon").text = "x"
            write_xml(str(root / "input.xml.gz"), source)
            RemoveEmptyUnknownPronunciationsJob.run(job)
            with uopen(job.out_lexicon, "rt") as src:
                result = ET.parse(src).getroot()
            self.assertEqual([p.text for p in result.findall('lemma[@special="unknown"]/phon')], ["x"])

    def test_granite_prototype_explicitly_requests_unknown_lemma(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import oov_prototype
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.huggingface import HuggingFaceBpeBlissLexiconJob

        model = SimpleNamespace(model_config=SimpleNamespace(target_size=185))
        with (
            patch.object(oov_prototype.ctc_bpe, "run", return_value=[]),
            patch.object(oov_prototype.ctc_granite, "run", return_value=[]) as granite_run,
        ):
            oov_prototype.run(model)
        variants = granite_run.call_args.kwargs["variants"]
        original = next(v for v in variants if v.descriptor.endswith("_original-lexicon"))
        original_source = original.prepared_lexicon_file.creator.lexicon
        self.assertIsInstance(original_source.creator, RemoveEmptyUnknownPronunciationsJob)
        self.assertIsInstance(original_source.creator.lexicon.creator, HuggingFaceBpeBlissLexiconJob)
        self.assertTrue(original_source.creator.lexicon.creator.add_unknown_lemma)
        dump_config = original.prepared_lexicon_file.creator.state_tying.creator.config
        allophones = dump_config.allophone_tool.acoustic_model.allophones
        self.assertTrue(allophones.add_all)
        self.assertFalse(allophones.add_from_lexicon)
        self.assertNotIn("add-from-file", str(allophones))
        fallback = next(v for v in variants if "known-excluding" in v.descriptor)
        fallback_source = fallback.prepared_lexicon_file.creator.lexicon.creator.lexicon
        self.assertEqual(fallback_source, original_source)
        # Preserve the general factory's existing closed-vocabulary default.
        self.assertFalse(HuggingFaceBpeBlissLexiconJob(tk.Path("base"), tk.Path("repo")).add_unknown_lemma)

    def test_real_decoder_smoke_graph(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_decoder_smoke import (
            OovPrototypeDecoderSmokeJob,
            smoke_lexicon,
            RUNNER,
        )

        job = OovPrototypeDecoderSmokeJob(tk.Path("prototype-bin"), tk.Path("/usr/bin/python3"))
        self.assertEqual(len(job.cases), 12)
        self.assertEqual(len({case["name"] for case in job.cases}), 12)
        compile(RUNNER, "smoke-runner", "exec")
        for kind in ("bpe", "granite"):
            lex, size = smoke_lexicon(kind)
            self.assertEqual(len(lex.phonemes), size)
            self.assertEqual(list(lex.phonemes)[0], "<blank>")
            known = next(case for case in job.cases if case["name"] == f"{kind}-known-exclusion")
            self.assertEqual(known["expected_score"], 0)
            self.assertEqual(known["config"].creator.config.lib_rasr.search_algorithm.unknown_word_penalty, -10)
        lex, _ = smoke_lexicon("granite")
        self.assertEqual(lex.lemmata[-1].special, "unknown-word-start+nonword")

    def test_pinned_build_and_lm_image(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025 import tools
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech import lm

        binary = tools.get_oov_prototype_rasr_binary_path()
        build = binary.creator
        clone = build.source_folder.creator
        self.assertEqual(clone.commit, "5083eddece2982cb3b55c09daf1368e6d460f467")
        self.assertIn("unknown-word-start+nonword", clone.patches[0])
        self.assertIsNone(binary.hash_overwrite)
        self.assertIsNone(build.source_folder.hash_overwrite)
        crps = []

        def image(crp, **kwargs):
            crps.append(crp)
            return SimpleNamespace(out_image=tk.Path("prototype.image"))

        with patch.object(lm, "CreateLmImageJob", side_effect=image):
            config = lm.get_word_lm_config(
                tk.Path("unique-prototype-lexicon"),
                lm.ArpaLmParams(scale=0.3, lm_file=tk.Path("test.arpa")),
                rasr_binary_path_override=binary,
            )
        self.assertEqual(crps[0].lm_util_exe, binary.join_right("lm-util.linux-x86_64-standard"))
        self.assertEqual(config.scale, 0.3)

    def test_piece_roles(self):
        for token, role in [
            ("▁un", "start"),
            ("usual", "internal"),
            ("▁", "separator"),
            ("'", "internal"),
            (".", "internal"),
            ("▁!", "start"),
        ]:
            self.assertEqual(classify_piece(token, special=False, blank=False), role)
        self.assertEqual(classify_piece("<unk>", special=True, blank=True), "blank")
        for token in ["▁a▁b", "a b", "a▁", "▁▁", ""]:
            with self.assertRaises(ValueError):
                classify_piece(token, special=False, blank=False)

    def test_matrix(self):
        for kind, expected in [("bpe", 64), ("granite", 43)]:
            variants = prototype_variants(kind)
            self.assertEqual(len(variants), expected)
            self.assertEqual(len({v.descriptor for v in variants}), expected)
            tree = [v for v in variants if isinstance(v.search_algorithm_params, TreeTimesyncRecogParams)]
            for v in tree:
                p = v.search_algorithm_params
                self.assertEqual(v.prior_scale, 0)
                self.assertEqual(v.blank_penalty, 0)
                self.assertEqual(p.recombination_mode, "on")
                self.assertFalse(p.sentence_end_fallback)
                self.assertEqual(len(p.max_beam_sizes), 1)
                self.assertEqual(len(p.score_thresholds), 1)
                if p.unknown_word_fallback == "disabled":
                    self.assertTrue(
                        any(
                            other.search_algorithm_params.unknown_word_fallback == "known-excluding"
                            and other.search_algorithm_params.word_lm_params == p.word_lm_params
                            and other.search_algorithm_params.max_beam_sizes == p.max_beam_sizes
                            and other.search_algorithm_params.score_thresholds == p.score_thresholds
                            and other.search_algorithm_params.unknown_word_penalty == 0
                            for other in tree
                        )
                    )
            expected_penalties = {-4, -2, -1, 0, 1, 2, 4}
            if kind == "bpe":
                expected_penalties.update({6, 8, 10, 12})
                expected_penalties.update(range(10, 101, 10))
            self.assertEqual({v.search_algorithm_params.unknown_word_penalty for v in tree}, expected_penalties)

    def test_bpe_positive_penalty_followup_preserves_original_conditions(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import oov_prototype

        with patch.object(oov_prototype, "BPE_CLOSED_LIMIT_PENALTIES", ()):
            expanded = prototype_variants("bpe")
            with patch.object(oov_prototype, "BPE_POSITIVE_PENALTIES", ()):
                original = prototype_variants("bpe")
        self.assertEqual(len(original), 47)
        for old, current in zip(original, expanded):
            self.assertEqual(old, current)
        added = expanded[len(original) :]
        self.assertEqual(
            {
                (v.search_algorithm_params.word_lm_params.scale, v.search_algorithm_params.unknown_word_penalty)
                for v in added
            },
            {(scale, beta) for scale in (0.4, 0.6) for beta in (6, 8, 10, 12)},
        )
        for variant in added:
            params = variant.search_algorithm_params
            self.assertEqual(params.unknown_word_fallback, "known-excluding")
            self.assertEqual(params.max_beam_sizes, [256])
            self.assertEqual(params.score_thresholds, [16.0])
            self.assertEqual(params.max_word_end_beam_size, 256)
            self.assertEqual(params.word_end_score_threshold, 16.0)

    def test_bpe_closed_limit_sweep(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import oov_prototype

        expanded = prototype_variants("bpe")
        with patch.object(oov_prototype, "BPE_CLOSED_LIMIT_PENALTIES", ()):
            original = prototype_variants("bpe")
        self.assertEqual(len(original), 55)
        self.assertEqual(expanded[: len(original)], original)
        added = expanded[len(original) :]
        self.assertEqual(len(added), 9)
        self.assertEqual([v.search_algorithm_params.unknown_word_penalty for v in added], list(range(20, 101, 10)))
        anchor = next(v for v in original if "known-excluding_4gram-lm0.6_beta10_" in v.descriptor)
        from dataclasses import replace

        for variant in added:
            self.assertEqual(
                replace(
                    variant,
                    descriptor=anchor.descriptor,
                    search_algorithm_params=replace(variant.search_algorithm_params, unknown_word_penalty=10.0),
                ),
                anchor,
            )
        self.assertEqual(sum(v.descriptor == anchor.descriptor for v in expanded), 1)

    def test_config_opt_in(self):
        configs = []

        def capture(config, post):
            configs.append(config)
            return SimpleNamespace(out_config=tk.Path("test.config"))

        with patch(
            "i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog_rasr_config.WriteRasrConfigJob",
            side_effect=capture,
        ):
            for p in [TreeTimesyncRecogParams(True, [6]), prototype_variants("granite")[1].search_algorithm_params]:
                get_tree_timesync_recog_config(tk.Path("lexicon"), [], p)
        self.assertNotIn("unknown-word-fallback", str(configs[0]))
        self.assertIn("unknown-word-fallback", str(configs[1]))
        self.assertIn("word-start-marked", str(configs[1]))
        self.assertEqual(configs[1].lib_rasr.acoustic_model.state_tying.type, "monophone")
        self.assertEqual(configs[1].lib_rasr.acoustic_model.hmm.states_per_phone, 1)

    def test_mixed_traceback(self):
        formatter = GraniteOovTracebackFormatter(tk.Path("unused"))
        formatter.tokenizer = SimpleNamespace(
            convert_tokens_to_string=lambda xs: "".join(xs).replace("▁", " ").lstrip()
        )

        def item(word, start, end):
            return SimpleNamespace(lemma=word, start_time=start, end_time=end, confidence_score=None)

        trace = [
            item("HELLO", 0, 2),
            item("▁un&&", 2, 4),
            item("usual&&", 4, 7),
            item("[BLANK]", 7, 8),
            item("▁&&", 8, 9),
            item("▁word&&", 9, 12),
            item(".&&", 12, 13),
            item("▁&&", 13, 14),
        ]
        self.assertEqual(formatter.traceback_to_transcription(trace), "HELLO UNUSUAL WORD.")
        lines = formatter.traceback_to_ctm_str(trace, 80).splitlines()
        self.assertEqual(lines[1].split()[2:5], ["0.160", "0.400", "UNUSUAL"])
        self.assertEqual(lines[2].split()[2:5], ["0.720", "0.320", "WORD."])
        decimal = [item("▁1&&", 0, 1), item(".&&", 1, 2), item("5&&", 2, 3)]
        self.assertEqual(formatter.traceback_to_transcription(decimal), "1.5")
        self.assertNotIn("&&", formatter.traceback_to_ctm_str(trace, 80))

    def test_byte_fallback_detokenization_and_timing(self):
        import re

        def decode(tokens):
            result, pending = "", bytearray()
            for token in tokens:
                if re.fullmatch(r"<0x[0-9A-F]{2}>", token):
                    pending.append(int(token[3:5], 16))
                else:
                    result += pending.decode("utf-8", errors="replace") + token.replace("▁", " ")
                    pending.clear()
            return (result + pending.decode("utf-8", errors="replace")).removeprefix(" ")

        formatter = GraniteOovTracebackFormatter(tk.Path("unused"))
        formatter.tokenizer = SimpleNamespace(convert_tokens_to_string=decode)
        for tokens, expected, start, end in [
            (["▁", "<0xC2>", "<0xA2>", "3", "7", "▁and"], "¢37 AND", 1, 5),
            (["▁", "<0xE2>", "<0x82>", "<0xAC>", "▁and"], "€ AND", 1, 4),
            (["▁", "<0xF0>", "<0x9F>", "<0xA6>", "<0x8A>", "▁and"], "🦊 AND", 1, 5),
            (["▁", "<0xC2>"], "�", 1, 2),
        ]:
            with self.subTest(tokens=tokens):
                trace = [
                    SimpleNamespace(lemma=token + "&&", start_time=i, end_time=i + 1, confidence_score=0.9)
                    for i, token in enumerate(tokens)
                ]
                self.assertEqual(formatter.traceback_to_transcription(trace), expected)
                words = formatter._words(trace)
                self.assertEqual(words[0][1:3], (start, end))
                if len(words) > 1:
                    self.assertEqual(words[1][1:3], (len(tokens) - 1, len(tokens)))
                # A preceding ordinary lemma is not glued to the byte-decoded word.
                known = SimpleNamespace(lemma="PAID", start_time=0, end_time=1, confidence_score=None)
                self.assertEqual(formatter.traceback_to_transcription([known, *trace]), "PAID " + expected)

    def test_lexicon_and_state_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                dict(id=i, label=label, token=token, role=role)
                for i, (label, token, role) in enumerate(
                    [
                        ("<blank>", "<unk>", "blank"),
                        ("▁a", "▁a", "start"),
                        ("b", "b", "internal"),
                        ("#", "#", "internal"),
                        ("▁", "▁", "separator"),
                    ]
                )
            ]
            lex = Lexicon()
            for row in rows:
                lex.add_phoneme(row["label"], variation="none")
            lex.add_lemma(Lemma(orth=["A"], phon=["▁a", "▁a ▁a"], synt=["A"]))
            lex.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=[""], synt=["<UNK>"], special="unknown"))
            write_xml(str(root / "source.xml.gz"), lex.to_xml())
            (root / "pieces.json").write_text(json.dumps(rows))
            job = SimpleNamespace(
                lexicon=tk.Path(str(root / "source.xml.gz")),
                classification=tk.Path(str(root / "pieces.json")),
                out_lexicon=tk.Path(str(root / "fallback.xml.gz")),
                out_report=tk.Path(str(root / "report.json")),
            )
            BuildGraniteOovLexiconJob.run(job)
            result = Lexicon()
            result.load(job.out_lexicon.get_path())
            self.assertEqual(list(result.phonemes), list(lex.phonemes))
            self.assertEqual(result.lemmata[0].phon, ["▁a"])
            self.assertEqual(result.lemmata[-1].special, "unknown-word-start+nonword")
            self.assertEqual(result.lemmata[-1].synt, [])
            (root / "vocab").write_text("\n".join(lex.phonemes) + "\n")
            tying = "".join(
                f"{p}{{#+#}}{flags}.0 {i}\n" for i, p in enumerate(lex.phonemes) for flags in ["", "@i", "@f", "@i@f"]
            )
            (root / "tying").write_text("# comment\n   # indented comment\n\n" + tying)
            (root / "lm.arpa").write_text("\\data\\\nngram 1=2\n\\1-grams:\n-3 <UNK>\n-1 A\n\\end\\\n")
            check = SimpleNamespace(
                lexicon=job.out_lexicon,
                vocab=tk.Path(str(root / "vocab")),
                label_aliases={},
                state_tying=tk.Path(str(root / "tying")),
                arpa_lm=tk.Path(str(root / "lm.arpa")),
                out_lexicon=tk.Path(str(root / "checked.xml.gz")),
                out_report=tk.Path(str(root / "checked.json")),
            )
            ValidateOovPrototypeInputsJob.run(check)
            self.assertEqual((root / "checked.xml.gz").read_bytes(), (root / "fallback.xml.gz").read_bytes())
            (root / "tying").write_text(tying.replace("#{#+#}.0 3\n", ""))
            with self.assertRaisesRegex(ValueError, "Incorrect emission index for #"):
                ValidateOovPrototypeInputsJob.run(check)
            (root / "tying").write_text(tying.replace("b{#+#}@i@f.0 2", "b{#+#}@i@f.0 99"))
            with self.assertRaisesRegex(ValueError, "Incorrect emission index"):
                ValidateOovPrototypeInputsJob.run(check)


if __name__ == "__main__":
    unittest.main()
