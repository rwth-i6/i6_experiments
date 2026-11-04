"""Prototype graph, lexicon, boundary, and mixed traceback regression tests."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from i6_core.lib.lexicon import Lemma, Lexicon
from i6_core.util import uopen, write_xml
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_prototype import (
    BuildGraniteOovLexiconJob,
    RemoveEmptyUnknownPronunciationsJob,
    ValidateOovPrototypeInputsJob,
    classify_piece,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.oov_traceback import (
    GraniteOovTracebackFormatter,
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
