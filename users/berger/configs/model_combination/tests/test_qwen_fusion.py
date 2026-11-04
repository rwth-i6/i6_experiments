"""LLM token mapping, job wiring, and cached ONNX/HF numerical equivalence."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.data.llm_fusion import (
    encode_word,
    encode_words,
    rewrite_lexicon,
    token_name,
)

# The recipe LM module requires experiment-package initialization first.
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import training  # noqa: F401

from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm import (
    OnnxTokenLmParams,
    get_word_lm_config,
)


class CharacterTokenizer:
    all_special_ids = [0]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]

    def decode(self, ids, clean_up_tokenization_spaces=False):
        assert not clean_up_tokenization_spaces
        return "".join(chr(i) for i in ids)


class QwenFusionTest(unittest.TestCase):
    def test_tokenization_policy_and_failure_checks(self):
        tokenizer = CharacterTokenizer()
        self.assertEqual(encode_words(tokenizer, ["DON'T", "STOP"]), list(map(ord, " don't stop")))
        for word in ("two words", "", " padded", "line\nbreak"):
            with self.assertRaises(ValueError):
                encode_word(tokenizer, word)
        with self.assertRaises(ValueError):
            encode_word(tokenizer, "\x00")
        with patch.object(tokenizer, "encode", side_effect=lambda s, **kw: [1] if s == " a b" else list(map(ord, s))):
            with self.assertRaisesRegex(ValueError, "full-string"):
                encode_words(tokenizer, ["A", "B"])

    def test_lexicon_preserves_acoustics_and_neutral_lemmas(self):
        root = ET.fromstring(
            """<lexicon><phoneme-inventory><phoneme><symbol>p</symbol></phoneme></phoneme-inventory>
          <lemma><orth>DON'T</orth><phon>p p</phon><synt><tok>DON'T</tok></synt></lemma>
          <lemma><orth>.</orth><phon>p</phon><synt/></lemma>
          <lemma special="unknown"><orth>[UNKNOWN]</orth><synt><tok>&lt;UNK&gt;</tok></synt></lemma>
          <lemma special="sentence-begin"><orth>[BOS]</orth><synt><tok>&lt;s&gt;</tok></synt></lemma>
          <lemma special="sentence-end"><orth>[EOS]</orth><synt><tok>&lt;/s&gt;</tok></synt></lemma>
          </lexicon>"""
        )
        cfg = dict(bos_token_id=200, eos_token_id=200, vocab_size=256)
        output, report = rewrite_lexicon(root, CharacterTokenizer(), cfg)
        self.assertEqual(report["num_words"], 1)
        self.assertEqual([t.text for t in output.find("lemma/synt")], [token_name(i) for i in map(ord, " don't")])
        self.assertEqual(len(output.findall("lemma")[1].find("synt")), 0)
        self.assertEqual(len(output.findall("lemma")[2].find("synt")), 0)
        self.assertEqual(output.findall("lemma")[3].findtext("synt/tok"), token_name(200))
        self.assertEqual(output.findall("lemma")[4].findtext("synt/tok"), token_name(200))
        expected = copy.deepcopy(root)
        for tree in (expected, output):
            for lemma in tree.findall("lemma"):
                lemma.remove(lemma.find("synt"))
        self.assertEqual(ET.tostring(expected), ET.tostring(output))
        with self.assertRaisesRegex(ValueError, "outside softmax"):
            rewrite_lexicon(root, CharacterTokenizer(), dict(cfg, vocab_size=100))
        root.find("lemma").set("special", "unknown-final")
        with self.assertRaisesRegex(ValueError, "closed-tree"):
            rewrite_lexicon(root, CharacterTokenizer(), cfg)

    def test_text_variants_preserve_words_and_gate_initial_ids(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.qwen_text_variants import (
            rewrite_variant_lexicon,
            word_spellings,
            oracle_sequences,
        )

        self.assertEqual(word_spellings("DON'T"), ("don't", "Don't", "DON'T"))
        source = ET.fromstring(
            """<lexicon><phoneme-inventory><phoneme><symbol>x</symbol></phoneme></phoneme-inventory>
          <lemma name="word"><orth>EUGENIUS</orth><phon>x x</phon><eval><tok>EUGENIUS</tok></eval></lemma>
          <lemma><orth>.</orth><phon>x</phon><synt/></lemma>
          <lemma><orth>_</orth><phon>x</phon><synt/></lemma>
          <lemma special="blank"><orth>[BLANK]</orth><phon>x</phon><synt/></lemma>
          <lemma special="sentence-begin"><orth>[BOS]</orth><phon/><synt/></lemma>
          <lemma special="sentence-end"><orth>[EOS]</orth><phon/><synt/></lemma>
          <lemma special="unknown"><orth>[UNK]</orth><synt/></lemma></lexicon>"""
        )
        tokenizer = CharacterTokenizer()
        cfg = dict(vocab_size=256, bos_token_id=200, eos_token_id=200)
        out, report = rewrite_variant_lexicon(source, tokenizer, cfg)
        self.assertEqual(report["alias_ids"], [ord("E"), ord("e")])
        self.assertEqual(report["counts"], {"initial": 3, "regular": 4, "structural": 5})
        self.assertEqual(ET.tostring(source.find("phoneme-inventory")), ET.tostring(out.find("phoneme-inventory")))
        words = [l for l in out.findall("lemma") if l.findtext("orth") == "EUGENIUS"]
        self.assertEqual(len(words), 6)
        self.assertEqual(len({l.get("name") for l in words}), 6)
        surfaces = set()
        for lemma in words:
            ids = [int(t.text.removeprefix("llm_")) for t in lemma.findall("synt/tok")]
            if ids[0] >= 256:
                ids[0] = report["alias_ids"][ids[0] - 256]
                self.assertNotEqual(ids[0], ord(" "))
            else:
                self.assertEqual(ids[0], ord(" "))
            surfaces.add(tokenizer.decode(ids))
            self.assertEqual(lemma.findtext("phon"), "x x")
            self.assertEqual(lemma.findtext("eval/tok"), "EUGENIUS")
        self.assertEqual(surfaces, {p + s for p in ("", " ") for s in ("eugenius", "Eugenius", "EUGENIUS")})
        self.assertEqual(
            next(l for l in out.findall("lemma") if l.findtext("orth") == ".").findtext("synt/tok"),
            token_name(ord(".")),
        )
        self.assertEqual(len(next(l for l in out.findall("lemma") if l.findtext("orth") == "_").find("synt")), 0)
        texts = [tokenizer.decode(ids) for ids in oracle_sequences(tokenizer, ["HI", "THERE"])]
        self.assertEqual(len(texts), 18)
        self.assertTrue(all(" " in s.strip() for s in texts))
        bad = copy.deepcopy(source)
        bad.find("lemma").set("special", "unknown-final")
        with self.assertRaisesRegex(ValueError, "closed-tree"):
            rewrite_variant_lexicon(bad, tokenizer, cfg)

    def test_variant_traceback_restores_output_and_preserves_raw_route(self):
        from types import SimpleNamespace
        from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_traceback import (
            QwenTextVariantTracebackFormatter,
            original_qwen_orth,
        )
        from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog import BpeTracebackFormatter

        raw = [
            SimpleNamespace(lemma=name, start_time=i, end_time=i + 1, am_score=0.0, lm_score=0.0)
            for i, name in enumerate(("[BLANK]", "qwen-v1:0:3:EUGENIUS", "qwen-v1:1:1:HERE", "[SENTENCE-END]"))
        ]
        expected = [SimpleNamespace(**dict(vars(x), lemma=original_qwen_orth(x.lemma))) for x in raw]
        formatter = QwenTextVariantTracebackFormatter()
        self.assertEqual(formatter.traceback_to_transcription(raw), "EUGENIUS HERE")
        self.assertEqual(
            formatter.traceback_to_ctm_str(raw, 80), BpeTracebackFormatter().traceback_to_ctm_str(expected, 80)
        )
        self.assertEqual(raw[1].lemma, "qwen-v1:0:3:EUGENIUS")
        with self.assertRaisesRegex(ValueError, "Malformed"):
            original_qwen_orth("qwen-v1:bad")

    @unittest.skipUnless(os.environ.get("RASR_LIB_DIR"), "Set RASR_LIB_DIR for the native variant-route test")
    def test_native_variant_routes_match_best_hf_spelling(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from i6_core.rasr import RasrCommand
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.qwen_text_variants import (
            rewrite_variant_lexicon,
            oracle_sequences,
            VARIANT_NATIVE_RUNNER,
        )
        from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.transformer_lm.qwen import (
            Qwen3TextVariantsRasrLm,
            export_qwen_onnx,
        )
        from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog_rasr_config import (
            get_tree_timesync_recog_config,
            TreeTimesyncRecogParams,
        )
        from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.label_scorer_config import (
            get_no_op_label_scorer_config,
        )

        torch.set_num_threads(2)
        torch.manual_seed(9)
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            bos_token_id=1,
            eos_token_id=1,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        tokenizer = CharacterTokenizer()
        source = ET.fromstring(
            """<lexicon><phoneme-inventory>
            <phoneme><symbol>blank</symbol><variation>none</variation></phoneme>
            <phoneme><symbol>a</symbol><variation>none</variation></phoneme>
            <phoneme><symbol>b</symbol><variation>none</variation></phoneme></phoneme-inventory>
            <lemma special="blank"><orth>[BLANK]</orth><phon>blank</phon><synt/></lemma>
            <lemma><orth>HI</orth><phon>a</phon></lemma>
            <lemma><orth>THERE</orth><phon>b</phon></lemma>
            <lemma special="sentence-begin"><orth>[BOS]</orth><phon/><synt/></lemma>
            <lemma special="sentence-end"><orth>[EOS]</orth><phon/><synt/></lemma></lexicon>"""
        )
        lex, report = rewrite_variant_lexicon(source, tokenizer, config.to_dict())
        candidates = []
        with torch.inference_mode():
            for ids in oracle_sequences(tokenizer, ["HI", "THERE"]):
                seq = [1, *ids]
                scores = -model(torch.tensor([seq])).logits[0].log_softmax(-1)
                candidates.append(float(scores[torch.arange(len(seq)), torch.tensor([*ids, 1])].double().sum()))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            lex_file, vocab, model_file = (directory / name for name in ("lex.xml", "vocab.txt", "model.onnx"))
            ET.ElementTree(lex).write(lex_file, encoding="utf-8")
            vocab.write_text("".join(f"{token_name(i)} {i}\n" for i in range(256 + len(report["alias_ids"]))))
            export_qwen_onnx(Qwen3TextVariantsRasrLm(model, report["alias_ids"]), str(model_file))
            lm = get_word_lm_config(
                tk.Path(str(lex_file)),
                OnnxTokenLmParams(tk.Path(str(model_file)), tk.Path(str(vocab)), scale=1.0, use_gpu=False),
            )
            job = get_tree_timesync_recog_config(
                tk.Path(str(lex_file)),
                [get_no_op_label_scorer_config()],
                TreeTimesyncRecogParams(
                    collapse_repeated_labels=True,
                    max_beam_sizes=[256],
                    score_thresholds=[200.0],
                    max_word_end_beam_size=256,
                    word_end_score_threshold=200.0,
                    sentence_end_fallback=False,
                ),
                lm_config=lm,
            ).creator
            job.config.lib_rasr.acoustic_model.state_tying.type = "monophone"
            filename = directory / "rasr.config"
            RasrCommand.write_config(job.config, job.post_config, str(filename))
            payload = dict(words=["HI", "THERE"], labels=[1, 2], vocab_size=3, expected_cost=min(candidates))
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    VARIANT_NATIVE_RUNNER,
                    os.environ["RASR_LIB_DIR"],
                    str(filename),
                    json.dumps(payload),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout[-5000:] + proc.stderr[-5000:])

    def test_tiny_qwen_cached_onnx_matches_huggingface(self):
        import numpy as np
        import torch
        import onnxruntime as ort
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.transformer_lm.qwen import (
            Qwen3RasrLm,
            export_qwen_onnx,
            verify_qwen_export,
            Qwen3TextVariantsRasrLm,
            verify_qwen_text_variants,
        )

        torch.set_num_threads(2)
        torch.manual_seed(3)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            # Like the real 0.6B checkpoint, heads * head_dim != hidden_size.
            head_dim=16,
            attention_dropout=0.0,
            rope_theta=1e6,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        sequences = [[1, 2, 3, 4, 5], [1, 3, 4, 5], [1, 8, 5, 4, 2, 7]]
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "model.onnx")
            export_qwen_onnx(Qwen3RasrLm(model), filename)
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 2
            session = ort.InferenceSession(filename, sess_options=options, providers=["CPUExecutionProvider"])
            self.assertLess(verify_qwen_export(model, session, sequences), 1e-5)
            original_run = session.run

            def nan_run(*args):
                outputs = original_run(*args)
                outputs[0][:] = np.nan
                return outputs

            with patch.object(session, "run", side_effect=nan_run):
                with self.assertRaisesRegex(ValueError, "Nonfinite"):
                    verify_qwen_export(model, session, sequences)
            aliases = [2, 3, 8]
            variant_file = str(Path(directory) / "variants.onnx")
            export_qwen_onnx(Qwen3TextVariantsRasrLm(model, aliases), variant_file)
            variant_session = ort.InferenceSession(
                variant_file, sess_options=options, providers=["CPUExecutionProvider"]
            )
            self.assertLess(verify_qwen_text_variants(model, variant_session, sequences, aliases), 1e-5)
            original_variant_run = variant_session.run

            def bad_mask(*args):
                outputs = original_variant_run(*args)
                outputs[0][:, :, 32:] = 0
                return outputs

            with patch.object(variant_session, "run", side_effect=bad_mask):
                with self.assertRaisesRegex(ValueError, "alias score/mask"):
                    verify_qwen_text_variants(model, variant_session, sequences, aliases)
