"""Small CPU tests for UNK validation without loading the real 96-layer LM."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sisyphus import tk

# Initialize the experiment package before the LM modules (existing import cycle).
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import (
    oov_length,  # noqa: F401 -- initializes the existing import cycle
)
from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_neural_lm import (
    check_unknown_mapping,
    probe_unknown_scores,
)


class FakeSession:
    def __init__(self, bad=None):
        self.bad, self.feeds = bad, []

    def get_inputs(self):
        return [
            SimpleNamespace(name=n, type="tensor(int32)", shape=s)
            for n, s in (("tokens", [None, None]), ("tokens:size1", [None]), ("state_l000_k_in:size1", [None]))
        ] + [
            SimpleNamespace(name=f"state_l000_{kind}_in", type="tensor(float)", shape=[None, None, 4])
            for kind in ("k", "v")
        ]

    def run(self, names, feed):
        assert names == ["scores"]
        self.feeds.append(feed)
        n = feed["tokens"].shape[1]
        scores = np.broadcast_to(-np.log(np.array([0.2, 0.3, 0.5], dtype=np.float32)), (1, n, 3)).copy()
        if self.bad == "shape":
            scores = scores[..., :2]
        if self.bad == "nan":
            scores[0, -1, 1] = np.nan
        if self.bad == "logits":
            scores[:] = 0
        if self.bad == "negative":
            scores[:] = -np.log(3)
        if self.bad == "zero-unk":
            scores[0, -1] = [np.log(2), 1000, np.log(2)]
        return [scores]


class OovNeuralLmTest(unittest.TestCase):

    def test_mapping_allows_shared_bos_eos_id_but_requires_exact_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            vocab, lex = Path(d) / "vocab", Path(d) / "lex.xml"
            vocab.write_text("<s> 0\n</s> 0\n<UNK> 1\nTHE 2\n")
            lex.write_text('<lexicon><lemma special="unknown"><synt><tok>&lt;UNK&gt;</tok></synt></lemma></lexicon>')
            ids, size = check_unknown_mapping((tk.Path(str(lex)),), tk.Path(str(vocab)), "<UNK>")
            self.assertEqual(size, 3)
            self.assertEqual(ids["<UNK>"], 1)
            with self.assertRaisesRegex(ValueError, "absent"):
                check_unknown_mapping((tk.Path(str(lex)),), tk.Path(str(vocab)), "<unk>")
            lex.write_text('<lexicon><lemma special="unknown"><synt/></lemma></lexicon>')
            with self.assertRaisesRegex(ValueError, "exactly one unknown"):
                check_unknown_mapping((tk.Path(str(lex)),), tk.Path(str(vocab)), "<UNK>")

    def test_invalid_outputs_fail(self):
        for bad in ("shape", "nan", "logits", "negative", "zero-unk"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe_unknown_scores(FakeSession(bad), {"<s>": 0, "<UNK>": 1, "THE": 2}, 3, "<UNK>")


def test_probe_reports_the_unknown_column_and_feeds_its_actual_id():
    session = FakeSession()
    result = probe_unknown_scores(session, {"<s>": 0, "<UNK>": 1, "THE": 2}, 3, "<UNK>")
    np.testing.assert_allclose([r["unknown_probability"] for r in result], 0.3, rtol=1e-6)
    assert session.feeds[1]["tokens"].tolist() == [[0, 1]]
    assert session.feeds[2]["tokens"].tolist() == [[0, 2]]
