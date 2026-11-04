"""Dynamic BPE word scoring: token ids, teacher forcing and development-only wiring."""

from types import SimpleNamespace

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.wordwise_transformer import (
    BpeWordwiseGateJob,
    WordwiseTransformerBackend,
    read_index_vocabulary,
)


def test_vocabulary_aliases_and_dense_indices(tmp_path):
    word = tmp_path / "word.vocab"
    word.write_text("<s> 0\n</s> 0\n<UNK> 1\nHELLO 2\nWORLD 3\n")
    assert read_index_vocabulary(word)["WORLD"] == 3
    bpe = tmp_path / "bpe.vocab"
    bpe.write_text(repr({"<s>": 0, "</s>": 0, "<unk>": 1, "HE@@": 2, "LLO": 3}))
    assert read_index_vocabulary(bpe, bpe=True)["HE@@"] == 2
    word.write_text("</s> 0\n<UNK> 1\nHELLO 3\n")
    with pytest.raises(ValueError, match="dense"):
        read_index_vocabulary(word)


def test_word_and_bpe_retokenization_are_independent_of_am_ids():
    word = WordwiseTransformerBackend.__new__(WordwiseTransformerBackend)
    word.kind, word.vocab, word.unknown = "word", {"HELLO": 2, "WORLD": 3}, 1
    assert word.tokenize(["hello", " WORLD", "UNSEEN"]) == [[2], [3], [1]]
    with pytest.raises(ValueError, match="one completed"):
        word.tokenize(["HELLO WORLD"])
    bpe = WordwiseTransformerBackend.__new__(WordwiseTransformerBackend)
    bpe.kind, bpe.vocab, bpe.unknown = "bpe", {"HE@@": 4, "LLO": 2}, 1
    surfaces = []
    bpe.encoder = SimpleNamespace(segment_sentence=lambda s: surfaces.append(s) or ["HE@@", "LLO"])
    assert bpe.tokenize([" hello"]) == [[4, 2]]
    assert surfaces == ["HELLO"]


@pytest.mark.parametrize("batch_size", [1, 8])
def test_teacher_forcing_and_eos_accounting(batch_size):
    backend = WordwiseTransformerBackend.__new__(WordwiseTransformerBackend)
    backend.vocab, backend.checked, backend.batch_size = {"</s>": 0, "<unk>": 1, "A": 2, "B": 3}, False, batch_size
    seen = []

    def forward(outputs, inputs):
        assert outputs == ["scores"]
        tokens, lengths = inputs["tokens"], inputs["tokens:size1"]
        assert tokens.dtype == lengths.dtype == np.int32
        costs = []
        for row, length in zip(tokens, lengths):
            prefix = row[:length].tolist()
            seen.append(prefix)
            probabilities = np.roll(np.array([0.1, 0.2, 0.3, 0.4]), prefix[-1])
            costs.append(-np.log(probabilities))
        return [np.array(costs)]

    backend.session = SimpleNamespace(run=forward)
    requests = [SimpleNamespace(prefix=[0], tokens=[2, 3]), SimpleNamespace(prefix=[0, 2, 3], tokens=[0])]
    costs = backend.score(requests)
    assert seen == [[0], [0, 2], [0, 2, 3]]
    assert costs[0] == pytest.approx([-np.log(0.3), -np.log(0.2)])
    assert costs[1] == pytest.approx([-np.log(0.2)])
    assert backend.checked
    assert backend.initial_tokens() == backend.sentence_end_tokens() == [0]


def test_ctc_rna_native_gate(tmp_path, monkeypatch, rasr_bpe_lib_dir):
    binary = rasr_bpe_lib_dir
    job = BpeWordwiseGateJob(tk.Path(str(binary)), code_version="test")
    job.out_report, job.out_log = tk.Path(str(tmp_path / "validation.json")), tk.Path(str(tmp_path / "rasr.log"))
    monkeypatch.chdir(tmp_path)
    job.run()
    assert '"passed": true' in (tmp_path / "validation.json").read_text()


@pytest.mark.parametrize("scale", [0.4, 0.8])
def test_unknown_word_penalty_is_outside_lm_scale(monkeypatch, scale):
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.wordwise_unknown import (
        UnknownPenaltyWordwiseBackend,
    )

    def init(self, config):
        self.kind, self.unknown = "word", 1
        self.session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])

    monkeypatch.setattr(WordwiseTransformerBackend, "__init__", init)
    monkeypatch.setattr(WordwiseTransformerBackend, "score", lambda self, requests: [[3.0, 5.0, 7.0], [2.0]])
    backend = UnknownPenaltyWordwiseBackend({"unknown-word-penalty": 4.0, "outer-lm-scale": scale})
    requests = [SimpleNamespace(prefix=[0], tokens=[2, 1, 0]), SimpleNamespace(prefix=[0, 1], tokens=[1])]
    costs = backend.score(requests)
    # Known token and EOS unchanged. Every UNK occurrence pays beta once,
    # including after an earlier UNK; the history itself is not penalized.
    assert costs[0] == pytest.approx([3.0, 5.0 + 4.0 / scale, 7.0])
    assert costs[1] == pytest.approx([2.0 + 4.0 / scale])
    assert scale * costs[0][1] - scale * 5.0 == pytest.approx(4.0)
