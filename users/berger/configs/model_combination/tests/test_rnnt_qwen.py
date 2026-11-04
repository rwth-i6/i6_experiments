"""Offline checks for the pinned RNN-T dynamic LLM experiment setup."""

from types import SimpleNamespace

import pytest
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.rnnt_llm import RnntRawPiecesJob


def test_request_based_llm_adapter():
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.rnnt_llm import (
        register_rnnt_llm_scorer,
        ZeroRnntLlmBackend,
    )

    registered = {}

    class Base:
        def __init__(self, config):
            pass

    librasr = SimpleNamespace(LlmScorer=Base, register_llm_scorer_type=lambda name, cls: registered.update({name: cls}))
    cls = register_rnnt_llm_scorer(librasr, "zero", lambda config: ZeroRnntLlmBackend())
    scorer = cls({})
    assert registered["zero"] is cls
    assert scorer.initial_tokens() == [0]
    assert scorer.tokenize(["hello", " world"]) == [[1], [1]]
    assert scorer.spelling_variants(["hello", "WORLD", "123"]) == [["hello", "Hello"], ["world", "World"], ["123"]]
    requests = [SimpleNamespace(prefix=[0], tokens=[1, 2]), SimpleNamespace(prefix=[0, 1], tokens=[3])]
    assert scorer.score(requests) == [[0, 0], [0]]
    received = []
    scorer.backend.score_continuations = lambda p, c: received.append((p, c)) or [[2, 3], [4]]
    assert scorer.score(requests) == [[2, 3], [4]]
    assert received == [([[0], [0, 1]], [[1, 2], [3]])]
    scorer.backend.tokenize = lambda texts: [[[1], [2]] for _ in texts]
    with pytest.raises(ValueError, match="exactly one"):
        scorer.tokenize(["hello"])


def test_granite_text_and_speech_preserve_hypothesized_casing():
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.dynamic_qwen_tuned import (
        TunedQwenScorer,
    )
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_speech import QwenSpeechScorer

    tokenizer = SimpleNamespace(
        encode=lambda s, **kw: list(s.encode()),
        decode=lambda ids, **kw: bytes(ids).decode(),
        all_special_ids=[],
    )
    text, speech = TunedQwenScorer.__new__(TunedQwenScorer), QwenSpeechScorer.__new__(QwenSpeechScorer)
    for scorer in (text, speech):
        scorer.tokenizer, scorer.formatting = tokenizer, "original"
    inputs = ["HELLO", " WORLD", " Commission's", " Europe"]
    assert text.tokenize(inputs) == speech.tokenize(inputs)
    assert [tokenizer.decode(row[0]) for row in speech.tokenize(inputs)] == [
        "HELLO",
        " WORLD",
        " Commission's",
        " Europe",
    ]
    assert all(len(row) == 1 for row in speech.tokenize(inputs))


def test_raw_inventory_keeps_final_blank_and_model_order(monkeypatch, tmp_path):
    import transformers

    labels = ["▁hello", "▁world", "▁thank", "▁you", "▁commission's", "▁proposal", "▁European", "▁Union", "<blank>"]
    tokenizer = SimpleNamespace(
        convert_ids_to_tokens=lambda ids: [labels[i] for i in ids],
        encode=lambda text, **kw: [labels.index("▁" + word) for word in text.split()],
        decode=lambda ids, **kw: " ".join(labels[i][1:] for i in ids),
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a: SimpleNamespace(vocab_size=9, blank_token_id=8)
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a: tokenizer)
    job = RnntRawPiecesJob(tk.Path("repo"), code_version="test")
    job.out_vocab, job.out_report = tk.Path(str(tmp_path / "pieces.txt")), tk.Path(str(tmp_path / "report.json"))
    job.out_blank = SimpleNamespace(set=lambda value: setattr(job, "blank", value))
    job.run()
    assert job.blank == 8
    assert (tmp_path / "pieces.txt").read_text().splitlines() == labels[:-1] + ["[BLANK]"]


def test_subset_selects_audio_before_recognition(tmp_path):
    import gzip
    import json
    from datasets import Dataset
    from i6_experiments.example_setups.seq2seq_rasr_2025.data.benchmark_subset import PrepareBenchmarkSubsetJob

    Dataset.from_dict({"id": ["a", "b", "c"], "text": ["one", "two", "three"]}).save_to_disk(str(tmp_path / "source"))
    references = tmp_path / "refs.jsonl"
    references.write_text("".join(json.dumps(dict(id=seq_id, reference=seq_id)) + "\n" for seq_id in "abc"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(excluded_empty_reference_ids=["empty"])))
    segments = tmp_path / "segments.gz"
    with gzip.open(segments, "wt") as dst:
        dst.write("dev/b/b\ndev/empty/empty\n")
    job = PrepareBenchmarkSubsetJob(
        tk.Path(str(tmp_path / "source")),
        tk.Path(str(references)),
        tk.Path(str(manifest)),
        tk.Path(str(segments)),
        code_version="test",
    )
    job.out_dataset = tk.Path(str(tmp_path / "selected"))
    job.out_references, job.out_manifest = (
        tk.Path(str(tmp_path / "out.jsonl")),
        tk.Path(str(tmp_path / "out-manifest.json")),
    )
    job.run()
    from datasets import load_from_disk

    assert list(load_from_disk(job.out_dataset.get_path())["id"]) == ["b"]
    assert json.loads((tmp_path / "out.jsonl").read_text())["id"] == "b"
