import numpy as np
import pytest
import torch

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.dynamic_llm import (
    casing_surfaces,
    decode_byte_pieces,
    QwenContinuationScorer,
    register_qwen_scorer,
)


def test_casing_and_bytes():
    assert decode_byte_pieces("<0xC2><0xA2>37") == "¢37"
    assert decode_byte_pieces("<0xC2>") == "�"
    assert casing_surfaces(" don't", ("original", "lower", "capitalized", "upper")) == [" don't", " Don't", " DON'T"]
    assert casing_surfaces(" <0xC2><0xA2>37", ("original", "upper")) == [" ¢37"]


def test_continuation_positions_padding_and_order():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(1)
    backend = object.__new__(QwenContinuationScorer)
    backend.device, backend.batch_size, backend.end = "cpu", 2, [2]
    backend.model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
        )
    ).eval()
    backend.reset()
    prefixes = [[1, 4, 6], [1], [1, 7]]
    continuations = [[8, 9, 2], [5, 2], []]
    actual = backend.score_continuations(prefixes, continuations)
    for p, c, scores in zip(prefixes, continuations, actual):
        assert len(scores) == len(c)
        if not c:
            continue
        ids = torch.tensor([p + c])
        with torch.inference_mode():
            logits = backend.model(ids, use_cache=False).logits[0]
            expected = torch.nn.functional.cross_entropy(
                logits[len(p) - 1 : len(p) + len(c) - 1], torch.tensor(c), reduction="none"
            )
        assert np.allclose(scores, expected.numpy(), atol=1e-5)


def test_native_zero_word_penalty(tmp_path, monkeypatch, rasr_dynamic_lib_dir):
    monkeypatch.syspath_prepend(str(rasr_dynamic_lib_dir))
    import librasr

    if not hasattr(librasr, "LlmScorer"):
        pytest.fail("Requested RASR binary predates dynamic LLM interface")
    register_qwen_scorer(librasr, zero=True)
    vocab = tmp_path / "pieces.txt"
    vocab.write_text("[BLANK]\n▁he\nllo\n▁world\n")
    cfg = librasr.Configuration()
    cfg.set("lib-rasr.lexicon.file", f"vocab-text:{vocab}")
    cfg.set("lib-rasr.label-scorer.type", "no-op")
    for k, v in {
        "type": "llm-timesync-beam-search",
        "max-beam-size": "1",
        "pre-llm-max-beam-size": "1",
        "blank-label-index": "0",
        "collapse-repeated-labels": "true",
        "llm-scale": "0",
        "word-penalty": "-1",
        "llm.type": "qwen-dynamic-zero",
        "log-stepwise-statistics": "false",
    }.items():
        cfg.set("lib-rasr.search-algorithm." + k, v)
    for channel in ("log", "error", "warning", "statistics"):
        cfg.set(f"*.{channel}.channel", str(tmp_path / "rasr.log"))
    librasr.init_logging(cfg)
    search = librasr.SearchAlgorithm(config=cfg)
    path = [0, 1, 1, 0, 2, 2, 0, 3, 3, 0]
    costs = np.full((len(path), 4), 1e6, dtype=np.float32)
    costs[np.arange(len(path)), path] = 0
    trace = search.recognize_segment(costs)
    assert [x.lemma for x in trace if x.lemma != "[BLANK]"] == ["▁he", "llo", "▁world"]
    assert abs(trace[-1].am_score + trace[-1].lm_score + 2) < 0.001
