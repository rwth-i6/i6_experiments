"""Reusable template contracts; no downloaded models or GPU required."""

import types
from pathlib import Path

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.qwen_asr import recog as qwen


def test_template_boundary():
    root = Path(qwen.__file__).parents[2]
    assert not (root / "experiments/open_asr").exists()
    assert not (root / "model_pipelines/speech_llm").exists()
    assert not list(root.rglob("*oov*.py"))
    for path in root.rglob("*.py"):
        text = path.read_text()
        assert "normalize_hf_score" not in text, path
        assert "/work/asr4/berger" not in text, path


def test_partial_output_and_formatting():
    assert qwen.format_for_corpus("Don't — stop!", uppercase=True, strip_punctuation=True) == "DON'T STOP"
    assert qwen.format_for_corpus("Don't — stop!") == "Don't — stop!"


def test_speech_scorer_preserves_history_and_scores_eos():
    registry = {}
    transitions = types.SimpleNamespace(INITIAL_LABEL=1, LABEL_TO_LABEL=2, LABEL_LOOP=3, SENTENCE_END=4)

    class Base:
        def __init__(self, config):
            pass

    backend = types.SimpleNamespace(score=lambda contexts: [([1.0, 2.0], len(c)) for c in contexts])
    rasr = types.SimpleNamespace(
        LabelScorer=Base,
        TransitionType=transitions,
        register_label_scorer_type=lambda name, cls: registry.update({name: cls}),
    )
    qwen.register_scorer(rasr, backend)
    scorer = registry["qwen-asr"](None)
    assert scorer.get_initial_scoring_context() == ()
    assert scorer.extended_scoring_context((3,), 4, transitions.LABEL_TO_LABEL) == (3, 4)
    assert scorer.extended_scoring_context((3,), 4, transitions.SENTENCE_END) == (3,)
    assert scorer.compute_scores_with_times([(), (3, 4)]) == [([1.0, 2.0], 0), ([1.0, 2.0], 2)]


def test_qwen_full_history_scores_use_audio_prompt_and_bounded_batches():
    import torch

    calls = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, inputs_embeds, attention_mask, use_cache):
            assert not use_cache
            assert inputs_embeds.shape[0] <= 2
            torch.testing.assert_close(inputs_embeds[:, :2], torch.ones_like(inputs_embeds[:, :2]))
            calls.append(inputs_embeds.shape)
            return types.SimpleNamespace(logits=torch.zeros((*inputs_embeds.shape[:2], 10)))

    backend = qwen.QwenAsrBackend.__new__(qwen.QwenAsrBackend)
    backend.model, backend.device, backend.batch_size = Model(), "cpu", 2
    backend.prompt_ids, backend.prompt_embeddings = [0, 1], torch.ones((1, 2, 3))
    scores = backend.score([(), (2,), (3,), (4,)])
    assert [t for _, t in scores] == [0, 1, 1, 1]
    assert len(calls) == 3
    for row, _ in scores:
        torch.testing.assert_close(torch.tensor(row), torch.full((10,), torch.log(torch.tensor(10.0)).item()))


def test_native_objects_are_not_pickled():
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog import OfflineRasrRecogForwardStep

    step = OfflineRasrRecogForwardStep.__new__(OfflineRasrRecogForwardStep)
    step._search_algorithm, step._aligner = object(), object()
    state = step.__getstate__()
    assert state["_search_algorithm"] is state["_aligner"] is None
    assert step._search_algorithm is not None
    restored = OfflineRasrRecogForwardStep.__new__(OfflineRasrRecogForwardStep)
    restored.__setstate__(state)
    assert restored._search_algorithm is restored._aligner is None


def test_last_hidden_projection_matches_full_sequence():
    import torch
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.transformer_lm.pytorch_modules import (
        TransformerLm,
    )

    # Exercise the production forward with a small deterministic network.
    model = TransformerLm.__new__(TransformerLm)
    torch.nn.Module.__init__(model)
    model.embed = torch.nn.Embedding(11, 8)
    model.positional_encoding = model.input_linear = torch.nn.Identity()
    model.transformer_blocks = torch.nn.ModuleList()
    model.output_layernorm = torch.nn.LayerNorm(8)
    model.dropout = torch.nn.Dropout(0.0)
    model.output_linear = torch.nn.Linear(8, 11)
    model.causal_mask = torch.ones(5, 5, dtype=torch.bool).tril()
    model.eval()
    tokens = torch.tensor([[0, 2, 3, 4, 5], [0, 6, 7, 0, 0]])
    lengths = torch.tensor([5, 3])
    mask = torch.arange(5)[None] < lengths[:, None]
    projected_shapes = []
    handle = model.output_linear.register_forward_pre_hook(lambda _, args: projected_shapes.append(args[0].shape))
    with torch.inference_mode():
        full = model(tokens, mask)
        last = model(tokens, mask, last_positions=lengths - 1)
    handle.remove()
    torch.testing.assert_close(last[:, 0], full[torch.arange(2), lengths - 1])
    assert projected_shapes == [torch.Size([5, 2, 8]), torch.Size([1, 2, 8])]
