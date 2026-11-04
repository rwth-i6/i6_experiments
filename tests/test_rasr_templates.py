"""Reusable template contracts; no downloaded models or GPU required."""

import types
from pathlib import Path

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.qwen_asr import recog as qwen


def test_template_imports_resolve_without_user_research():
    import ast
    from importlib.util import resolve_name

    root = Path(qwen.__file__).parents[2]
    prefix = "i6_experiments.example_setups.seq2seq_rasr_2025"
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).with_suffix("").parts
        package = prefix + ("." + ".".join(relative[:-1]) if relative[:-1] else "")
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                name = (
                    resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
                )
                assert not name.startswith("i6_experiments.users.berger"), (path, name)
                if name.startswith(prefix + "."):
                    target = root.joinpath(*name[len(prefix) + 1 :].split("."))
                    assert target.is_dir() or target.with_suffix(".py").is_file(), (path, name)


def test_qwen_scores_match_real_multimodal_full_forward():
    import numpy as np
    import torch
    from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3Config, BatchFeature
    from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig

    torch.manual_seed(12)
    torch.set_num_threads(2)
    model = Qwen3ASRForConditionalGeneration(
        Qwen3ASRConfig(
            audio_config=Qwen3ASREncoderConfig(
                num_mel_bins=16,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                d_model=16,
                output_dim=16,
                downsample_hidden_size=8,
            ),
            text_config=Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
            ),
            audio_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )
    ).eval()
    features, feature_mask = torch.randn(1, 16, 100), torch.ones(1, 100, dtype=torch.long)
    with torch.inference_mode():
        audio = model.get_audio_features(features, feature_mask, return_dict=True).pooler_output
    prompt = torch.tensor([[1] + [2] * audio.shape[0] + [7]])
    inputs = dict(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        input_features=features,
        input_features_mask=feature_mask,
    )
    backend = qwen.QwenAsrBackend.__new__(qwen.QwenAsrBackend)
    backend.model, backend.device, backend.batch_size = model, "cpu", 2
    backend.processor = types.SimpleNamespace(apply_transcription_request=lambda **kwargs: BatchFeature(inputs))
    backend.set_audio(np.zeros(160, dtype="float32"), 16000)
    prefixes = [(), (4,), (5, 6), (7,), (8,)]
    actual = backend.score(prefixes)
    with torch.inference_mode():
        for prefix, (costs, length) in zip(prefixes, actual):
            ids = torch.tensor([backend.prompt_ids + list(prefix)])
            logits = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                input_features=features,
                input_features_mask=feature_mask,
                use_cache=False,
            ).logits[0, -1]
            torch.testing.assert_close(torch.tensor(costs), -logits.float().log_softmax(-1), atol=2e-6, rtol=1e-5)
            assert length == len(prefix)
    assert not np.allclose(actual[1][0], actual[2][0])


def test_forward_step_pickle_roundtrip_rebuilds_lazily_in_fresh_worker(tmp_path):
    import os
    import pickle
    import subprocess
    import sys
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog import OfflineRasrRecogForwardStep

    class CannotPickle:
        def __reduce__(self):
            raise TypeError("native object cannot be pickled")

    step = OfflineRasrRecogForwardStep.__new__(OfflineRasrRecogForwardStep)
    step._search_algorithm = step._aligner = CannotPickle()
    step.sample_rate = 16000
    path = tmp_path / "step.pkl"
    path.write_bytes(pickle.dumps(step))
    script = """
import pickle,sys
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog import OfflineRasrRecogForwardStep
calls=[]
def build(self):
    calls.append(self.sample_rate)
    return object()
OfflineRasrRecogForwardStep.init_search_algorithm=build
with open(sys.argv[1],"rb") as src: step=pickle.load(src)
assert step._search_algorithm is step._aligner is None
assert calls == []
assert step.search_algorithm is step.search_algorithm
assert calls == [16000]
"""
    subprocess.run([sys.executable, "-c", script, str(path)], check=True, env=os.environ.copy(), timeout=30)


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
