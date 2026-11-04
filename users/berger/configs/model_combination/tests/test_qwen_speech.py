"""Offline model/API tests for audio-conditioned Speech-LLM fusion."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_speech import QwenSpeechScorer


def test_audio_embeddings_and_variable_length_teacher_forcing():
    from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3Config
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
    scorer = QwenSpeechScorer.__new__(QwenSpeechScorer)
    scorer.model, scorer.device, scorer.batch_size = model, "cpu", 2
    scorer.end, scorer.eos_ids, scorer.initial = [3], [3], []
    scorer.tokenizer = SimpleNamespace(encode=lambda *a, **kw: [4, 5, 6])
    features, mask = torch.randn(1, 16, 100), torch.zeros(1, 100, dtype=torch.long)
    mask[:, :40] = 1
    with torch.inference_mode():
        audio = model.get_audio_features(features, mask, return_dict=True).pooler_output
    prompt = torch.tensor([[1] + [2] * audio.shape[0] + [7]])
    inputs = dict(
        input_ids=prompt, attention_mask=torch.ones_like(prompt), input_features=features, input_features_mask=mask
    )
    scorer.set_audio(inputs)
    assert scorer.validate()["passed"]
    before = scorer.score_continuations([scorer.initial], [[4, 5]])[0]
    old_embeddings = scorer.prompt_embeddings.clone()
    scorer.reset()
    torch.testing.assert_close(scorer.prompt_embeddings, old_embeddings)
    scorer.set_audio(dict(inputs, input_features=features + 2))
    after = scorer.score_continuations([scorer.initial], [[4, 5]])[0]
    assert not np.allclose(before, after, atol=1e-9, rtol=1e-9)
    with pytest.raises(ValueError, match="current audio prompt"):
        scorer.score_continuations([[9]], [[4]])
    with pytest.raises(ValueError, match="Mismatched"):
        scorer.score_continuations([scorer.initial], [])


def test_native_accepts_partial_and_either_eos():
    scorer = QwenSpeechScorer.__new__(QwenSpeechScorer)
    scorer.eos_ids = [3, 8]
    scorer.processor = SimpleNamespace(decode=lambda ids, **kwargs: "hello")
    for tokens, truncated in [([4, 5], True), ([4, 3], False), ([4, 8], False)]:
        scorer.model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[1, 2] + tokens]))
        result = scorer.native(dict(input_ids=torch.tensor([[1, 2]])), max_new_tokens=2)
        assert result["raw_token_ids"] == tokens
        assert result["truncated"] == truncated
