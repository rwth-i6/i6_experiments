import json
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import pretrained_correction as pc


def test_prior_sign_inventory_and_frame_costs(tmp_path, monkeypatch):
    def init(self, *args, **kwargs):
        self.family, self.blank = "ctc", 1
        self.tokenizer = SimpleNamespace(convert_ids_to_tokens=lambda i: ["a", "blank", "b"][i])

    monkeypatch.setattr(pc.BaseAdapter, "__init__", init)
    raw = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    monkeypatch.setattr(pc.BaseAdapter, "encode_inputs", lambda self, inputs: raw.copy())
    prior = tmp_path / "prior.json"
    data = dict(log_prior=np.log([0.2, 0.7, 0.1]).tolist(), blank=1, pieces=["a", "blank", "b"])
    prior.write_text(json.dumps(data))
    adapter = pc.CorrectedAdapter("repo", "ctc", prior=tk.Path(str(prior)), prior_scale=0.2)
    np.testing.assert_allclose(adapter.encode_inputs({}), raw + 0.2 * np.log([0.2, 0.7, 0.1]), rtol=1e-6)
    assert adapter.validation["prior_shape_checked"]
    zero = pc.CorrectedAdapter("repo", "ctc")
    np.testing.assert_array_equal(zero.encode_inputs({}), raw)
    data["pieces"] = ["b", "blank", "a"]
    prior.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="indices"):
        pc.CorrectedAdapter("repo", "ctc", prior=tk.Path(str(prior)), prior_scale=0.2)


def test_zero_encoder_ilm_nonblank_normalization_and_frame_independence(monkeypatch):
    adapter = pc.CorrectedAdapter.__new__(pc.CorrectedAdapter)
    adapter.blank, adapter.device, adapter.family = 1, "cpu", "rnnt"
    adapter.ilm_scale, adapter.validation, adapter.ilm_cache = 0.2, {}, OrderedDict()
    adapter.encoder = torch.randn(1, 2, 4)
    adapter.model = SimpleNamespace(
        decoder=lambda ids, cache=None: torch.nn.functional.one_hot(ids, num_classes=4).float(),
        joint=lambda decoder_hidden_states, encoder_hidden_states: decoder_hidden_states + encoder_hidden_states,
    )
    costs = [3.0, 4.0, 5.0, 6.0]
    monkeypatch.setattr(
        pc.BaseAdapter, "score_contexts", lambda self, contexts: [(costs.copy(), t) for t, _ in contexts]
    )
    result = adapter.score_contexts([(0, (2,)), (1, (2,)), (0, ())])
    expected = torch.tensor([0.0, 1.0, 0.0]).log_softmax(0).numpy()
    np.testing.assert_allclose(np.asarray(result[0][0])[[0, 2, 3]], np.array(costs)[[0, 2, 3]] + 0.2 * expected)
    assert result[0][0][1] == costs[1]
    assert result[0][0] == result[1][0]
    assert result[0][0] != result[2][0]
    assert adapter.validation["blank_unchanged"]
    assert np.isclose(np.exp(adapter.ilm_cache[(2,)][[0, 2, 3]]).sum(), 1)
    adapter.ilm_scale = 0
    assert adapter.score_contexts([(0, ())])[0][0] == costs


def test_scoped_injection_restores_original_on_failure(monkeypatch):
    original = pc.hf_benchmark.HfBenchmarkAdapter
    with pytest.raises(RuntimeError):
        with pc.corrected_adapter():
            assert pc.hf_benchmark.HfBenchmarkAdapter is not original
            raise RuntimeError("test")
    assert pc.hf_benchmark.HfBenchmarkAdapter is original


@pytest.mark.parametrize("beam", [4, 8])
def test_native_speech_beam_with_real_tiny_multimodal_model(beam):
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
                eos_token_id=3,
                pad_token_id=0,
                bos_token_id=1,
            ),
            audio_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )
    ).eval()
    features, mask = torch.randn(1, 16, 100), torch.ones(1, 100, dtype=torch.long)
    with torch.inference_mode():
        audio = model.get_audio_features(features, mask, return_dict=True).pooler_output
    ids = torch.tensor([[1] + [2] * audio.shape[0] + [7]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), input_features=features, input_features_mask=mask)
    scorer = SimpleNamespace(model=model, processor=SimpleNamespace(decode=lambda *a, **kw: "text"), eos_ids=[3])
    result = pc.speech_beam_native(scorer, inputs, beam=beam, max_new_tokens=4)
    assert 0 < len(result["raw_token_ids"]) <= 4
    assert result["beam"] == beam
    assert result["truncated"] == (result["raw_token_ids"][-1] != 3)
