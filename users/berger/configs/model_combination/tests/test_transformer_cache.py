"""Compare real word-LM cache transitions with independent full-prefix forwards."""

import pytest
import torch

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.transformer_lm.pytorch_modules import (
    PositionalEncodingConfig,
    TransformerBlockConfig,
    TransformerLinearConfig,
    TransformerLm,
    TransformerLmConfig,
    TransformerLmOnnxWrapper,
    TransformerMHSAConfig,
)


@pytest.mark.parametrize("split", [1, 3])
def test_cached_suffix_matches_full_history_and_last_projection(split):
    torch.manual_seed(34)
    torch.set_num_threads(2)
    cfg = TransformerLmConfig(
        embed_dim=8,
        hid_dim=8,
        vocab_dim=11,
        num_layers=2,
        dropout=0.0,
        pos_enc_config=PositionalEncodingConfig(embed_dim=8, max_len=64, dropout=0.0),
        block_config=TransformerBlockConfig(
            linear_config=TransformerLinearConfig(input_dim=8, ff_dim=16, output_dim=8, dropout=0.0),
            mhsa_config=TransformerMHSAConfig(input_dim=8, num_heads=2, dropout=0.0),
        ),
    )
    full = TransformerLm(step=0, epoch=1, cfg=cfg).eval()
    cached = TransformerLmOnnxWrapper(step=0, epoch=1, cfg=cfg).eval()
    cached.load_state_dict(full.state_dict())
    tokens = torch.tensor([[0, 7, 2, 4, 1], [0, 3, 8, 1, 6]])
    lengths = torch.tensor([5, 4])
    mask = torch.arange(5)[None, :] < lengths[:, None]
    empty_cache = [torch.zeros(2, 0, 8) for _ in range(4)]
    with torch.inference_mode():
        expected = -full(tokens, mask).log_softmax(-1)
        prefix = cached(tokens[:, :split], torch.full((2,), split), torch.zeros(2, dtype=torch.long), *empty_cache)
        suffix = cached(tokens[:, split:], lengths - split, torch.full((2,), split), *prefix[1:])
        for b, length in enumerate(lengths.tolist()):
            torch.testing.assert_close(prefix[0][b], expected[b, :split], atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(suffix[0][b, : length - split], expected[b, split:length], atol=2e-6, rtol=1e-5)
        projected = -full(tokens, mask, last_positions=lengths - 1).log_softmax(-1)
        torch.testing.assert_close(projected[:, 0], expected[torch.arange(2), lengths - 1])
    assert expected.std() > 0.01  # uniform fake outputs cannot establish cache parity
