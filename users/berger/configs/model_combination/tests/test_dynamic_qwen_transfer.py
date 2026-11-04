import pytest

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.dynamic_qwen_tuned import (
    reward_costs,
    TunedQwenScorer,
)


def test_token_reward_is_outside_lm_scale_and_can_be_negative():
    for scale in (0.2, 0.4):
        actual = reward_costs([1, 0.1, 7], [3, 4, 2], reward=0.5, scale=scale, excluded={2})
        assert sum(actual) * scale == pytest.approx(8.1 * scale - 1)
        assert actual[1] < 0
    with pytest.raises(ValueError):
        reward_costs([], [], reward=0.5, scale=0, excluded={2})


def test_initial_capital_changes_only_first_word():
    class Tokenizer:
        all_special_ids = []

        def encode(self, text, **kwargs):
            return list(map(ord, text))

        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))

    backend = object.__new__(TunedQwenScorer)
    backend.formatting, backend.tokenizer = "initial-capital", Tokenizer()
    words = backend.tokenize(["HELLO", " WORLD", " <0xC2><0xA2>37"])
    assert [backend.tokenizer.decode(v[0]) for v in words] == ["Hello", " world", " ¢37"]
