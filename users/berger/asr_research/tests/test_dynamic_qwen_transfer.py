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


def test_transfer_grid_preserves_original_jobs():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.dynamic_qwen import (
        run,
        run_rescoring_transfer,
    )

    controls = run()
    control = controls["lm0.3_wordpen0_beam16_pre64_cases"].hypotheses.creator
    runs = run_rescoring_transfer()
    assert len(runs) == 18
    for score in runs.values():
        job = score.hypotheses.creator
        assert job.binary == control.binary
        assert job.gate == control.gate
        cfg = job.rasr_config.creator.config.lib_rasr.search_algorithm
        assert cfg.llm.outer_lm_scale == cfg.llm_scale
        assert cfg.llm.type == "qwen-dynamic-tuned"
        assert cfg.max_beam_size == [16]
    cfg = control.rasr_config.creator.config.lib_rasr.search_algorithm
    assert cfg.llm.type == "qwen-dynamic"


def test_beam_grid_reuses_winner_and_only_changes_beam_caps():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.dynamic_qwen import (
        run_rescoring_transfer,
        run_beam_tuning,
    )

    selected = run_rescoring_transfer()["lm0.3_tokreward0.5_wordpen-1_initial-capital_period0_beam16_pre64"]
    original = selected.hypotheses.creator
    runs = run_beam_tuning()
    assert len(runs) == 10
    assert sum(score == selected for score in runs.values()) == 1
    pairs = set()
    for score in runs.values():
        job = score.hypotheses.creator
        cfg = job.rasr_config.creator.config.lib_rasr.search_algorithm
        pairs.add((cfg.max_beam_size[0], cfg.pre_llm_max_beam_size))
        assert cfg.score_threshold == [14.0]
        assert cfg.pre_llm_score_threshold == 14.0
        assert cfg.llm_scale == 0.3
        assert cfg.word_penalty == -1.0
        assert cfg.llm.token_reward == 0.5
        assert cfg.llm.formatting == "initial-capital"
        assert cfg.llm.terminal_period is False
        assert job.code_version == original.code_version
        assert job.binary == original.binary
        assert job.dataset == original.dataset
        assert job.gate == original.gate
    assert pairs == {(b, p) for b in (16, 32, 64) for p in (64, 128, 256)} | {(128, 256)}


def test_threshold_grid_establishes_thresholds_before_tightening_caps():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.dynamic_qwen import (
        run_threshold_tuning,
    )

    runs = run_threshold_tuning()
    assert len(runs) == 25
    thresholds = set()
    for score in runs.values():
        cfg = score.hypotheses.creator.rasr_config.creator.config.lib_rasr.search_algorithm
        assert cfg.max_beam_size == [128]
        assert cfg.pre_llm_max_beam_size == 256
        assert cfg.llm_scale == 0.3
        assert cfg.word_penalty == -1
        assert cfg.llm.token_reward == 0.5
        assert cfg.llm.formatting == "initial-capital"
        assert cfg.llm.terminal_period is False
        thresholds.add((cfg.score_threshold[0], cfg.pre_llm_score_threshold))
    assert thresholds == {(f, p) for f in (4, 6, 8, 10, 12) for p in (4, 6, 8, 10, 12)}


def test_tightened_grid_reuses_threshold_reference():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.dynamic_qwen import (
        run_threshold_tuning,
        run_tightened_beam_tuning,
    )

    reference = run_threshold_tuning()[
        "lm0.3_tokreward0.5_wordpen-1_initial-capital_period0_beam128_pre256_thr6_prethr6"
    ]
    runs = run_tightened_beam_tuning()
    assert len(runs) == 18
    assert runs["lm0.3_tokreward0.5_wordpen-1_initial-capital_period0_beam128_pre256_thr6_prethr6"] == reference
    pairs = set()
    for score in runs.values():
        job = score.hypotheses.creator
        cfg = job.rasr_config.creator.config.lib_rasr.search_algorithm
        pairs.add((cfg.max_beam_size[0], cfg.pre_llm_max_beam_size))
        assert cfg.score_threshold == [6.0]
        assert cfg.pre_llm_score_threshold == 6.0
        assert cfg.llm_scale == 0.3
        assert cfg.word_penalty == -1
        assert cfg.llm.token_reward == 0.5
        assert cfg.llm.formatting == "initial-capital"
        assert cfg.llm.terminal_period is False
        assert job.code_version == reference.hypotheses.creator.code_version
        assert job.binary == reference.hypotheses.creator.binary
    assert pairs == (
        {(128, 256)}
        | {(b, 256) for b in (1, 2, 4, 8, 16, 32, 64)}
        | {(128, p) for p in (4, 8, 16, 32, 64, 128)}
        | {(4, 16), (8, 32), (16, 64), (32, 128)}
    )
