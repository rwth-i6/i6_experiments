import pytest


def test_earnings_diagnostic_cannot_leak_to_other_corpora():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.rnnt_qwen import run

    with pytest.raises(ValueError, match="Earnings22"):
        run(data={"librispeech-test-other": None}, tuning_only=False, earnings_pruning=True)
