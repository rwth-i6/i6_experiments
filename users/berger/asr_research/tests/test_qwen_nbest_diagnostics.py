from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_nbest_diagnostics import (
    surfaces,
    fit_examples,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.hf_scoring import normalize_hf_english


def test_surfaces_preserve_scoring_words_and_pronoun():
    variants = surfaces("I FOUND IT BUT I DIDN'T KEEP IT")
    assert "but I" in variants["pronoun-period"]
    assert ", but" in variants["clause-period"]
    assert len({normalize_hf_english(s) for s in variants.values()}) == 1


def test_sweep_can_remove_eos_without_averaging_tokens():
    rows = [
        dict(
            id="a",
            candidates=[
                dict(text="wrong", am_cost=0, lm_cost=0, qwen={"p": dict(nll=5, token_costs=[4, 1])}),
                dict(text="right", am_cost=0, lm_cost=0, qwen={"p": dict(nll=11, token_costs=[1, 10])}),
            ],
        )
    ]
    targets = {"a": dict(candidate_errors=[1, 0], oracle_errors=0, oracle_text="right")}
    report = fit_examples(rows, targets)
    assert report["oracle_guided"]
    assert report["best"][0]["excess_errors"] == 0
    assert report["best"][0]["eos_weight"] == 0
