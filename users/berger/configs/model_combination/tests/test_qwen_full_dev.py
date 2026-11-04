from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_full_dev import (
    settings,
    selection_indices,
)


def test_cached_selection_weights_eos_and_lm_separately():
    candidates = [
        dict(text="wrong", am_cost=0, lm_cost=0, qwen={"p": dict(nll=5, token_costs=[4, 1])}),
        dict(text="right", am_cost=0, lm_cost=10, qwen={"p": dict(nll=11, token_costs=[1, 10])}),
    ]
    config = dict(policy="p", scale=1, retain_first_pass_lm=0, eos_weight=0, word_penalty=0, token_penalty=0)
    assert selection_indices(candidates, [config])[0] == 1
    assert selection_indices(candidates, [dict(config, eos_weight=1)])[0] == 0
    assert selection_indices(candidates, [dict(config, retain_first_pass_lm=1)])[0] == 0


def test_sample_setting_is_in_full_dev_grid_after_cost_rescaling():
    assert (
        dict(policy="lowercase", scale=1, retain_first_pass_lm=0, eos_weight=1, word_penalty=-2, token_penalty=-1)
        in settings()
    )
