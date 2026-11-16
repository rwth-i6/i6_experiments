"""Guard scientific comparison semantics, not the current grid or tuned scales."""

import copy

import pytest

from i6_experiments.users.berger.model_pipelines.common.study_summary import render_tables, validate


def evidence():
    rows = []
    for model, lm, errors in [("Granite CTC", "None", 8), ("Speech Qwen 0.6B", "None", 6),
                              ("Speech Qwen 1.7B", "None", 4), ("Granite CTC", "Speech Qwen 1.7B", 5)]:
        family = f"{model} | {lm} | " + ("standalone" if lm == "None" else "one-pass")
        rows.append(dict(model=model, lm=lm, family=family, group="g", condition=family, dataset="example",
                         status="complete", normalizer="norm", wer=errors,
                         summary=dict(ins=0, sub=errors, del_=0, ref_len=100)))
        rows[-1]["summary"]["del"] = rows[-1]["summary"].pop("del_")
    rows[-1].update(baseline_indices=[0, 2], delta=1, delta_ci=[0.1, 2])
    protocol = dict(ready=True, selected={r["family"] + " | fast": dict(r) for r in rows})
    comparison = dict(protocol_sha256="frozen", comparisons={"fast/example": dict(dataset="example", rows=rows)})
    return comparison, protocol


def test_fusion_that_beats_am_but_loses_to_speech_is_a_regression():
    comparison, protocol = evidence()
    validate(comparison, protocol, "frozen")
    comparison["comparisons"]["fast/example"]["rows"][-1]["delta"] = -3
    with pytest.raises(ValueError, match="better standalone"):
        validate(comparison, protocol, "frozen")


def test_matching_speech_model_size_is_required():
    comparison, protocol = evidence()
    comparison["comparisons"]["fast/example"]["rows"][-1].update(baseline_indices=[0, 1], delta=-1)
    with pytest.raises(ValueError, match="constituents"):
        validate(comparison, protocol, "frozen")


def test_test_winner_cannot_replace_frozen_dev_condition():
    comparison, protocol = evidence()
    comparison["comparisons"]["fast/example"]["rows"][-1]["condition"] = "better-test-setting"
    with pytest.raises(ValueError, match="not frozen on dev"):
        validate(comparison, protocol, "frozen")


@pytest.mark.parametrize("problem", ["partial", "normalizer", "protocol", "counts"])
def test_invalid_evidence_is_rejected(problem):
    comparison, protocol = evidence()
    rows = comparison["comparisons"]["fast/example"]["rows"]
    if problem == "partial":
        rows.pop()
    elif problem == "normalizer":
        rows[0]["normalizer"] = "old"
    elif problem == "protocol":
        comparison["protocol_sha256"] = "other"
    else:
        rows[0]["summary"]["sub"] += 1
    with pytest.raises(ValueError):
        validate(comparison, protocol, "frozen")


def test_subsets_are_shown_separately_without_pooled_winner():
    comparison, _ = evidence()
    source = comparison["comparisons"].pop("fast/example")
    for dataset in ["loquacious-test", "loquacious-test.yodas"]:
        comparison["comparisons"]["fast/" + dataset] = copy.deepcopy(source)
        comparison["comparisons"]["fast/" + dataset]["dataset"] = dataset
    rendered = render_tables(comparison)
    main, subsets = rendered.split("## Loquacious source subsets")
    assert "| Loquacious |" in main and "| yodas |" not in main
    assert "| yodas |" in subsets and "| Loquacious |" not in subsets
    assert "**" not in rendered  # no held-out winner styling
