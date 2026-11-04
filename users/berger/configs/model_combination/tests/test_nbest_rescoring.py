import json

from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.nbest_rescoring import (
    lm_surface,
    select_candidate,
    EvaluateQwenNbestJob,
)


def candidate(text, first_pass, nll):
    return dict(
        text=text,
        first_pass_cost=first_pass,
        qwen={p: dict(nll=nll) for p in ("lowercase", "initial-capital", "initial-capital-period")},
    )


def test_surface_policy_does_not_change_words():
    assert lm_surface("don't tell NASA", "lowercase") == "don't tell nasa"
    assert lm_surface("don't tell NASA", "initial-capital-period") == "Don't tell nasa."
    assert lm_surface('"hello!"', "initial-capital") == '"Hello!"'
    assert lm_surface("", "initial-capital-period") == ""


def test_evaluation_reports_first_pass_and_oracle(tmp_path):
    ref = tmp_path / "refs.jsonl"
    data = tmp_path / "candidates.jsonl"
    ref.write_text(json.dumps(dict(id="a", group_id="a", chunk_index=0, reference="right word")) + "\n")
    data.write_text(
        json.dumps(dict(id="a", candidates=[candidate("wrong word", 1, 20), candidate("right word", 2, 1)])) + "\n"
    )
    job = EvaluateQwenNbestJob(tk.Path(str(ref)), tk.Path(str(data)), code_version="test")
    job.out_summary = tk.Path(str(tmp_path / "summary.json"))
    job.out_hypotheses = tk.Path(str(tmp_path / "hypotheses"))
    job.run()
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["n50_lowercase_scale0"]["wer_percent"] == 50
    assert result["n50_lowercase_scale0.1"]["wer_percent"] == 0
    assert result["n50_oracle_scale0"]["wer_percent"] == 0
    assert result["n1_oracle_scale0"]["wer_percent"] == 50


def test_rescoring_uses_summed_cost_and_preserves_first_pass_and_nbest_cutoff():
    short = candidate("short", 1.0, 3.0)
    long = candidate("a much longer sentence", 2.0, 4.0)
    # Averaging by LLM-token count would incorrectly prefer the long candidate.
    for c, costs in ((short, [3.0]), (long, [1.0, 1.0, 1.0, 1.0])):
        for scores in c["qwen"].values():
            scores["token_costs"] = costs
    assert select_candidate([short, long], 2, 1.0, "lowercase") is short
    long["first_pass_cost"] = -1.0
    assert select_candidate([short, long], 2, 1.0, "lowercase") is long
    assert select_candidate([short, long], 1, 1.0, "lowercase") is short
