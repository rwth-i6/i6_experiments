"""Next Qwen round: bounded dev grids, candidate reuse and frozen evaluation."""

import json
import pytest
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import qwen_tuning, rnnt_qwen, qwen


@pytest.fixture(scope="module")
def development():
    return qwen_tuning.run()


def test_two_development_corpora_and_no_nested_short_subsets(development):
    for result in development:
        assert set(result["data"]) == {"librispeech-dev-other", "loquacious-dev.short"}
        for key, conditions in result["scores"].items():
            expected = {"loquacious-dev.short"} if "corpus-lexicon" in key else set(result["data"])
            assert set(conditions) == expected


def test_tuning_rejects_heldout_and_large_model_evaluation():
    with pytest.raises(ValueError):
        qwen.run(tuning=True, data={"librispeech-test-other": None})
    with pytest.raises(ValueError):
        rnnt_qwen.run(tuning_only=True, data={"librispeech-test-other": None}, text_size="1.7b")


def test_rescoring_scale_selection(tmp_path):
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_nbest_selection import (
        SelectQwenScaleJob,
    )

    candidates = []
    for text, am, nll in (("a", 0.0, 10.0), ("b", 2.0, 1.0)):
        candidates.append(
            dict(
                text=text,
                am_cost=am,
                lm_cost=0,
                qwen={"initial-capital-period": dict(nll=nll, token_costs=[nll - 0.5, 0.5])},
            )
        )
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(dict(id="x", candidates=candidates)) + "\n")
    for scale, expected in ((0.1, "a"), (0.5, "b")):
        job = SelectQwenScaleJob(tk.Path(str(path)), scale, code_version="test")
        job.out_hypotheses = tk.Path(str(tmp_path / f"hyp{scale}.jsonl"))
        job.run()
        assert json.loads((tmp_path / f"hyp{scale}.jsonl").read_text())["text"] == expected
