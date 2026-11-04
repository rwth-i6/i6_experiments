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


def test_size_sweeps_and_shared_nbest_generation(development):
    small, large = development[1], development[3]
    for result, size in ((small, "0.6B"), (large, "1.7B")):
        for condition in ("one-pass", "one-pass-scale-0.1", "one-pass-scale-0.5"):
            job = result["scores"][f"granite-ctc/{condition}"]["librispeech-dev-other"].creator.hypotheses.creator
            cfg = job.rasr_config.creator.config.lib_rasr.search_algorithm
            assert cfg.llm.model.creator.repo_id == f"Qwen/Qwen3-{size}-Base"
            assert cfg.max_beam_size == [32] and cfg.pre_llm_max_beam_size == 128
            assert cfg.log_stepwise_statistics is False
        for resource in ("librispeech", "corpus"):
            for suffix in ("", "-lm0.1", "-lm0.5"):
                key = f"granite-ctc/two-pass-{resource}-lexicon{suffix}"
                assert "loquacious-dev.short" in result["scores"][key]
    key = "granite-ctc/two-pass-librispeech-lexicon"
    def scoring(result):
        return result["scores"][key]["librispeech-dev-other"].creator.hypotheses.creator.candidates.creator
    assert scoring(small).candidates == scoring(large).candidates
    assert scoring(small).repo != scoring(large).repo
    selected = large["scores"][key + "-lm0.1"]["librispeech-dev-other"].creator.hypotheses.creator
    assert selected.candidates == scoring(large).out_scores


def test_focused_rnnt_budget_and_model_sizes(development):
    for result, size in ((development[2], "0.6B"), (development[4], "1.7B")):
        for key, conditions in result["scores"].items():
            for summary in conditions.values():
                job = summary.creator.hypotheses.creator
                if not hasattr(job, "kind"):
                    continue
                cfg = job.config.creator.config.lib_rasr.search_algorithm
                if job.kind == "speech":
                    assert size == "0.6B"
                    assert cfg.max_beam_size in ([8], [16])
                    assert cfg.pre_llm_max_beam_size == 32
                    assert cfg.score_threshold == [3.0]
                elif job.kind == "text":
                    assert job.auxiliary_repo.creator.repo_id == f"Qwen/Qwen3-{size}-Base"
                    assert cfg.score_threshold == [2.0]


def test_tuning_rejects_heldout_and_large_model_evaluation():
    with pytest.raises(ValueError):
        qwen.run(tuning=True, data={"librispeech-test-other": None})
    with pytest.raises(ValueError):
        rnnt_qwen.run(tuning_only=True, data={"librispeech-test-other": None}, text_size="1.7b")


def test_rescoring_scale_selection(tmp_path):
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_nbest_selection import SelectQwenScaleJob
    candidates = []
    for text, am, nll in (("a", 0.0, 10.0), ("b", 2.0, 1.0)):
        candidates.append(dict(text=text, am_cost=am, lm_cost=0,
                               qwen={"initial-capital-period": dict(nll=nll, token_costs=[nll - 0.5, 0.5])}))
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(dict(id="x", candidates=candidates)) + "\n")
    for scale, expected in ((0.1, "a"), (0.5, "b")):
        job = SelectQwenScaleJob(tk.Path(str(path)), scale, code_version="test")
        job.out_hypotheses = tk.Path(str(tmp_path / f"hyp{scale}.jsonl"))
        job.run()
        assert json.loads((tmp_path / f"hyp{scale}.jsonl").read_text())["text"] == expected
