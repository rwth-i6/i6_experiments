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


def test_selection_is_cost_minimization_not_token_average():
    candidates = [candidate("wrong", 1, 20), candidate("right", 2, 1)]
    assert select_candidate(candidates, 50, 0, "lowercase")["text"] == "wrong"
    assert select_candidate(candidates, 50, 0.1, "lowercase")["text"] == "right"
    assert select_candidate(candidates, 1, 0.1, "lowercase")["text"] == "wrong"


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


def test_driver_uses_actual_dev_split_and_open_candidates():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.nbest_qwen import run

    conditions = run()
    assert set(conditions) == {"open-no-lm", "open-4gram"}
    for evaluation in conditions.values():
        scoring = evaluation.candidates.creator
        generation = scoring.candidates.creator
        assert generation.binary.creator.source_folder.creator.commit == "ff206f90c64fa5c44fa1acb7059f23722bfefc8c"
        assert generation.binary.creator.source_folder.creator.patches
        assert generation.n == 50
        assert generation.dataset.creator.dataset.split == "validation"
        assert generation.dataset.creator.dataset.config == "other"
        assert evaluation.references.creator == generation.dataset.creator
        cfg = generation.rasr_config.creator.config.lib_rasr
        assert cfg.search_algorithm.unknown_word_fallback == "known-excluding"
        assert cfg.search_algorithm.recombination_mode == "on"
        assert cfg.search_algorithm.max_word_end_beam_size == 256
        validation = cfg.lexicon.file.creator
        assert validation.decoder_smoke_report.creator.rasr_binary_path == generation.binary
