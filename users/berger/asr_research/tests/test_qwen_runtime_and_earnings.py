import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.runtime_budget import (
    AUDIO_SECONDS, apply_runtime_budget,
)


def test_duration_based_runtime_requests():
    assert AUDIO_SECONDS["gigaspeech-cleaned"] / 3600 == pytest.approx(35.14311)
    assert AUDIO_SECONDS["spgispeech"] / 3600 == pytest.approx(100.00208)
    for dataset, expected in (("gigaspeech-cleaned", 60), ("spgispeech", 168)):
        job = SimpleNamespace(rqmt=dict(time=48))
        assert apply_runtime_budget(job, dataset) is job
        assert job.rqmt["time"] == expected
        assert expected > AUDIO_SECONDS[dataset] / 3600 * 1.5
        assert "runtime_cap_hours" not in job.rqmt
    gpu_job = SimpleNamespace(rqmt=dict(time=363, gpu=1, gpu_mem=24, runtime_cap_hours=363))
    apply_runtime_budget(gpu_job, "spgispeech")
    assert gpu_job.rqmt == dict(time=168, gpu=1, gpu_mem=48)
    job = SimpleNamespace(rqmt=dict(time=48))
    apply_runtime_budget(job, "earnings22-cleaned")
    assert job.rqmt == dict(time=48)


def test_settings_enforces_cap_and_selects_48gb_queue():
    settings = Path(__file__).resolve().parents[3] / "settings.py"
    if not settings.exists():
        pytest.skip("Local Sisyphus settings unavailable")
    source = ast.parse(settings.read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "check_engine_limits")
    namespace = {"CPU_SLOW_JOBLIST": []}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(settings), "exec"), namespace)
    task = SimpleNamespace(_job=SimpleNamespace())
    check = namespace["check_engine_limits"]
    assert check(dict(time=363), task)["time"] == 168
    resources = check(dict(time=363, runtime_cap_hours=363), task)
    assert resources["time"] == 168
    assert "runtime_cap_hours" not in resources
    resources = check(dict(time=168, gpu=1, gpu_mem=48), task)
    assert resources["sbatch_args"] == ["-p", "gpu_48gb"]


def test_earnings_pruning_grid_and_unchanged_controls():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.earnings_qwen import run
    result = run()
    scores = result["scores"]
    assert len(scores) == 13  # native + 10 new pruning settings + 2 existing controls
    for key, rows in scores.items():
        assert set(rows) == {"earnings22-cleaned"}
        job = rows["earnings22-cleaned"].creator.hypotheses.creator
        if key.endswith("/native"):
            continue
        cfg = job.config.creator.config.lib_rasr.search_algorithm
        assert job.kind == "speech"
        assert cfg.llm_scale == 0.3 and cfg.word_penalty == 0
        assert cfg.max_beam_size[0] in {8, 16, 24, 32}
        assert cfg.pre_llm_max_beam_size in {32, 64, 128}
        assert cfg.score_threshold[0] in {3.0, 3.5, 4.0}
        assert cfg.pre_llm_score_threshold == cfg.score_threshold[0]
        assert cfg.log_stepwise_statistics is False
        assert job.rqmt["time"] == 48
    control = scores["parakeet-rnnt/speech-qwen"]["earnings22-cleaned"].creator.hypotheses.creator
    fast = scores["parakeet-rnnt/speech-qwen-fast"]["earnings22-cleaned"].creator.hypotheses.creator
    assert control.config.creator.config.lib_rasr.search_algorithm.max_beam_size == [32]
    assert control.config.creator.config.lib_rasr.search_algorithm.score_threshold == [4.0]
    assert fast.config.creator.config.lib_rasr.search_algorithm.max_beam_size == [8]
    # Wall-time resource changes must not invalidate recognition results.
    old_resources, old_id = dict(fast.rqmt), fast._sis_id()
    apply_runtime_budget(fast, "spgispeech")
    assert fast.rqmt["time"] == 168 and fast.rqmt["gpu_mem"] == 48 and fast._sis_id() == old_id
    fast.rqmt = old_resources


def test_earnings_diagnostic_cannot_leak_to_other_corpora():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.rnnt_qwen import run
    with pytest.raises(ValueError, match="Earnings22"):
        run(data={"librispeech-test-other": None}, tuning_only=False, earnings_pruning=True)
