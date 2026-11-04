"""Speech-Qwen size comparisons preserve old jobs and use size-specific controls."""

import pytest

from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import (
    qwen_speech, rnnt_qwen, qwen_tuning, pretrained_extensions,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.result_collection import describe, GROUPS


def test_large_speech_dev_grid_and_native_beams():
    data = qwen_tuning.development_data()
    large = qwen_speech.run(data=data, tuning=True, speech_size="1.7b")
    rnnt = rnnt_qwen.run(data=data, speech_baselines=large["scores"]["qwen-speech/native"],
                         tuning_only=True, speech_size="1.7b", speech_only=True)
    assert len(large["scores"]) == 5  # greedy + four scales
    assert len(rnnt["scores"]) == 4
    for result, prefix in ((large, "open-asr/qwen-speech-1.7b"),
                           (rnnt, "open-asr/qwen-parakeet-rnnt-speech-1.7b")):
        for key, rows in result["scores"].items():
            condition = key.split("/", 1)[1]
            for name, score in rows.items():
                job = score.creator.hypotheses.creator
                repo = job.auxiliary_repo if hasattr(job, "kind") else job.speech_repo
                assert repo.creator.repo_id == "Qwen/Qwen3-ASR-1.7B-hf"
                assert repo.creator.revision == qwen_speech.SPEECH_MODELS["1.7b"][1]
                assert job.rqmt["gpu_mem"] == 48
                assert f"{prefix}/{condition}/{name}" in job._sis_alias_prefixes
                if hasattr(job, "kind"):
                    assert job.kind == "speech"
                    assert job.speech_baseline == large["scores"]["qwen-speech/native"][name]
                    cfg = job.config.creator.config.lib_rasr.search_algorithm
                    assert cfg.max_beam_size == [8]
                    assert cfg.pre_llm_max_beam_size == 32
                    assert cfg.score_threshold == [3.0]
    beams = pretrained_extensions.run_standalone(data, speech_size="1.7b")
    assert len(beams) == 8
    assert {job.beam for job, _ in beams} == {4, 8}
    for job, _ in beams:
        assert job.speech_repo.creator.repo_id == "Qwen/Qwen3-ASR-1.7B-hf"
        assert job.rqmt["gpu_mem"] == 48


def test_large_speech_evaluation_has_no_sweeps_or_text_jobs():
    # Reuse a development preparation under a held-out key: graph test only.
    prepared = qwen_tuning.development_data()["librispeech-dev-other"]
    data = {"librispeech-test-other": prepared}
    large = qwen_speech.run(data=data, speech_size="1.7b")
    assert set(large["scores"]) == {"qwen-speech/native", "qwen-speech/granite-fusion"}
    rnnt = rnnt_qwen.run(data=data, speech_baselines=large["scores"]["qwen-speech/native"],
                         tuning_only=False, evaluation_only=True, speech_size="1.7b", speech_only=True)
    assert set(rnnt["scores"]) == {"parakeet-rnnt/speech-qwen-fast"}


@pytest.mark.parametrize("group,condition,model,lm", [
    ("qwen-speech-1.7b", "native-beam8", "Speech Qwen 1.7B", "None"),
    ("qwen-speech-1.7b", "granite-fusion", "Granite CTC", "Speech Qwen 1.7B"),
    ("qwen-parakeet-rnnt-speech-1.7b", "speech-qwen-fast", "Parakeet RNN-T", "Speech Qwen 1.7B"),
    ("qwen-parakeet-rnnt-1.7b", "text-qwen-beam8-thr2", "Parakeet RNN-T", "Text Qwen 1.7B"),
])
def test_report_sizes_are_unambiguous(group, condition, model, lm):
    assert group in GROUPS
    assert describe(dict(group=group, condition=condition))[:2] == (model, lm)
