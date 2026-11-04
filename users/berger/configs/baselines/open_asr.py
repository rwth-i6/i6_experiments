"""Register the public Open ASR benchmark matrix and its scoring outputs."""

from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.rnnt_qwen import run as run_rnnt_qwen
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.qwen import run as run_qwen
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.qwen_speech import run as run_qwen_speech


def run() -> None:
    # Full held-out suite plus Loquacious, reusing completed two-pass results.
    # New hyperparameter/size grids stay in the separate qwen_tuning config.
    granite = run_qwen(loquacious=True, two_pass=True, matched_controls=True)
    speech = run_qwen_speech(data=granite["data"])
    rnnt = run_rnnt_qwen(
        data=granite["data"],
        speech_baselines=speech["scores"]["qwen-speech/native"],
        tuning_only=False,
        evaluation_only=True,
        matched_controls=True,
    )
    granite_large = run_qwen(data=granite["data"], text_size="1.7b", two_pass=True)
    rnnt_large = run_rnnt_qwen(
        data=granite["data"], speech_baselines=speech["scores"]["qwen-speech/native"],
        tuning_only=False, evaluation_only=True, text_size="1.7b",
    )
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.pretrained_extensions import run_standalone

    run_standalone(granite["data"])
    speech_large = run_qwen_speech(data=granite["data"], speech_size="1.7b")
    rnnt_speech_large = run_rnnt_qwen(
        data=granite["data"], speech_baselines=speech_large["scores"]["qwen-speech/native"],
        tuning_only=False, evaluation_only=True, speech_size="1.7b", speech_only=True,
    )
    run_standalone(granite["data"], speech_size="1.7b")
    from sisyphus import tk
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_benchmark import (
        QwenBenchmarkOverviewJob,
    )

    scores, runtimes, labels = {}, {}, {}
    for result in (granite, speech, rnnt):
        scores.update(result["scores"])
        runtimes.update(result["runtimes"])
    for result in (granite_large, rnnt_large):
        for key, value in result["scores"].items():
            model, condition = key.split("/", 1)
            new_key = f"{model}-qwen1.7b/{condition}"
            scores[new_key] = value
            runtimes[new_key] = result["runtimes"][key]
    for result in (speech_large, rnnt_speech_large):
        for key, value in result["scores"].items():
            model, condition = key.split("/", 1)
            new_key = f"{model}-speech1.7b/{condition}"
            scores[new_key] = value
            runtimes[new_key] = result["runtimes"][key]
    for condition in scores:
        method = condition.split("/", 1)[1]
        auxiliary = "none"
        if method.startswith(("speech-qwen", "granite-fusion")):
            auxiliary = "Speech-Qwen 1.7B" if "speech1.7b/" in condition else "Speech-Qwen 0.6B"
        elif method.startswith(("one-pass", "two-pass", "text-qwen")):
            auxiliary = "Text Qwen 1.7B" if "qwen1.7b/" in condition else "Text Qwen 0.6B"
        labels[condition] = dict(
            lexicon="corpus" if "corpus-lexicon" in condition else "librispeech"
            if "librispeech-lexicon" in condition else "none",
            lm=auxiliary,
        )
    overview = QwenBenchmarkOverviewJob(scores, runtimes, labels)
    tk.register_output("open-asr/qwen-combination/overview", overview.out_report)
    from .results import py as register_reports

    register_reports()


def py() -> None:
    """Standalone suite config, honoring the currently enabled conditions."""
    # The suite already supplies open-asr/ prefixes; no corpus context here.
    run()


def main() -> None:
    py()
