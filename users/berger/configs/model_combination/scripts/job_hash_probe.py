"""Construct the migration comparison graph without executing jobs."""

import argparse
import json


def main(output):
    from sisyphus import tk
    from i6_experiments.users.berger.data.huggingface import DownloadHuggingFaceRepoJob
    from i6_experiments.users.berger.model_pipelines.common.rnnt_llm import RnntQwenRecogJob
    from i6_experiments.users.berger.configs.model_combination.experiments.open_asr import qwen, qwen_speech, rnnt_qwen

    assert DownloadHuggingFaceRepoJob.__module__.startswith("i6_experiments.example_setups.")
    assert RnntQwenRecogJob.__module__.startswith("i6_experiments.example_setups.")

    ctc = qwen.run(loquacious=True, two_pass=True, matched_controls=True)
    speech = qwen_speech.run(data=ctc["data"])
    rnnt_qwen.run(
        data=ctc["data"],
        speech_baselines=speech["scores"]["qwen-speech/native"],
        tuning_only=False,
        evaluation_only=True,
        matched_controls=True,
    )
    large = qwen_speech.run(data=ctc["data"], speech_size="1.7b")
    rnnt_qwen.run(
        data=ctc["data"],
        speech_baselines=large["scores"]["qwen-speech/native"],
        tuning_only=False,
        evaluation_only=True,
        speech_size="1.7b",
        speech_only=True,
    )
    rnnt_qwen.run(tuning_only=True, focused_tuning=True)
    from i6_experiments.users.berger.configs.model_combination.experiments.librispeech import training, recognition

    for name in ("ctc_bpe", "ffnn_transducer_bpe", "full_ctx_transducer_bpe", "aed_bpe"):
        model = getattr(training, name).run(descriptor=name)
        getattr(recognition, name).run(model=model, corpora=["dev-other"])
    jobs = sorted(job._sis_id() for job in tk.sis_graph.jobs())
    with open(output, "w") as dst:
        json.dump(jobs, dst, indent=2)
    print("JOB_HASHES", len(jobs), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Write sorted job IDs to this JSON file")
    main(parser.parse_args().output)
