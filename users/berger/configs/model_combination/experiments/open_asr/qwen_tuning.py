"""Focused Qwen development experiments; never tune on held-out corpora."""

import hashlib
import inspect
from pathlib import Path

from ...model_pipelines.common.experiment_context import ExperimentContext


def development_data():
    from . import source_version
    from ...data import loquacious_benchmark, benchmark_subset
    from ...data.loquacious.datasets import _get_dev_short_segments
    from ...data.open_asr import BenchmarkDataset, PrepareOpenAsrDatasetJob

    source = inspect.getsource(loquacious_benchmark.PrepareLoquaciousBenchmarkJob).encode()
    version = hashlib.sha256(source + source_version().encode()).hexdigest()
    if version == "cc9862f6d7ae2716f3c36d6df65b4598901c240a6166ed98e9c11343ac28282b":
        version = "504e4a4d6c28795fb1daa1f5e074528f4e0b104c7ff20b6eb4c0e8d8572b7eee"
    parent = loquacious_benchmark.PrepareLoquaciousBenchmarkJob("dev", code_version=version)
    with ExperimentContext("open-asr/qwen-parakeet-rnnt/tuning"):
        short = benchmark_subset.PrepareBenchmarkSubsetJob(
            parent.out_dataset, parent.out_references, parent.out_manifest,
            _get_dev_short_segments(),
            code_version=hashlib.sha256(Path(benchmark_subset.__file__).read_bytes()).hexdigest(),
        )
        short.add_alias("dev.short")
    ls = PrepareOpenAsrDatasetJob(
        BenchmarkDataset("librispeech-dev-other", "openslr/librispeech_asr",
                         "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1", "other", "validation"),
        code_version=source_version(),
    )
    return {"librispeech-dev-other": ls, "loquacious-dev.short": short}


def run():
    from . import qwen, qwen_speech, rnnt_qwen

    data = development_data()
    speech = qwen_speech.run(data=data, tuning=True)
    results = [speech]
    for size in ("0.6b", "1.7b"):
        results.append(qwen.run(data=data, tuning=True, text_size=size, two_pass=True))
        results.append(rnnt_qwen.run(
            data=data, speech_baselines=speech["scores"]["qwen-speech/native"],
            tuning_only=True, focused_tuning=True, text_size=size,
        ))
    large_speech = qwen_speech.run(data=data, tuning=True, speech_size="1.7b")
    results.append(large_speech)
    results.append(rnnt_qwen.run(
        data=data, speech_baselines=large_speech["scores"]["qwen-speech/native"],
        tuning_only=True, speech_size="1.7b", speech_only=True,
    ))
    from .pretrained_extensions import run_standalone
    run_standalone(data, speech_size="1.7b")
    return results
