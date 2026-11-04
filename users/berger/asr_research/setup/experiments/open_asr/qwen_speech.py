"""Qwen3-ASR standalone and audio-conditioned Granite fusion experiments."""

import hashlib
from pathlib import Path

from sisyphus import tk
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob

from ...model_pipelines.common.experiment_context import ExperimentContext

SPEECH_REPO = "Qwen/Qwen3-ASR-0.6B-hf"
SPEECH_REVISION = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"
SPEECH_MODELS = {
    "0.6b": (SPEECH_REPO, SPEECH_REVISION),
    "1.7b": ("Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"),
}


def fusion_config(vocabulary, *, scale=0.3):
    from .qwen import dynamic_config

    cfg = dynamic_config(vocabulary, None)
    search = cfg.lib_rasr.search_algorithm
    search.llm = RasrConfig()
    search.llm.type = "qwen-speech"
    search.llm_scale = scale
    search.word_penalty = 0.0
    return cfg


def run(*, data=None, tuning=False, speech_size="0.6b"):
    from . import MODELS, source_version, scoring_version
    from ... import tools
    from ...data.huggingface import DownloadHuggingFaceRepoJob
    from ...model_pipelines.common import dynamic_llm, dynamic_llm_jobs, qwen_speech
    from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob
    from ...model_pipelines.common.qwen_benchmark import QwenBenchmarkOverviewJob
    from ...data.loquacious_benchmark import LoquaciousSubsetJob, FILTERS
    from ...data.loquacious.datasets import _get_dev_short_segments

    if data is None:
        from .qwen import run as run_qwen

        data = run_qwen()["data"]
    if tuning and (data is None or set(data) - {"librispeech-dev-other", "loquacious-dev.short"}):
        raise ValueError("Speech-Qwen tuning requires explicit development datasets")
    speech_id, speech_revision = SPEECH_MODELS[speech_size]
    prefix = f"open-asr/qwen-speech-{speech_size}"
    # Hash runtime code, not alias/output wiring. Preserve the reviewed runtime's
    # existing fusion identity; recognition parameters are already job inputs.
    version = hashlib.sha256(Path(qwen_speech.__file__).read_bytes()).hexdigest()
    if version == "c5cafcfd0cec939ec1d2198b97635dcf1bfe33fb7b9dc2ff7eb27d78fee8d3c5":
        version = "12d174b8537a64b44dfe89562d226dfa6196a0f1b69d8fe59a99910b5bff1a71"
    native_version = hashlib.sha256(Path(qwen_speech.__file__).read_bytes()).hexdigest()
    # Reviewed batching-only migration: native generate() is unchanged.
    # Future runtime edits still invalidate native jobs unless reviewed again.
    if native_version in {
        "644de60ac0ca5038bdb1c5c7330a5d9bf3056e15b42c7ea006684c42f2a584e7",
        "c5cafcfd0cec939ec1d2198b97635dcf1bfe33fb7b9dc2ff7eb27d78fee8d3c5",
    }:
        native_version = "a514c5497cdf56142ec58409f162984032229e9b268a01709d329fda364166dc"
    with ExperimentContext(prefix):
        speech = DownloadHuggingFaceRepoJob(speech_id, revision=speech_revision).out_repo_dir
        speech.creator.add_alias("model")
    _, repo_id, revision = MODELS["granite-ctc"]
    granite = DownloadHuggingFaceRepoJob(repo_id, revision=revision).out_repo_dir
    raw_version = hashlib.sha256(
        Path(dynamic_llm.__file__).read_bytes()
        + Path(dynamic_llm_jobs.__file__).read_bytes()
        + source_version().encode()
        + scoring_version().encode()
    ).hexdigest()
    raw = dynamic_llm_jobs.GraniteRawPiecesJob(granite, code_version=raw_version)
    binary = tools.get_llm_search_rasr_binary_path()
    scores, runtimes, labels = {}, {}, {}

    def register(name, condition, prepared, make_job):
        path = f"{prefix}/{condition}/{name}"
        with ExperimentContext(path):
            job = make_job()
            if speech_size == "1.7b":
                job.rqmt.update(gpu_mem=48, mem=96)
            if condition != "native":
                from .runtime_budget import apply_runtime_budget
                apply_runtime_budget(job, name)
            job.add_alias("recognition")
            score = ScoreBenchmarkJob(prepared.out_references, job.out_hypotheses, code_version=scoring_version())
            score.add_alias("scoring")
            for suffix, output in (
                ("hypotheses", job.out_hypotheses),
                ("scoring", score.out_summary),
                ("runtime", job.out_runtime),
                ("validation", job.out_validation),
            ):
                tk.register_output(suffix, output)
            if condition != "native":
                tk.register_output("rasr.log", job.out_log)
        key = f"qwen-speech/{condition}"
        scores.setdefault(key, {})[name] = score.out_summary
        runtimes.setdefault(key, {})[name] = job.out_runtime
        labels[key] = dict(lexicon="none", lm=f"audio-conditioned Qwen3-ASR-{speech_size.upper()}")
        if name.startswith("loquacious-") and name != "loquacious-dev.short":
            split = name.split("-", 1)[1]
            for subset in list(FILTERS) + (["short"] if split == "dev" else []):
                filtered = LoquaciousSubsetJob(
                    prepared.out_references,
                    job.out_hypotheses,
                    subset,
                    segments=_get_dev_short_segments() if subset == "short" else None,
                    manifest=prepared.out_manifest,
                )
                sub_name = f"{name}.{subset}"
                with ExperimentContext(f"{prefix}/{condition}/{sub_name}"):
                    sub_score = ScoreBenchmarkJob(
                        filtered.out_references, filtered.out_hypotheses, code_version=scoring_version()
                    )
                    sub_score.add_alias("scoring")
                    tk.register_output("scoring", sub_score.out_summary)
                scores[key][sub_name], runtimes[key][sub_name] = sub_score.out_summary, None
        return score

    for name, prepared in data.items():
        baseline = register(
            name,
            "native",
            prepared,
            lambda: qwen_speech.QwenSpeechRecogJob(prepared.out_dataset, speech, code_version=native_version),
        )
        # Default .3 is a transfer setting, not a held-out-selected optimum.
        # Additional scales are development-only; domain subsets (especially
        # Yodas) expose the deletion-vs-LM-strength trade-off without extra audio.
        scales = (0.05, 0.1, 0.3, 0.5) if tuning or name == "loquacious-dev" else (0.3,)
        for scale in scales:
            config = WriteRasrConfigJob(fusion_config(raw.out_vocab, scale=scale), RasrConfig()).out_config

            def make_job():
                return qwen_speech.QwenSpeechRecogJob(
                    prepared.out_dataset,
                    speech,
                    code_version=version,
                    granite_repo=granite,
                    binary=binary,
                    vocabulary=raw.out_vocab,
                    rasr_config=config,
                    baseline=baseline.out_summary,
                    batch_size=8,
                )

            register(name, "granite-fusion" if scale == 0.3 else f"granite-fusion-scale-{scale:g}", prepared, make_job)
    with ExperimentContext(prefix):
        overview = QwenBenchmarkOverviewJob(scores, runtimes, labels)
        tk.register_output("overview", overview.out_report)
    return dict(data=data, scores=scores, runtimes=runtimes)
