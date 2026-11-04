"""Parakeet RNN-T + text/Speech-Qwen, without a pronunciation lexicon."""

import hashlib
from pathlib import Path

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob

from ...model_pipelines.common.experiment_context import ExperimentContext


def recognition_config(raw, *, kind, auxiliary_repo=None, scale=0.3, beam=32, threshold=6.0, pre_beam=None):
    cfg = RasrConfig()
    root = cfg.lib_rasr
    root.lexicon.file = DelayedFormat("vocab-text:{}", raw.out_vocab)
    root.label_scorer.type, root.label_scorer.scale = "hf-benchmark", 1.0
    s = root.search_algorithm
    s.type = "llm-rnnt-timesync-beam-search"
    s.blank_label_index = raw.out_blank
    s.collapse_repeated_labels = False
    s.max_labels_per_timeframe = 10
    s.length_norm_scale = 0.0
    s.recombination_mode = "viterbi" if beam == 1 else "sum"
    s.max_beam_size = [beam]
    s.pre_llm_max_beam_size = pre_beam if pre_beam is not None else (1 if beam == 1 else 128)
    s.score_threshold, s.pre_llm_score_threshold = [threshold], threshold
    s.word_piece_convention, s.word_piece_marker = "word-start-marked", "▁"
    s.word_separator = " "
    s.log_stepwise_statistics = False
    s.word_penalty, s.llm_scale = 0.0, 0.0 if kind == "zero" else scale
    if kind == "zero":
        s.llm.type = "qwen-dynamic-zero"
    elif kind == "text":
        s.llm.type, s.llm.model = "qwen-dynamic-tuned", auxiliary_repo
        s.llm.outer_lm_scale, s.llm.token_reward = scale, 0.5
        s.llm.formatting, s.llm.terminal_period = "original", False
        s.word_penalty = -1.0
    elif kind == "speech":
        s.llm.type = "qwen-speech"
    else:
        raise ValueError(kind)
    return cfg


def focused_conditions(*, text_size="0.6b"):
    rows = [(f"text-qwen-scale-{s:g}", "text", s, 32, 2.0) for s in (0.1, 0.3, 0.5)]
    rows += [(f"text-qwen-beam{b}-thr2", "text", 0.3, b, 2.0, 32) for b in (8, 16)]
    if text_size == "0.6b":
        rows += [(f"speech-qwen-scale-{s:g}-beam8-pre32-thr3", "speech", s, 8, 3.0, 32)
                 for s in (0.05, 0.1, 0.2, 0.3)]
        rows += [("speech-qwen-scale-0.1-beam16-pre32-thr3", "speech", 0.1, 16, 3.0, 32)]
    return rows


def earnings_pruning_conditions():
    # Diagnostic tuning on Earnings22, explicitly not a held-out result.
    # Keep LM scale/casing/rewards fixed; vary only search budgets.
    rows = [("speech-qwen-fast", "speech", 0.3, 8, 3.0, 32),
            ("speech-qwen", "speech", 0.3, 32, 4.0, 128)]
    budgets = [(32, 128, t) for t in (3.0, 3.5)]
    budgets += [(24, 128, t) for t in (3.5, 4.0)]
    budgets += [(16, 128, t) for t in (3.0, 3.5, 4.0)]
    budgets += [(8, 128, 4.0)]
    budgets += [(16, 64, t) for t in (3.5, 4.0)]
    rows += [(f"speech-qwen-earnings-prune-beam{b}-pre{p}-thr{t:g}", "speech", 0.3, b, t, p)
             for b, p, t in budgets]
    return rows


def run(*, data=None, speech_baselines=None, tuning_only=True, evaluation_only=False,
        focused_tuning=False, text_size="0.6b", earnings_pruning=False, matched_controls=False,
        speech_size="0.6b", speech_only=False):
    from . import MODELS, scoring_version, recognition_version
    from .qwen_speech import SPEECH_MODELS
    from ... import tools
    from ...data.huggingface import DownloadHuggingFaceRepoJob
    from ...data.llm_fusion import QWEN_BASE_MODELS
    from ...data.loquacious_benchmark import LoquaciousSubsetJob, FILTERS
    from ...data.loquacious.datasets import _get_dev_short_segments
    from ...model_pipelines.common import rnnt_llm, qwen_speech, dynamic_qwen_tuned
    from ...model_pipelines.common.hf_benchmark import HfBenchmarkRecogJob
    from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob
    from ...model_pipelines.common.qwen_benchmark import QwenBenchmarkOverviewJob

    if tuning_only and evaluation_only:
        raise ValueError("Choose tuning or held-out evaluation, not both")
    if speech_size != "0.6b" and not speech_only:
        raise ValueError("Non-default Speech-Qwen sizes require speech_only=True")
    if earnings_pruning and (tuning_only or evaluation_only or text_size != "0.6b"
                            or data is None or set(data) != {"earnings22-cleaned"}):
        raise ValueError("Earnings22 pruning is an explicit corpus-specific diagnostic only")
    if focused_tuning and not tuning_only:
        raise ValueError("Focused tuning is development-only")
    if tuning_only:
        import inspect
        from . import source_version
        from ...data import loquacious_benchmark, benchmark_subset

        # Match the existing parent preparation hash, but decode only dev.short.
        if data is None:
            preparation = inspect.getsource(loquacious_benchmark.PrepareLoquaciousBenchmarkJob).encode()
            data_version = hashlib.sha256(preparation + source_version().encode()).hexdigest()
            if data_version == "cc9862f6d7ae2716f3c36d6df65b4598901c240a6166ed98e9c11343ac28282b":
                data_version = "504e4a4d6c28795fb1daa1f5e074528f4e0b104c7ff20b6eb4c0e8d8572b7eee"
            parent = loquacious_benchmark.PrepareLoquaciousBenchmarkJob("dev", code_version=data_version)
            with ExperimentContext("open-asr/qwen-parakeet-rnnt/tuning"):
                short = benchmark_subset.PrepareBenchmarkSubsetJob(
                    parent.out_dataset,
                    parent.out_references,
                    parent.out_manifest,
                    _get_dev_short_segments(),
                    code_version=hashlib.sha256(Path(benchmark_subset.__file__).read_bytes()).hexdigest(),
                )
                short.add_alias("dev.short")
            data = {"loquacious-dev.short": short}
        elif set(data) - {"loquacious-dev.short", "librispeech-dev-other"}:
            raise ValueError("Tuning mode accepts only dev.short and LibriSpeech dev-other")
    elif data is None:
        from .qwen import run as run_qwen

        data = run_qwen()["data"]
    if speech_baselines is None and not tuning_only and not earnings_pruning:
        from .qwen_speech import run as run_speech

        speech_baselines = run_speech(data=data, speech_size=speech_size)["scores"]["qwen-speech/native"]
    prefix = "open-asr/qwen-parakeet-rnnt"
    if text_size != "0.6b":
        prefix += f"-{text_size}"
    if speech_only:
        prefix += f"-speech-{speech_size}"
    version = hashlib.sha256(
        Path(rnnt_llm.__file__).read_bytes()
        + Path(qwen_speech.__file__).read_bytes()
        + Path(dynamic_qwen_tuned.__file__).read_bytes()
    ).hexdigest()
    # Reviewed migration: runtime unchanged; adding conditions must not restart
    # existing recognitions, inventory or gate. Config files hash pruning values.
    if version == "cc10aaf0ab333567f13e93418f11e8a0030ffbb8a8f1f624c5e62230afef11ba":
        version = "abab72d5216298b977645312dfa7d53f0802be70adc00d21f68e1ffbb44592fb"
    _, repo_id, revision = MODELS["parakeet-rnnt"]
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model"]
    repo = DownloadHuggingFaceRepoJob(repo_id, revision=revision, allow_patterns=patterns).out_repo_dir
    text_repo, text_revision = QWEN_BASE_MODELS[text_size]
    text = DownloadHuggingFaceRepoJob(text_repo, revision=text_revision, allow_patterns=patterns).out_repo_dir
    speech_id, speech_revision = SPEECH_MODELS[speech_size]
    speech = DownloadHuggingFaceRepoJob(speech_id, revision=speech_revision).out_repo_dir
    if speech_baselines is None:
        # Register standalone Speech-Qwen controls only, not the Granite/full suite.
        native_version = hashlib.sha256(Path(qwen_speech.__file__).read_bytes()).hexdigest()
        if native_version in {
            "644de60ac0ca5038bdb1c5c7330a5d9bf3056e15b42c7ea006684c42f2a584e7",
            "c5cafcfd0cec939ec1d2198b97635dcf1bfe33fb7b9dc2ff7eb27d78fee8d3c5",
        }:
            native_version = "a514c5497cdf56142ec58409f162984032229e9b268a01709d329fda364166dc"
        speech_baselines = {}
        for name, prepared in data.items():
            path = f"{prefix}/speech-native/{name}"
            with ExperimentContext(path):
                native = qwen_speech.QwenSpeechRecogJob(prepared.out_dataset, speech, code_version=native_version)
                native.add_alias("recognition")
                score = ScoreBenchmarkJob(
                    prepared.out_references, native.out_hypotheses, code_version=scoring_version()
                )
                score.add_alias("scoring")
                for suffix, output in (
                    ("hypotheses", native.out_hypotheses),
                    ("scoring", score.out_summary),
                    ("runtime", native.out_runtime),
                ):
                    tk.register_output(suffix, output)
            speech_baselines[name] = score.out_summary
    binary = tools.get_llm_rnnt_search_rasr_binary_path()
    with ExperimentContext(prefix):
        raw = rnnt_llm.RnntRawPiecesJob(repo, code_version=version)
        raw.add_alias("inventory")
        gate = rnnt_llm.RnntLlmGateJob(binary, code_version=version)
        gate.add_alias("gate")
        tk.register_output("gate", gate.out_report)
    scores, runtimes, labels = {}, {}, {}

    def register(name, condition, prepared, make_job):
        path = f"{prefix}/{condition}/{name}"
        # The scope must be active during construction: Sisyphus captures the
        # alias prefix there. Recognition/scoring aliases are sibling leaves.
        with ExperimentContext(path):
            job = make_job()
            if speech_only and speech_size == "1.7b":
                job.rqmt.update(gpu_mem=48, mem=96)
            if isinstance(job, rnnt_llm.RnntQwenRecogJob) and job.kind != "zero":
                from .runtime_budget import apply_runtime_budget
                apply_runtime_budget(job, name)
            job.add_alias("recognition")
            score = ScoreBenchmarkJob(prepared.out_references, job.out_hypotheses, code_version=scoring_version())
            score.add_alias("scoring")
            for suffix, output in (
                ("hypotheses", job.out_hypotheses),
                ("scoring", score.out_summary),
                ("runtime", job.out_runtime),
                ("segments", score.out_segments),
            ):
                tk.register_output(suffix, output)
            if isinstance(job, rnnt_llm.RnntQwenRecogJob):
                tk.register_output("rasr.log", job.out_log)
                tk.register_output("validation", job.out_validation)
        key = f"parakeet-rnnt/{condition}"
        scores.setdefault(key, {})[name], runtimes.setdefault(key, {})[name] = score.out_summary, job.out_runtime
        labels[key] = dict(
            lexicon="none",
            lm=f"Speech-Qwen {speech_size.upper()}"
            if condition.startswith("speech")
            else "Qwen Base"
            if condition.startswith("text")
            else "none",
        )
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
                    sub = ScoreBenchmarkJob(
                        filtered.out_references, filtered.out_hypotheses, code_version=scoring_version()
                    )
                    sub.add_alias("scoring")
                    tk.register_output("scoring", sub.out_summary)
                scores[key][sub_name], runtimes[key][sub_name] = sub.out_summary, None

    for name, prepared in data.items():
        if not speech_only:
            register(
                name,
                "native",
                prepared,
                lambda: HfBenchmarkRecogJob(
                    prepared.out_dataset, repo, "rnnt", native=True, code_version=recognition_version("rnnt")
                ),
            )
        conditions = [
            ("zero-greedy", "zero", 0.0, 1, 6.0),
            ("zero-beam32", "zero", 0.0, 32, 6.0),
            ("text-qwen", "text", 0.3, 32, 6.0),
            ("speech-qwen", "speech", 0.3, 32, 6.0),
        ]
        if evaluation_only:
            # Freeze dev-selected thresholds; no zero-LM controls or held-out sweeps.
            conditions = [("text-qwen-beam8-thr2", "text", 0.3, 8, 2.0, 32)]
            if text_size == "0.6b":
                conditions += [("speech-qwen-fast", "speech", 0.3, 8, 3.0, 32)]
            if matched_controls:
                conditions += [(f"zero-beam8-thr{t:g}", "zero", 0.0, 8, t, 32) for t in (2.0, 3.0)]
        if focused_tuning:
            conditions = [("zero-greedy", "zero", 0.0, 1, 6.0),
                          ("zero-beam32", "zero", 0.0, 32, 6.0)]
            conditions += [(f"zero-beam8-thr{t:g}", "zero", 0.0, 8, t, 32) for t in (2.0, 3.0)]
            conditions += focused_conditions(text_size=text_size)
        if earnings_pruning:
            conditions = earnings_pruning_conditions()
        if not focused_tuning and not evaluation_only and not earnings_pruning and name in {"loquacious-dev", "loquacious-dev.short", "librispeech-dev-other"}:
            conditions.extend(
                (f"{kind}-qwen-scale-{scale:g}", kind, scale, 32, 6.0)
                for kind in ("text", "speech")
                for scale in (0.05, 0.1, 0.5)
            )
            conditions.extend(
                (f"{kind}-qwen-thr-{threshold:g}", kind, 0.3, 32, threshold)
                for kind in ("text", "speech")
                for threshold in (2.0, 3.0, 4.0)
            )
            # Target Speech-Qwen RTF <2 on dev; preserve held-out settings.
            conditions.extend(
                (f"speech-qwen-thr-{threshold:g}", "speech", 0.3, 32, threshold) for threshold in (1.0, 1.5)
            )
            conditions.extend(
                (f"speech-qwen-beam32-pre{pre}-thr3", "speech", 0.3, 32, 3.0, pre) for pre in (64, 32, 16)
            )
            conditions.extend((f"speech-qwen-beam{beam}-pre32-thr3", "speech", 0.3, beam, 3.0, 32) for beam in (16, 8))
        if speech_only:
            # Size comparison transfers the efficient 0.6B budget. Scale
            # selection stays on dev; no additional text-Qwen runs are made.
            conditions = [("speech-qwen-fast", "speech", 0.3, 8, 3.0, 32)]
            if tuning_only:
                conditions += [(f"speech-qwen-scale-{s:g}-beam8-pre32-thr3", "speech", s, 8, 3.0, 32)
                               for s in (0.05, 0.1, 0.5)]
        for setting in conditions:
            condition, kind, scale, beam, threshold = setting[:5]
            pre_beam = setting[5] if len(setting) == 6 else None
            auxiliary = text if kind == "text" else speech if kind == "speech" else None
            config = WriteRasrConfigJob(
                recognition_config(
                    raw,
                    kind=kind,
                    auxiliary_repo=auxiliary,
                    scale=scale,
                    beam=beam,
                    threshold=threshold,
                    pre_beam=pre_beam,
                ),
                RasrConfig(),
            ).out_config

            def make_job():
                return rnnt_llm.RnntQwenRecogJob(
                    prepared.out_dataset,
                    repo,
                    auxiliary,
                    raw.out_vocab,
                    binary,
                    config,
                    gate.out_report,
                    kind=kind,
                    code_version=version,
                    speech_baseline=speech_baselines[name] if kind == "speech" else None,
                )

            register(name, condition, prepared, make_job)
    with ExperimentContext(prefix):
        overview = QwenBenchmarkOverviewJob(scores, runtimes, labels)
        tk.register_output("overview", overview.out_report)
    return dict(scores=scores, runtimes=runtimes, data=data)
