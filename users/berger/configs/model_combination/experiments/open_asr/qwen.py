"""Frozen LibriSpeech settings for Granite + Qwen across public corpora."""

import hashlib
from pathlib import Path

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob


def dynamic_config(vocabulary, qwen, *, scale=0.3):
    cfg = RasrConfig()
    root = cfg.lib_rasr
    root.lexicon.file = DelayedFormat("vocab-text:{}", vocabulary)
    root.label_scorer.type = "no-op"
    root.label_scorer.scale = 1.0
    s = root.search_algorithm
    s.type = "llm-timesync-beam-search"
    s.blank_label_index = 0
    s.collapse_repeated_labels = True
    s.recombination_mode = "on"
    s.max_beam_size, s.pre_llm_max_beam_size = [32], 128
    s.score_threshold, s.pre_llm_score_threshold = [6.0], 6.0
    s.word_start_marker = "▁"
    s.llm_scale, s.word_penalty = scale, -1.0
    s.log_stepwise_statistics = False
    s.llm.type, s.llm.model = "qwen-dynamic-tuned", qwen
    s.llm.outer_lm_scale, s.llm.token_reward = scale, 0.5
    s.llm.formatting, s.llm.terminal_period = "original", False
    return cfg


def run(*, dataset_names=None, loquacious=True, two_pass=True, data=None,
        tuning=False, text_size="0.6b", matched_controls=False):
    from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
    from . import (
        MODELS,
        source_version,
        scoring_version,
        _lexicon,
        _granite_open_lexicon,
        public_resources,
        recognition_config,
    )
    from ... import tools
    from ...data.open_asr import DATASETS, PrepareOpenAsrDatasetJob
    from ...data.huggingface import DownloadHuggingFaceRepoJob
    from ...data.llm_fusion import QWEN_BASE_MODELS
    from ...data import loquacious_benchmark
    from ...model_pipelines.common import (
        dynamic_llm,
        dynamic_llm_jobs,
        dynamic_qwen_tuned,
        qwen_benchmark,
        qwen_full_dev,
    )
    from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob
    from ...model_pipelines.common.hf_benchmark import HfBenchmarkRecogJob
    from . import recognition_version
    from ..librispeech.nbest_qwen import run_preflight

    import inspect

    # Subset filtering is postprocessing, not dataset preparation or search.
    # Fingerprint preparation separately so fixes to compressed segment-list
    # reading don't invalidate all finished Loquacious/Open-ASR recognitions.
    preparation_source = inspect.getsource(loquacious_benchmark.PrepareLoquaciousBenchmarkJob).encode()
    version = hashlib.sha256(
        b"\0".join(Path(m.__file__).read_bytes() for m in (dynamic_qwen_tuned, qwen_benchmark, qwen_full_dev))
        + b"\0"
        + preparation_source
        + source_version().encode()
        + scoring_version().encode()
    ).hexdigest()
    # Reviewed migration: runtime/preparation unchanged; reuse scheduled jobs.
    if version == "cd0a7894ef8f929ec786ae21087e2d39d8a92b6f9ef54282ef1d634907fb22f5":
        version = "d5cb31d94484346ef450b900d038f74b2069ccffc103eaa315e6843ab6992000"
    # Exactly the original raw-piece and numerical gate version, so they reuse
    # finished LibriSpeech jobs instead of downloading/validating again.
    raw_version = hashlib.sha256(
        Path(dynamic_llm.__file__).read_bytes()
        + Path(dynamic_llm_jobs.__file__).read_bytes()
        + source_version().encode()
        + scoring_version().encode()
    ).hexdigest()
    if tuning and (
        data is None or set(data) - {"librispeech-dev-other", "loquacious-dev.short"}
    ):
        raise ValueError("Size/scale tuning requires explicit development datasets")
    provided_data = data is not None
    specs = [s for s in DATASETS if dataset_names is None or s.name in dataset_names]
    if dataset_names is not None and set(dataset_names) - {s.name for s in DATASETS}:
        raise ValueError("Unknown open-ASR datasets")
    if data is None:
        data = {s.name: PrepareOpenAsrDatasetJob(s, code_version=source_version()) for s in specs}
    if loquacious and not provided_data:
        data_version = hashlib.sha256(preparation_source + source_version().encode()).hexdigest()
        if data_version == "cc9862f6d7ae2716f3c36d6df65b4598901c240a6166ed98e9c11343ac28282b":
            data_version = "504e4a4d6c28795fb1daa1f5e074528f4e0b104c7ff20b6eb4c0e8d8572b7eee"
        data.update(
            {
                f"loquacious-{split}": loquacious_benchmark.PrepareLoquaciousBenchmarkJob(
                    split, code_version=data_version
                )
                for split in ("dev", "test")
            }
        )
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model"]
    _, model_id, revision = MODELS["granite-ctc"]
    repo = DownloadHuggingFaceRepoJob(model_id, revision=revision, allow_patterns=patterns).out_repo_dir
    qwen_repo, qwen_revision = QWEN_BASE_MODELS[text_size]
    qwen = DownloadHuggingFaceRepoJob(qwen_repo, revision=qwen_revision, allow_patterns=patterns).out_repo_dir
    dynamic_binary = tools.get_llm_search_rasr_binary_path()
    raw = dynamic_llm_jobs.GraniteRawPiecesJob(repo, code_version=raw_version)
    gate = dynamic_llm_jobs.DynamicQwenGateJob(dynamic_binary, qwen, code_version=raw_version)
    prefix = "open-asr/qwen-granite"
    if text_size != "0.6b":
        prefix += f"-{text_size}"
    preflight = run_preflight() if two_pass else None
    inv = (
        _lexicon(repo, "ctc", get_bliss_lexicon(), source_version(), f"{prefix}/librispeech-lexicon")
        if two_pass
        else None
    )
    ls_open = _granite_open_lexicon(repo, inv, preflight, f"{prefix}/librispeech-open-tree") if two_pass else None
    local_lexica = {}
    for name, (words, arpa) in public_resources(source_version()).items() if two_pass else ():
        if name in data:
            local_inv = _lexicon(repo, "ctc", words, source_version(), f"{prefix}/{name}/corpus-lexicon")
            local_lexica[name] = (
                local_inv,
                _granite_open_lexicon(repo, local_inv, preflight, f"{prefix}/{name}/corpus-open-tree"),
            )
    if loquacious and two_pass:
        from i6_experiments.common.datasets.loquacious.lexicon import get_bliss_lexicon as get_loquacious_lexicon

        words = get_loquacious_lexicon(
            use_stress_marker=False, add_silence=False, add_unknown_phoneme_and_mapping=False
        )
        local_inv = _lexicon(repo, "ctc", words, source_version(), f"{prefix}/loquacious/corpus-lexicon")
        local_open = _granite_open_lexicon(repo, local_inv, preflight, f"{prefix}/loquacious/corpus-open-tree")
        local_lexica.update({name: (local_inv, local_open) for name in data if name.startswith("loquacious-")})
    scores, runtimes, labels = {}, {}, {}

    def register(name, condition, prepared, hypotheses, runtime, lexicon, first_pass=False):
        path = f"{prefix}/{condition}/{name}"
        score = ScoreBenchmarkJob(prepared.out_references, hypotheses, code_version=scoring_version())
        score.add_alias(f"{path}/scoring")
        for suffix, value in (
            ("hypotheses", hypotheses),
            ("scoring", score.out_summary),
            ("segments", score.out_segments),
            ("runtime", runtime),
        ):
            tk.register_output(f"{path}/{suffix}", value)
        key = f"granite-ctc/{condition}"
        scores.setdefault(key, {})[name] = score.out_summary
        runtimes.setdefault(key, {})[name] = runtime
        labels[key] = dict(
            lexicon="dataset-specific" if lexicon == "corpus" else lexicon,
            lm="none" if first_pass else "Qwen3-0.6B-Base",
        )
        if name.startswith("loquacious-") and name != "loquacious-dev.short":
            from ...data.loquacious.datasets import _get_dev_short_segments

            split = name.split("-", 1)[1]
            subsets = list(loquacious_benchmark.FILTERS) + (["short"] if split == "dev" else [])
            for subset in subsets:
                filtered = loquacious_benchmark.LoquaciousSubsetJob(
                    prepared.out_references,
                    hypotheses,
                    subset,
                    segments=_get_dev_short_segments() if subset == "short" else None,
                    manifest=prepared.out_manifest,
                )
                sub_score = ScoreBenchmarkJob(
                    filtered.out_references, filtered.out_hypotheses, code_version=scoring_version()
                )
                sub_path = f"{prefix}/{condition}/loquacious-{split}.{subset}"
                sub_score.add_alias(f"{sub_path}/scoring")
                tk.register_output(f"{sub_path}/scoring", sub_score.out_summary)
                tk.register_output(f"{sub_path}/segments", sub_score.out_segments)
                tk.register_output(f"{sub_path}/hypotheses", filtered.out_hypotheses)
                subset_name = f"loquacious-{split}.{subset}"
                scores[key][subset_name] = sub_score.out_summary
                runtimes[key][subset_name] = None
        return score

    for name, prepared in data.items():
        prepared.add_alias(f"{prefix}/data/{name}")
        tk.register_output(f"{prefix}/data/{name}/manifest", prepared.out_manifest)
        native = HfBenchmarkRecogJob(
            prepared.out_dataset, repo, "ctc", code_version=recognition_version("ctc"), native=True
        )
        native.add_alias(f"{prefix}/native/{name}/recognition")
        register(name, "native", prepared, native.out_hypotheses, native.out_runtime, "none", first_pass=True)
        if matched_controls:
            cfg = dynamic_config(raw.out_vocab, qwen, scale=0.0)
            cfg.lib_rasr.search_algorithm.llm.type = "qwen-dynamic-zero"
            cfg.lib_rasr.search_algorithm.word_penalty = 0.0
            control = dynamic_llm_jobs.DynamicGraniteQwenJob(
                prepared.out_dataset, repo, qwen, raw.out_vocab, dynamic_binary,
                WriteRasrConfigJob(cfg, RasrConfig()).out_config, gate.out_report,
                zero=True, code_version=raw_version,
            )
            from .runtime_budget import apply_runtime_budget
            apply_runtime_budget(control, name)
            control.add_alias(f"{prefix}/zero-beam32-thr6/{name}/recognition")
            register(name, "zero-beam32-thr6", prepared, control.out_hypotheses,
                     control.out_runtime, "none", first_pass=True)
        for scale in ((0.1, 0.3, 0.5) if tuning else (0.3,)):
            condition = "one-pass" if scale == 0.3 else f"one-pass-scale-{scale:g}"
            dynamic = dynamic_qwen_tuned.TunedDynamicQwenJob(
                prepared.out_dataset, repo, qwen, raw.out_vocab, dynamic_binary,
                WriteRasrConfigJob(dynamic_config(raw.out_vocab, qwen, scale=scale), RasrConfig()).out_config,
                gate.out_report, code_version=version,
            )
            from .runtime_budget import apply_runtime_budget
            apply_runtime_budget(dynamic, name)
            dynamic.add_alias(f"{prefix}/{condition}/{name}/recognition")
            register(name, condition, prepared, dynamic.out_hypotheses, dynamic.out_runtime, "none")
            tk.register_output(f"{prefix}/{condition}/{name}/rasr.log", dynamic.out_log)
        if not two_pass:
            continue
        variants = [("librispeech", inv, ls_open)]
        if name in local_lexica:
            variants.append(("corpus", *local_lexica[name]))
        for resource, inventory, lexicon in variants:
            condition = f"two-pass-{resource}-lexicon"
            config = recognition_config(
                "ctc",
                inventory,
                lexicon=lexicon,
                lm=None,
                beam=256,
                word_beam=256,
                threshold=16,
                fallback=True,
                beta=1,
                alpha=2,
            )
            generation = qwen_benchmark.CompactGraniteNbestJob(
                prepared.out_dataset,
                repo,
                WriteRasrConfigJob(config, RasrConfig()).out_config,
                preflight["binary"],
                n=50,
                code_version=version,
            )
            lm = qwen_benchmark.FixedScoreQwenNbestJob(generation.out_candidates, qwen, code_version=version)
            apply_runtime_budget(generation, name)
            apply_runtime_budget(lm, name)
            selected = qwen_benchmark.SelectFixedQwenNbestJob(lm.out_scores, code_version=version)
            runtime = qwen_benchmark.QwenTwoPassRuntimeJob(generation.out_report, lm.out_report)
            path = f"{prefix}/{condition}/{name}"
            for suffix, job in (("generation", generation), ("qwen-scoring", lm), ("selection", selected)):
                job.add_alias(f"{path}/{suffix}")
            for suffix, value in (
                ("candidates", generation.out_candidates),
                ("generation", generation.out_report),
                ("qwen-scores", lm.out_scores),
                ("lm-scoring", lm.out_report),
                ("rasr.log", generation.out_log),
            ):
                tk.register_output(f"{path}/{suffix}", value)
            register(name, condition, prepared, selected.out_hypotheses, runtime.out_report, resource)
            if tuning:
                from ...model_pipelines.common import qwen_nbest_selection

                selection_version = hashlib.sha256(Path(qwen_nbest_selection.__file__).read_bytes()).hexdigest()
                for scale in (0.1, 0.5):
                    selection = qwen_nbest_selection.SelectQwenScaleJob(
                        lm.out_scores, scale, code_version=selection_version
                    )
                    selection.add_alias(f"{prefix}/{condition}-lm{scale:g}/{name}/selection")
                    register(name, f"{condition}-lm{scale:g}", prepared,
                             selection.out_hypotheses, runtime.out_report, resource)
            register(
                name,
                f"first-pass-{resource}-lexicon",
                prepared,
                generation.out_first_pass,
                generation.out_report,
                resource,
                first_pass=True,
            )
    overview = qwen_benchmark.QwenBenchmarkOverviewJob(scores, runtimes, labels)
    tk.register_output(f"{prefix}/overview", overview.out_report)
    return dict(scores=scores, runtimes=runtimes, data=data)
