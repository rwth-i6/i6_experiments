"""Public English short-form benchmark. Explicit entry point: ``run_all()``.

Do not import this into the historical LibriSpeech tuning grid. All six requested
conditions run for CTC/AED; RNNT tree conditions await a compatible RASR build.
No test-set tuning, LM training, or reference-derived vocabulary construction.
"""

import hashlib
from dataclasses import replace

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob
from i6_core.am.config import acoustic_model_config
from i6_experiments.users.berger.configs.model_combination.bootstrap import source_path

from ... import tools
from ...data.open_asr import DATASETS, PrepareOpenAsrDatasetJob
from ...data.huggingface import DownloadHuggingFaceRepoJob, HuggingFaceBpeBlissLexiconJob
from ...data.benchmark_lexicon import BenchmarkInventoryJob, PublicWordListLexiconJob
from ...model_pipelines.common.hf_benchmark import HfBenchmarkRecogJob
from ...model_pipelines.common.benchmark_scoring import (
    ScoreBenchmarkJob,
    CompareBenchmarkHypothesesJob,
    BenchmarkSummaryJob,
)


MODELS = {
    "granite-ctc": ("ctc", "ibm-granite/granite-speech-5.0-470m-turboctc-nc", "1ea89b7"),
    "parakeet-rnnt": ("rnnt", "nvidia/parakeet-rnnt-1.1b", "2acc4c61eede2f52ddefe74935de1930a9064d4a"),
    "cohere-aed": ("aed", "CohereLabs/cohere-transcribe-03-2026", "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"),
}
CONDITIONS = ("native", "rasr-lexfree", "tree", "tree-4gram", "tree-trafo", "tree-qwen")
RNNT_COMMIT = "59767f21727326691de3812affac4c4900449c65"


def source_version():
    """Hash shared data/normalization code, independently of decoding and scoring.

    Preserve the already scheduled suite's version when its runtime sources are
    unchanged. The original fingerprint also included this driver and scoring;
    the one-time migration below avoids rerunning ASR for a scoring-only fix.
    Config values and dependency paths still enter each job's ordinary hash.
    """
    paths = [
        source_path("data/open_asr.py"),
        source_path("data/benchmark_lexicon.py"),
        source_path("data/huggingface.py"),
        source_path("model_pipelines/common/hf_scoring.py"),
    ]
    paths += sorted(source_path("model_pipelines/common/hf_normalizer").glob("*.py"))
    digest = hashlib.sha256(b"\0".join(p.read_bytes() for p in paths)).hexdigest()
    if digest == "1da74feffccf8e357fb4e43996345ac6aeee3aabb31c34abb246885df69e2fd3":
        return "c37b1c7c749309ef10c6e704b895bdb7f689ffa5be54364a73afb0f017d73cb7"
    return digest


def recognition_version(family, dataset=None):
    """Refresh Cohere for chunk support without rerunning data or other models.

    The fingerprints below are the pre-chunk adapter and the reviewed
    chunk-capable adapter (whose single-chunk CTC/RNNT computation is unchanged).
    Future adapter edits invalidate all models unless explicitly audited here.
    """
    adapter = source_path("model_pipelines/common/hf_benchmark.py")
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    original = "33163df58f92d0f621edc3e31e36b516351f077c476a25e4d7b4b49517c2dc52"
    chunked = "75293c0009a1b78811cc07b1cc8cbc7675654e90cc5bc44939a55d7330a90cb0"
    logging_only = "a48d406d9930d7b541b09f4e5fe69b7e91556ca18049ca6199db506d8e283226"
    fallback_dispatch = "b7251e6267bf455beaa621dd9c323d161f91e056de9e61600d439b3609d6c9ac"
    partial_aed = "a2e96c25c0ca2172c98947c3aefe318ae65885d54b55b525aaebc069f2398d53"
    # Reviewed AED-only fix: return partial hypotheses at the shared label cap.
    # All completed AED corpora already emitted EOS before the cap; preserve
    # those jobs, and all CTC/RNNT jobs. Refresh only the failed AED corpora.
    if digest == partial_aed and (
        family != "aed" or dataset not in (None, "earnings22-cleaned", "gigaspeech-cleaned")
    ):
        digest = chunked
    # Disabling stepwise statistics changes logging only, not hypotheses.
    if digest in (logging_only, fallback_dispatch):
        digest = chunked
    if digest == original or (family in ("ctc", "rnnt") and digest == chunked):
        return source_version()
    return hashlib.sha256(source_version().encode() + b"\0" + digest.encode()).hexdigest()


def scoring_version():
    """Invalidate only scoring/comparison jobs when their implementation changes."""
    scoring = source_path("model_pipelines/common/benchmark_scoring.py")
    return hashlib.sha256(source_version().encode() + b"\0" + scoring.read_bytes()).hexdigest()


def rnnt_binary():
    from i6_core.tools.git import CloneGitRepositoryJob
    from i6_core.tools.compile import CMakeJob

    source = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/rasr.git", commit=RNNT_COMMIT, checkout_folder_name="rasr"
    )
    build = CMakeJob(
        source_folder=source.out_repository,
        num_processes=8,
        mem_rqmt=8,
        cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
    )
    source.add_alias("open-asr/tools/rasr-rnnt/source")
    build.add_alias("open-asr/tools/rasr-rnnt/cmake")
    tk.register_output("open-asr/tools/rasr-rnnt/binary", build.out_install_dir)
    return build.out_install_dir


def recognition_config(
    family,
    inventory,
    *,
    lexicon=None,
    lm=None,
    beam=32,
    threshold=12.0,
    word_beam=None,
    fallback=False,
    beta=0.0,
    alpha=0.0,
):
    """One AM scorer, ordinary closed tree, boundary-independent acoustic ids."""
    cfg = RasrConfig()
    if fallback and (family != "ctc" or lexicon is None):
        raise ValueError("Open-tree benchmark conditions require Granite CTC with a prepared lexicon")
    root = cfg.lib_rasr
    search = root.search_algorithm
    root.label_scorer.type = "no-op" if family == "ctc" else "hf-benchmark"
    root.label_scorer.scale = 1.0
    search.max_beam_size = [beam]
    search.score_threshold = [threshold]
    search.recombination_mode = "on"
    # Historical hashed value; job execution overrides this logging-only option
    # to false, including when reusing an already-written configuration file.
    search.log_stepwise_statistics = True
    if lexicon is None:
        root.lexicon.file = DelayedFormat("vocab-text:{}", inventory.out_vocab)
        search.type = {
            "ctc": "lexiconfree-timesync-beam-search",
            "aed": "lexiconfree-labelsync-beam-search",
            "rnnt": "lexiconfree-rnnt-timesync-beam-search",
        }[family]
        if family == "aed":
            search.sentence_end_label_index = inventory.out_eos
            search.length_norm_scale = 0.0
            search.pruning_strategy_type = "joint"
            search.max_labels_per_timestep = 500
        else:
            search.blank_label_index = inventory.out_blank
            if family == "ctc":
                search.collapse_repeated_labels = True
            else:
                search.max_labels_per_timeframe = 10
    else:
        if family == "rnnt":
            raise NotImplementedError("RNNT tree search awaits the user's RASR implementation")
        root.lexicon.file = lexicon
        root.acoustic_model = acoustic_model_config(states_per_phone=1, phon_history_length=0, phon_future_length=0)
        root.acoustic_model.state_tying.type = "monophone"
        root.acoustic_model.allophones.add_all = True
        root.acoustic_model.allophones.add_from_lexicon = False
        search.type = "tree-timesync-beam-search" if family == "ctc" else "tree-labelsync-beam-search"
        search.tree_builder_type = family
        search.unknown_word_fallback = "disabled"
        if fallback:
            search.unknown_word_fallback = "known-excluding"
            search.unknown_word_tokenization = "word-start-marked"
            search.unknown_word_penalty = beta
            search.unknown_piece_penalty = alpha
        search.sentence_end_fall_back = False
        search.max_word_end_beam_size = beam if word_beam is None else word_beam
        search.word_end_score_threshold = threshold
        if family == "ctc":
            search.collapse_repeated_labels = True
            search.force_blank_between_repeated_labels = True
        else:
            search.length_norm_scale = 0.0
            search.pruning_strategy_type = "joint"
            search.max_labels_per_timestep = 500
        if lm is not None:
            root.lm = lm
        else:
            root.lm.scale = 0.0
    return cfg


def _lexicon(repo, family, source, version, prefix):
    pronounce = HuggingFaceBpeBlissLexiconJob(source, repo, add_unknown_lemma=True)
    inventory = BenchmarkInventoryJob(repo, family, lexicon=pronounce.out_lexicon, code_version=version)
    pronounce.add_alias(f"{prefix}/pronunciations")
    inventory.add_alias(f"{prefix}/inventory")
    tk.register_output(f"{prefix}/coverage", pronounce.out_report)
    tk.register_output(f"{prefix}/inventory.json", inventory.out_metadata)
    tk.register_output(f"{prefix}/labels.txt", inventory.out_vocab)
    tk.register_output(f"{prefix}/lexicon", inventory.out_lexicon)
    return inventory


def _qwen(lexicon, preflight, prefix):
    from ...data.llm_fusion import QWEN_REPO, QWEN_REVISION, ExportQwenLmJob, ValidateQwenRasrJob
    from ...data.qwen_text_variants import (
        QwenTextVariantLexiconJob,
        ExportQwenTextVariantsJob,
        SCORE_CASES,
        VARIANT_NATIVE_RUNNER,
    )
    from ...data.librispeech.lm import OnnxTokenLmParams

    repo = DownloadHuggingFaceRepoJob(
        QWEN_REPO, revision=QWEN_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
    ).out_repo_dir
    base = ExportQwenLmJob(repo)
    lex = QwenTextVariantLexiconJob(lexicon, repo)
    export = ExportQwenTextVariantsJob(repo, lex.out_report, base.out_report)
    check = ValidateQwenRasrJob(
        lex.out_lexicon,
        repo,
        export.out_model,
        lex.out_vocab,
        export.out_report,
        preflight["binary"],
        preflight["correctness"].out_report,
        preflight["pruning"].out_report,
        tools.returnn_python_exe,
        runner=VARIANT_NATIVE_RUNNER,
        variant_aliases=lex.out_report,
        score_cases=SCORE_CASES,
    )
    for name, job in (("lexicon", lex), ("export", export), ("validation", check)):
        job.add_alias(f"{prefix}/{name}")
        tk.register_output(f"{prefix}/{name}", job.out_report)
    return check.out_lexicon, OnnxTokenLmParams(export.out_model, lex.out_vocab, scale=0.3, max_batch_size=1)


def _granite_open_lexicon(repo, inventory, preflight, prefix, arpa_lm=None):
    """Tokenizer-defined fallback inventory; no benchmark reference words added."""
    from ...data.librispeech import datasets
    from ...data.oov_prototype import InspectWordStartTokenizerJob, BuildGraniteOovLexiconJob
    from ..librispeech.oov_prototype import _validated_lexicon

    inspection = InspectWordStartTokenizerJob(repo, datasets.get_default_score_corpus("dev-other").bliss_corpus_file)
    extended = BuildGraniteOovLexiconJob(inventory.out_lexicon, inspection.out_table)
    for name, job in (("tokenizer", inspection), ("preparation", extended)):
        job.add_alias(f"{prefix}/{name}")
        tk.register_output(f"{prefix}/{name}", job.out_report)
    tk.register_output(f"{prefix}/pieces", inspection.out_table)
    return _validated_lexicon(
        extended.out_lexicon,
        inventory.out_vocab,
        preflight["binary"],
        prefix,
        decoder_smoke_report=preflight["correctness"].out_report,
        pruning_report=preflight["pruning"].out_report,
        arpa_lm=arpa_lm,
    )


def public_resources(version):
    from i6_core.tools.download import DownloadJob

    giga = DownloadHuggingFaceRepoJob(
        "wgb14/gigaspeech_lm",
        revision="57147dfe03d40315c0ea62cb8b22e9f9d3f13329",
        allow_patterns=["lexicon.txt", "4gram.arpa.gz"],
    )
    giga_words = PublicWordListLexiconJob(giga.out_repo_dir.join_right("lexicon.txt"), code_version=version)
    vox = DownloadJob(
        "https://dl.fbaipublicfiles.com/voxpopuli/lm/en/en_lm.lexicon",
        checksum="37b39c27b824ffd355d8c35bbb47b76bce69fef02183635f173802935d8f4d41",
    )
    vox_words = PublicWordListLexiconJob(vox.out_file, code_version=version)
    return {
        "gigaspeech-cleaned": (giga_words.out_lexicon, giga.out_repo_dir.join_right("4gram.arpa.gz")),
        "voxpopuli-cleaned": (vox_words.out_lexicon, None),
    }


def run_all(
    *,
    model_names=tuple(MODELS),
    dataset_names=tuple(s.name for s in DATASETS),
    corpus_resources=True,
    neural_lms=True,
    aed_tree=True,
    granite_open_tree=False,
    granite_open_transformer=False,
    corpus_open_tree=False,
):
    """Register the matrix; optional subsets support staged execution, not tuning.

    Models use fp32 and greedy native decoding. Tree beam=32/threshold=12 and LM
    scales=0.3 are frozen initial operating points, NOT claimed to be tuned for
    Cohere or the non-LibriSpeech datasets. No OOV fallback or subword LM is added.
    Set neural_lms=False to omit both neural-LM recognition conditions and their
    export/validation dependencies while tuning their search budget separately.
    Set aed_tree=False to keep AED native/lexicon-free evaluation only while
    label-sync tree scoring and termination are under investigation.
    granite_open_tree adds no-LM/4gram fallback runs plus matched closed controls,
    with costs/pruning selected on LibriSpeech dev-other, never benchmark tests.
    granite_open_transformer adds the GPU/stateless word Transformer fallback
    condition and its matched closed control, independently of neural_lms.
    """
    from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon

    # Initialize the existing experiment package before its LM module, whose
    # training-config imports otherwise re-enter the package mid-initialization.
    from ..librispeech.llm_fusion import run_preflight
    from ...data.librispeech.lm import ArpaLmParams, get_word_lm_config
    from ...data.oov_neural_lm import get_gpu_word_transformer_params

    if set(model_names) - MODELS.keys() or set(dataset_names) - {s.name for s in DATASETS}:
        raise ValueError("Unknown model or dataset selection")
    if granite_open_transformer and not granite_open_tree:
        raise ValueError("granite_open_transformer requires granite_open_tree=True")
    if corpus_open_tree and (not granite_open_tree or not corpus_resources):
        raise ValueError("corpus_open_tree requires granite_open_tree=True and corpus_resources=True")
    version = source_version()
    score_version = scoring_version()
    data = {s.name: PrepareOpenAsrDatasetJob(s, code_version=version) for s in DATASETS if s.name in dataset_names}
    for name, job in data.items():
        job.add_alias(f"open-asr/data/{name}")
        tk.register_output(f"open-asr/data/{name}/manifest", job.out_manifest)
    preflight = run_preflight() if any(MODELS[m][0] != "rnnt" for m in model_names) else None
    resources = public_resources(version) if corpus_resources else {}
    results = {}
    runtimes, resource_labels = {}, {}
    for model_name in model_names:
        family, repo_id, revision = MODELS[model_name]
        prefix = f"open-asr/{model_name}"
        download = DownloadHuggingFaceRepoJob(
            repo_id, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
        )
        download.add_alias(f"{prefix}/download")
        repo = download.out_repo_dir
        if family == "rnnt":
            binary = rnnt_binary()
            inventory = BenchmarkInventoryJob(repo, family, code_version=version)
            conditions = [("native", None, None, None), ("rasr-lexfree", None, None, None)]
        else:
            binary = preflight["binary"]
            inventory = _lexicon(repo, family, get_bliss_lexicon(), version, f"{prefix}/librispeech")
            lex = inventory.out_lexicon
            conditions = [
                ("native", None, None, None),
                ("rasr-lexfree", None, None, None),
            ]
            tree_enabled = family != "aed" or aed_tree
            if tree_enabled:
                conditions.extend(
                    [
                        ("tree", lex, None, None),
                        ("tree-4gram", lex, ArpaLmParams(scale=0.3), None),
                    ]
                )
            if tree_enabled and neural_lms:
                trafo = replace(get_gpu_word_transformer_params((lex,), f"{prefix}/lm"), scale=0.3, max_batch_size=1)
                qlex, qlm = _qwen(lex, preflight, f"{prefix}/qwen")
                conditions.extend([("tree-trafo", lex, trafo, None), ("tree-qwen", qlex, qlm, None)])
            if family == "ctc" and granite_open_tree:
                open_lex = _granite_open_lexicon(repo, inventory, preflight, f"{prefix}/open-tree")
                conditions.extend(
                    [
                        ("tree-open-control", open_lex, None, None),
                        ("tree-open", open_lex, None, None),
                        ("tree-open-control-4gram", open_lex, ArpaLmParams(scale=0.3), None),
                        ("tree-open-4gram", open_lex, ArpaLmParams(scale=0.3), None),
                    ]
                )
                if granite_open_transformer:
                    open_trafo = replace(
                        get_gpu_word_transformer_params((open_lex,), f"{prefix}/open-tree/lm"),
                        scale=0.3,
                        max_batch_size=1,
                    )
                    conditions.extend(
                        [
                            ("tree-open-control-trafo", open_lex, open_trafo, None),
                            ("tree-open-trafo", open_lex, open_trafo, None),
                        ]
                    )
            for dataset_name, (word_list, arpa) in resources.items():
                if not tree_enabled or dataset_name not in data:
                    continue
                local = _lexicon(repo, family, word_list, version, f"{prefix}/{dataset_name}/corpus-lexicon")
                conditions.append((f"corpus-tree-{dataset_name}", local.out_lexicon, None, dataset_name))
                if arpa is not None:
                    conditions.append(
                        (
                            f"corpus-tree-4gram-{dataset_name}",
                            local.out_lexicon,
                            ArpaLmParams(scale=0.3, lm_file=arpa),
                            dataset_name,
                        )
                    )
                if family == "ctc" and corpus_open_tree:
                    local_open = _granite_open_lexicon(
                        repo, local, preflight, f"{prefix}/{dataset_name}/open-tree", arpa_lm=arpa
                    )
                    # A corpus lexicon cannot be paired with an unrelated
                    # fixed word vocabulary (including the LibriSpeech Trafo).
                    pairs = [("", None)]
                    if arpa is not None:
                        pairs.append(("-corpus4gram", ArpaLmParams(scale=0.3, lm_file=arpa)))
                    for suffix, local_lm in pairs:
                        for mode in ("tree-open-control", "tree-open"):
                            conditions.append(
                                (
                                    f"{mode}-corpus-{dataset_name}{suffix}",
                                    local_open,
                                    local_lm,
                                    dataset_name,
                                )
                            )
        native_jobs = {}
        for condition, lexicon, lm_params, only_dataset in conditions:
            native = condition == "native"
            config = None
            if not native:
                lm = get_word_lm_config(lexicon, lm_params, rasr_binary_path_override=binary) if lm_params else None
                search_kwargs = {}
                if condition.startswith("tree-open"):
                    with_lm = lm_params is not None
                    transformer = condition.endswith("trafo")
                    search_kwargs = dict(
                        beam=64 if with_lm else 256,
                        threshold=16.0 if with_lm else 8.0,
                        word_beam=16 if transformer else None,
                        fallback=not condition.startswith("tree-open-control"),
                        beta=4.0 if transformer else (6.0 if with_lm else 1.0),
                        alpha=2.0,
                    )
                else:
                    search_kwargs["beam"] = 1 if lexicon is None else 32
                config = WriteRasrConfigJob(
                    recognition_config(family, inventory, lexicon=lexicon, lm=lm, **search_kwargs),
                    RasrConfig(),
                ).out_config
            scores = results.setdefault(f"{model_name}/{condition}", {})
            timing = runtimes.setdefault(f"{model_name}/{condition}", {})
            lex_source = only_dataset if only_dataset is not None else "LibriSpeech"
            lm_source = "none"
            if isinstance(lm_params, ArpaLmParams):
                lm_source = (
                    f"{only_dataset} 4gram" if only_dataset and lm_params.lm_file is not None else "LibriSpeech 4gram"
                )
            elif lm_params is not None:
                lm_source = "Qwen Base" if condition.endswith("qwen") else "LibriSpeech Transformer"
            resource_labels[f"{model_name}/{condition}"] = dict(
                lexicon=lex_source if lexicon is not None else "none",
                lm=lm_source,
            )
            for name, prepared in data.items():
                if only_dataset is not None and name != only_dataset:
                    continue
                path = f"{prefix}/{condition}/{name}"
                job = HfBenchmarkRecogJob(
                    prepared.out_dataset,
                    repo,
                    family,
                    code_version=(
                        hashlib.sha256(
                            recognition_version(family, name).encode()
                            + source_path("model_pipelines/common/benchmark_oov.py").read_bytes()
                        ).hexdigest()
                        if condition.startswith("tree-open")
                        else recognition_version(family, name)
                    ),
                    native=native,
                    rasr_config=config,
                    rasr_binary=binary if not native else None,
                    vocabulary=inventory.out_vocab if not native and lexicon is None else None,
                )
                if model_name == "granite-ctc" and condition == "tree-open-control-trafo" and name == "spgispeech":
                    # Previous 48h run completed 25,480/39,341 segments at a
                    # stable RTF. Resource-only change preserves its job hash.
                    job.rqmt["time"] = 96
                job.add_alias(path)
                score = ScoreBenchmarkJob(prepared.out_references, job.out_hypotheses, code_version=score_version)
                scores[name] = score.out_summary
                timing[name] = job.out_runtime
                for label, output in (
                    ("hypotheses", job.out_hypotheses),
                    ("runtime", job.out_runtime),
                    ("scoring", score.out_summary),
                    ("segments", score.out_segments),
                ):
                    tk.register_output(f"{path}/{label}", output)
                if native:
                    native_jobs[name] = job
                elif condition == "rasr-lexfree":
                    compare = CompareBenchmarkHypothesesJob(
                        native_jobs[name].out_hypotheses, job.out_hypotheses, code_version=score_version
                    )
                    tk.register_output(f"{path}/native-comparison", compare.out_report)
                    tk.register_output(f"{path}/native-differences", compare.out_differences)
    summary = BenchmarkSummaryJob(results, tuple(s.name for s in DATASETS), code_version=score_version)
    tk.register_output("open-asr/results.json", summary.out_report)
    from ...model_pipelines.common.benchmark_overview import BenchmarkResourceOverviewJob

    overview = BenchmarkResourceOverviewJob(results, runtimes, resource_labels)
    tk.register_output("open-asr/resource-overview.md", overview.out_report)
    # Local overviews remain runnable when another model/dataset has failed.
    for model_name in model_names:
        for dataset_name in data:
            local_scores = {
                key: {dataset_name: rows[dataset_name]}
                for key, rows in results.items()
                if key.startswith(f"{model_name}/") and dataset_name in rows
            }
            local_runtime = {key: {dataset_name: runtimes[key][dataset_name]} for key in local_scores}
            local_resources = {key: resource_labels[key] for key in local_scores}
            local_overview = BenchmarkResourceOverviewJob(local_scores, local_runtime, local_resources)
            tk.register_output(f"open-asr/overviews/{model_name}/{dataset_name}.md", local_overview.out_report)
    return results
