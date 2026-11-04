"""Small dev-only lexicon-free Qwen prototype, separate from word-LM trees."""

import hashlib
from pathlib import Path

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob

from ... import tools


def tuning_grid():
    # Start with bounded runtime; only one wider-beam diagnostic, not a Cartesian
    # product across all pruning settings. No held-out or non-LibriSpeech runs.
    rows = [
        dict(scale=s, penalty=p, beam=16, pre_beam=64, casing="original,lower,capitalized,upper")
        for s in (0.1, 0.3, 0.5)
        for p in (0.0, -1.0)
    ]
    rows += [
        dict(scale=0.3, penalty=0.0, beam=16, pre_beam=64, casing="lower"),
        dict(scale=0.3, penalty=0.0, beam=32, pre_beam=128, casing="original,lower,capitalized,upper"),
    ]
    rows += [dict(scale=0.0, penalty=0.0, beam=b, pre_beam=b, casing="lower") for b in (1, 16)]
    return rows


def run():
    from ..open_asr import MODELS, source_version, scoring_version
    from ...data.open_asr import BenchmarkDataset, PrepareOpenAsrDatasetJob
    from ...data.huggingface import DownloadHuggingFaceRepoJob
    from ...data.llm_fusion import QWEN_REPO, QWEN_REVISION
    from ...model_pipelines.common import dynamic_llm, dynamic_llm_jobs
    from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob

    prefix = "qwen-dynamic-9804f5aa/dev-other"
    version = hashlib.sha256(
        Path(dynamic_llm.__file__).read_bytes()
        + Path(dynamic_llm_jobs.__file__).read_bytes()
        + source_version().encode()
        + scoring_version().encode()
    ).hexdigest()
    dataset = PrepareOpenAsrDatasetJob(
        BenchmarkDataset(
            "librispeech-dev-other",
            "openslr/librispeech_asr",
            "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
            "other",
            "validation",
        ),
        code_version=source_version(),
    )
    _, model_id, revision = MODELS["granite-ctc"]
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model"]
    repo = DownloadHuggingFaceRepoJob(model_id, revision=revision, allow_patterns=patterns).out_repo_dir
    qwen = DownloadHuggingFaceRepoJob(QWEN_REPO, revision=QWEN_REVISION, allow_patterns=patterns).out_repo_dir
    binary = tools.get_llm_search_rasr_binary_path()
    inventory = dynamic_llm_jobs.GraniteRawPiecesJob(repo, code_version=version)
    gate = dynamic_llm_jobs.DynamicQwenGateJob(binary, qwen, code_version=version)
    inventory.add_alias(f"{prefix}/inventory")
    gate.add_alias(f"{prefix}/native-gate")
    tk.register_output(f"{prefix}/manifest", dataset.out_manifest)
    tk.register_output(f"{prefix}/inventory", inventory.out_report)
    tk.register_output(f"{prefix}/labels", inventory.out_vocab)
    tk.register_output(f"{prefix}/native-gate", gate.out_report)
    tk.register_output(f"{prefix}/native-gate.log", gate.out_log)
    outputs = {}
    for row in tuning_grid():
        zero = row["scale"] == 0
        name = f"lm{row['scale']:g}_wordpen{row['penalty']:g}_beam{row['beam']}_pre{row['pre_beam']}"
        name += "_lower" if row["casing"] == "lower" else "_cases"
        config = RasrConfig()
        root = config.lib_rasr
        root.lexicon.file = DelayedFormat("vocab-text:{}", inventory.out_vocab)
        root.label_scorer.type = "no-op"
        root.label_scorer.scale = 1.0
        search = root.search_algorithm
        search.type = "llm-timesync-beam-search"
        search.blank_label_index = 0
        search.collapse_repeated_labels = True
        search.recombination_mode = "on"
        search.max_beam_size = [row["beam"]]
        search.score_threshold = [14.0]
        search.pre_llm_max_beam_size = row["pre_beam"]
        search.pre_llm_score_threshold = 14.0
        search.word_start_marker = "▁"
        # Use the documented space default; an unquoted whitespace-only value
        # would be stripped by RASR's config-file parser.
        search.llm_scale = row["scale"]
        search.word_penalty = row["penalty"]
        search.log_stepwise_statistics = False
        search.llm.type = "qwen-dynamic-zero" if zero else "qwen-dynamic"
        search.llm.model = qwen
        search.llm.max_batch_size = 8
        search.llm.case_variants = row["casing"]
        recognition = dynamic_llm_jobs.DynamicGraniteQwenJob(
            dataset.out_dataset,
            repo,
            qwen,
            inventory.out_vocab,
            binary,
            WriteRasrConfigJob(config, RasrConfig()).out_config,
            gate.out_report,
            zero=zero,
            code_version=version,
        )
        score = ScoreBenchmarkJob(dataset.out_references, recognition.out_hypotheses, code_version=version)
        recognition.add_alias(f"{prefix}/{name}/recognition")
        score.add_alias(f"{prefix}/{name}/scoring")
        for suffix, path in (
            ("hypotheses", recognition.out_hypotheses),
            ("runtime", recognition.out_runtime),
            ("rasr.log", recognition.out_log),
            ("scoring", score.out_summary),
            ("segments", score.out_segments),
        ):
            tk.register_output(f"{prefix}/{name}/{suffix}", path)
        outputs[name] = score
    return outputs


def run_rescoring_transfer():
    """Bounded new grid; reuse the unchanged original build/inventory/gate."""
    import copy
    from ...model_pipelines.common import dynamic_qwen_tuned
    from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob

    controls = run()
    control = controls["lm0.3_wordpen0_beam16_pre64_cases"]
    original = control.hypotheses.creator
    version = hashlib.sha256(Path(dynamic_qwen_tuned.__file__).read_bytes() + original.code_version.encode()).hexdigest()
    prefix = "qwen-dynamic-transfer/dev-other"
    tokenization = dynamic_qwen_tuned.WordwiseQwenTokenizationJob(
        original.out_hypotheses, original.qwen_repo, code_version=version,
    )
    tokenization.add_alias(f"{prefix}/wordwise-tokenization")
    tk.register_output(f"{prefix}/wordwise-tokenization", tokenization.out_report)
    grid = [dict(scale=s, reward=r, penalty=p, formatting="cases", period=False)
            for s in (0.2, 0.3, 0.4) for r in (0.25, 0.5) for p in (0.0, -1.0)]
    grid += [dict(scale=0.3, reward=0.5, penalty=p, formatting="initial-capital", period=period)
             for p in (0.0, -1.0) for period in (False, True)]
    # Two numerical/behavioral controls for the new Python scorer implementation.
    grid += [dict(scale=0.3, reward=0, penalty=p, formatting="cases", period=False) for p in (0.0, -1.0)]
    outputs = {}
    for row in grid:
        config = copy.deepcopy(original.rasr_config.creator.config)
        search = config.lib_rasr.search_algorithm
        search.llm_scale = row["scale"]
        search.word_penalty = row["penalty"]
        search.llm.type = "qwen-dynamic-tuned"
        search.llm.outer_lm_scale = row["scale"]
        search.llm.token_reward = row["reward"]
        search.llm.formatting = row["formatting"]
        search.llm.terminal_period = row["period"]
        name = (f"lm{row['scale']:g}_tokreward{row['reward']:g}_wordpen{row['penalty']:g}_"
                f"{row['formatting']}_period{int(row['period'])}_beam16_pre64")
        recognition = dynamic_qwen_tuned.TunedDynamicQwenJob(
            original.dataset, original.repo, original.qwen_repo, original.vocabulary, original.binary,
            WriteRasrConfigJob(config, RasrConfig()).out_config, original.gate, code_version=version,
        )
        score = ScoreBenchmarkJob(control.references, recognition.out_hypotheses, code_version=version)
        recognition.add_alias(f"{prefix}/{name}/recognition")
        score.add_alias(f"{prefix}/{name}/scoring")
        for suffix, path in (("hypotheses", recognition.out_hypotheses), ("runtime", recognition.out_runtime),
                             ("rasr.log", recognition.out_log), ("scoring", score.out_summary),
                             ("segments", score.out_segments)):
            tk.register_output(f"{prefix}/{name}/{suffix}", path)
        outputs[name] = score
    return outputs


def run_beam_tuning():
    """Increase search budgets at the selected fixed-casing operating point."""
    import copy
    from ...model_pipelines.common.dynamic_qwen_tuned import TunedDynamicQwenJob
    from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob

    selected = run_rescoring_transfer()[
        "lm0.3_tokreward0.5_wordpen-1_initial-capital_period0_beam16_pre64"
    ]
    original = selected.hypotheses.creator
    prefix = "qwen-dynamic-beam/dev-other"
    grid = [(beam, pre) for beam in (16, 32, 64) for pre in (64, 128, 256)] + [(128, 256)]
    outputs = {}
    for beam, pre in grid:
        name = f"lm0.3_tokreward0.5_wordpen-1_initial-capital_period0_beam{beam}_pre{pre}_thr14"
        if (beam, pre) == (16, 64):
            # Register the already completed winner, without a new job identity.
            recognition, score = original, selected
        else:
            config = copy.deepcopy(original.rasr_config.creator.config)
            search = config.lib_rasr.search_algorithm
            search.max_beam_size = [beam]
            search.pre_llm_max_beam_size = pre
            recognition = TunedDynamicQwenJob(
                original.dataset, original.repo, original.qwen_repo, original.vocabulary, original.binary,
                WriteRasrConfigJob(config, RasrConfig()).out_config, original.gate,
                code_version=original.code_version,
            )
            score = ScoreBenchmarkJob(selected.references, recognition.out_hypotheses,
                                      code_version=original.code_version)
            recognition.add_alias(f"{prefix}/{name}/recognition")
            score.add_alias(f"{prefix}/{name}/scoring")
        for suffix, path in (("hypotheses", recognition.out_hypotheses), ("runtime", recognition.out_runtime),
                             ("rasr.log", recognition.out_log), ("scoring", score.out_summary),
                             ("segments", score.out_segments)):
            tk.register_output(f"{prefix}/{name}/{suffix}", path)
        outputs[name] = score
    return outputs


def run_threshold_tuning():
    """Stage one: establish score thresholds at generous fixed beam caps."""
    import copy
    from ...model_pipelines.common.dynamic_qwen_tuned import TunedDynamicQwenJob
    from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob

    selected = run_rescoring_transfer()[
        "lm0.3_tokreward0.5_wordpen-1_initial-capital_period0_beam16_pre64"
    ]
    original = selected.hypotheses.creator
    prefix = "qwen-dynamic-threshold/dev-other"
    # Tune both pruning stages jointly: a restrictive pre-LLM threshold can
    # discard candidates before the LM ever gets a chance to score them.
    grid = [(final, pre) for final in (4, 6, 8, 10, 12) for pre in (4, 6, 8, 10, 12)]
    outputs = {}
    for final, pre in grid:
        config = copy.deepcopy(original.rasr_config.creator.config)
        search = config.lib_rasr.search_algorithm
        search.max_beam_size = [128]
        search.pre_llm_max_beam_size = 256
        search.score_threshold = [float(final)]
        search.pre_llm_score_threshold = float(pre)
        name = f"lm0.3_tokreward0.5_wordpen-1_initial-capital_period0_beam128_pre256_thr{final}_prethr{pre}"
        recognition = TunedDynamicQwenJob(
            original.dataset, original.repo, original.qwen_repo, original.vocabulary, original.binary,
            WriteRasrConfigJob(config, RasrConfig()).out_config, original.gate,
            code_version=original.code_version,
        )
        score = ScoreBenchmarkJob(selected.references, recognition.out_hypotheses,
                                  code_version=original.code_version)
        recognition.add_alias(f"{prefix}/{name}/recognition")
        score.add_alias(f"{prefix}/{name}/scoring")
        for suffix, path in (("hypotheses", recognition.out_hypotheses), ("runtime", recognition.out_runtime),
                             ("rasr.log", recognition.out_log), ("scoring", score.out_summary),
                             ("segments", score.out_segments)):
            tk.register_output(f"{prefix}/{name}/{suffix}", path)
        outputs[name] = score
    # Register the finished narrow reference without rerunning it.
    tk.register_output(f"{prefix}/reference-beam16-pre64/scoring", selected.out_summary)
    tk.register_output(f"{prefix}/reference-beam16-pre64/runtime", original.out_runtime)
    return outputs


def run_tightened_beam_tuning():
    """Stage two: tighten caps with the selected 6/6 thresholds fixed."""
    import copy
    from ...model_pipelines.common.dynamic_qwen_tuned import TunedDynamicQwenJob
    from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob

    selected = run_rescoring_transfer()[
        "lm0.3_tokreward0.5_wordpen-1_initial-capital_period0_beam16_pre64"
    ]
    original = selected.hypotheses.creator
    prefix = "qwen-dynamic-beam-thr6/dev-other"
    # Independent axes locate which cap limits quality; joint points check
    # practical smaller operating points without a large Cartesian product.
    grid = [(128, 256)]
    grid += [(beam, 256) for beam in (1, 2, 4, 8, 16, 32, 64)]
    grid += [(128, pre) for pre in (4, 8, 16, 32, 64, 128)]
    grid += [(4, 16), (8, 32), (16, 64), (32, 128)]
    outputs = {}
    for beam, pre in grid:
        config = copy.deepcopy(original.rasr_config.creator.config)
        search = config.lib_rasr.search_algorithm
        search.max_beam_size = [beam]
        search.pre_llm_max_beam_size = pre
        search.score_threshold = [6.0]
        search.pre_llm_score_threshold = 6.0
        name = f"lm0.3_tokreward0.5_wordpen-1_initial-capital_period0_beam{beam}_pre{pre}_thr6_prethr6"
        recognition = TunedDynamicQwenJob(
            original.dataset, original.repo, original.qwen_repo, original.vocabulary, original.binary,
            WriteRasrConfigJob(config, RasrConfig()).out_config, original.gate,
            code_version=original.code_version,
        )
        score = ScoreBenchmarkJob(selected.references, recognition.out_hypotheses,
                                  code_version=original.code_version)
        # 128/256 reconstructs the completed threshold-6/6 job exactly.
        if (beam, pre) != (128, 256):
            recognition.add_alias(f"{prefix}/{name}/recognition")
            score.add_alias(f"{prefix}/{name}/scoring")
        for suffix, path in (("hypotheses", recognition.out_hypotheses), ("runtime", recognition.out_runtime),
                             ("rasr.log", recognition.out_log), ("scoring", score.out_summary),
                             ("segments", score.out_segments)):
            tk.register_output(f"{prefix}/{name}/{suffix}", path)
        outputs[name] = score
    return outputs
