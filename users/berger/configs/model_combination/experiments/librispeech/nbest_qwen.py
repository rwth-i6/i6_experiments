"""Quick two-pass Qwen baseline on LibriSpeech dev-other, not the test suite."""

import hashlib
from pathlib import Path

from sisyphus import tk
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob


def run_preflight():
    """Run the existing accounting/pruning gates on the fixed OOV binary."""
    from ... import tools
    from ...data.oov_multitoken_smoke import OovMultiTokenDecoderSmokeJob
    from ...data.oov_length_smoke import (
        OovDelayedPenaltyProbeJob,
        OovFirstPiecePruningProbeJob,
        RequireOovPruningProbeJob,
    )

    binary = tools.get_nbest_fixed_rasr_binary_path()
    prefix = "tools/rasr-nbest-ff206f90"
    smoke = OovMultiTokenDecoderSmokeJob(binary, tools.returnn_python_exe)
    delayed = OovDelayedPenaltyProbeJob(binary, tools.returnn_python_exe)
    first = OovFirstPiecePruningProbeJob(binary, tools.returnn_python_exe)
    pruning = RequireOovPruningProbeJob(delayed.out_report, first_piece_report=first.out_report)
    for name, job in (
        ("accounting", smoke),
        ("delayed-penalty", delayed),
        ("first-piece", first),
        ("pruning-acceptance", pruning),
    ):
        job.add_alias(f"{prefix}/{name}")
        tk.register_output(f"{prefix}/{name}", job.out_report)
        if hasattr(job, "out_logs"):
            tk.register_output(f"{prefix}/{name}-logs", job.out_logs)
    return dict(binary=binary, correctness=smoke, pruning=pruning)


def run():
    from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
    from ..open_asr import MODELS, source_version, recognition_version, scoring_version
    from ..open_asr import _lexicon, _granite_open_lexicon, recognition_config
    from ...data.open_asr import BenchmarkDataset, PrepareOpenAsrDatasetJob
    from ...data.huggingface import DownloadHuggingFaceRepoJob
    from ...data.llm_fusion import QWEN_REPO, QWEN_REVISION
    from ...data.librispeech.lm import ArpaLmParams, get_word_lm_config
    from ...model_pipelines.common import nbest_rescoring
    from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob
    from ...model_pipelines.common import qwen_nbest_diagnostics as diagnostics
    from ...model_pipelines.common import qwen_full_dev


    prefix = "qwen-nbest/dev-other"
    version = hashlib.sha256(
        Path(nbest_rescoring.__file__).read_bytes() + recognition_version("ctc").encode() + scoring_version().encode()
    ).hexdigest()
    # HF's original dev-other split; no evaluation text is used for lexica.
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
    dataset.add_alias(f"{prefix}/data")
    tk.register_output(f"{prefix}/manifest", dataset.out_manifest)
    family, model_id, revision = MODELS["granite-ctc"]
    repo = DownloadHuggingFaceRepoJob(
        model_id, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
    ).out_repo_dir
    preflight = run_preflight()
    inventory = _lexicon(repo, family, get_bliss_lexicon(), source_version(), f"{prefix}/lexicon")
    lexicon = _granite_open_lexicon(repo, inventory, preflight, f"{prefix}/open-tree")
    qwen = DownloadHuggingFaceRepoJob(
        QWEN_REPO, revision=QWEN_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
    ).out_repo_dir
    result = {}
    for name, scale, beta in (("open-no-lm", 0.0, 1.0), ("open-4gram", 0.3, 6.0)):
        lm = (
            get_word_lm_config(lexicon, ArpaLmParams(scale=scale), rasr_binary_path_override=preflight["binary"])
            if scale
            else None
        )
        config = recognition_config(
            "ctc",
            inventory,
            lexicon=lexicon,
            lm=lm,
            beam=256,
            word_beam=256,
            threshold=16.0,
            fallback=True,
            beta=beta,
            alpha=2.0,
        )
        # RASR supports on/off, not a separate lattice recombination mode.
        # Its n-best API extracts paths from the surviving search lattice.
        generation = nbest_rescoring.GraniteNbestJob(
            dataset.out_dataset,
            repo,
            WriteRasrConfigJob(config, RasrConfig()).out_config,
            preflight["binary"],
            n=50,
            code_version=version,
        )
        scoring = nbest_rescoring.ScoreQwenNbestJob(generation.out_candidates, qwen, code_version=version)
        evaluation = nbest_rescoring.EvaluateQwenNbestJob(
            dataset.out_references, scoring.out_scores, code_version=version
        )
        first_pass = ScoreBenchmarkJob(dataset.out_references, generation.out_first_pass, code_version=version)
        for suffix, job in (
            ("generation", generation),
            ("qwen-scoring", scoring),
            ("evaluation", evaluation),
            ("first-pass-scoring", first_pass),
        ):
            job.add_alias(f"{prefix}/{name}/{suffix}")
        for suffix, output in (
            ("candidates", generation.out_candidates),
            ("generation", generation.out_report),
            ("rasr.log", generation.out_log),
            ("qwen-candidates", scoring.out_scores),
            ("lm-scoring", scoring.out_report),
            ("summary", evaluation.out_summary),
            ("hypotheses", evaluation.out_hypotheses),
            ("first-pass-score", first_pass.out_summary),
        ):
            tk.register_output(f"{prefix}/{name}/{suffix}", output)
        result[name] = evaluation
        full_dev = qwen_full_dev.EvaluateQwenFullDevJob(
            dataset.out_references, scoring.out_scores,
            code_version=hashlib.sha256(Path(qwen_full_dev.__file__).read_bytes() + version.encode()).hexdigest(),
        )
        full_dev.add_alias(f"{prefix}/{name}/full-dev-tuning")
        tk.register_output(f"{prefix}/{name}/full-dev-tuning/summary", full_dev.out_summary)
        tk.register_output(f"{prefix}/{name}/full-dev-tuning/best-hypotheses", full_dev.out_best)
        # Keep the original generation/scoring hashes stable: diagnostics have
        # their own version and reuse the completed lists and Qwen scores.
        diagnostic_version = hashlib.sha256(Path(diagnostics.__file__).read_bytes() + version.encode()).hexdigest()
        selected_policy = "initial-capital-period_scale0.2" if name == "open-no-lm" else "lowercase_scale0"
        examples = diagnostics.SelectQwenOracleExamplesJob(
            dataset.out_references, scoring.out_scores,
            evaluation.out_hypotheses.join_right(f"n50_{selected_policy}.jsonl"),
            evaluation.out_hypotheses.join_right("n50_oracle_scale0.jsonl"),
            code_version=diagnostic_version,
        )
        probes = diagnostics.ProbeQwenOracleExamplesJob(examples.out_candidates, qwen, code_version=diagnostic_version)
        analysis = diagnostics.AnalyzeQwenOracleProbesJob(examples.out_examples, probes.out_scores,
                                                        code_version=diagnostic_version)
        for suffix, job in (("examples", examples), ("probes", probes), ("analysis", analysis)):
            job.add_alias(f"{prefix}/{name}/diagnostics/{suffix}")
        for suffix, output in (("examples", examples.out_examples), ("cached-sweep", examples.out_cached_sweep),
                               ("probe-scores", probes.out_scores), ("probe-sweep", analysis.out_report)):
            tk.register_output(f"{prefix}/{name}/diagnostics/{suffix}", output)
    return result
