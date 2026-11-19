"""Development sweeps with each model taking a turn as the first pass."""

import json
from pathlib import Path

from sisyphus import Job, Task, tk

from . import bidirectional_rescoring as runtime
from .experiment_context import ExperimentContext
from .job_reports import register_job_report
from .public_pilots import analysis_version, completed, frozen_protocol, sha


N = 16
TO_QWEN = (0.0, 0.15, 0.3, 0.5)
FROM_QWEN = (0.0, 0.1, 0.25, 0.5)
MODELS = (
    ("granite-ctc", "Granite CTC", "ctc"),
    ("parakeet-rnnt", "Parakeet RNN-T", "rnnt"),
    ("parakeet-tdt", "Parakeet TDT v3", "tdt"),
    ("fluid-speech", "Fluid Speech Pro", "tdt"),
    ("cohere", "Cohere Transcribe", "aed"),
)
FLUID = ("FluidVoice-Labs/fluid-speech-pro", "46027a3bffc5817ce8f9b29add0a062610134a90")
COHERE = ("CohereLabs/cohere-transcribe-03-2026", "b1eacc2686a3d08ceaae5f24a88b1d519620bc09")


class BidirectionalReportJob(Job):
    def __init__(self, rows, coverage, generation, *, code_version):
        self.rows, self.coverage, self.generation = rows, coverage, generation
        self.code_version = code_version
        self.out_report = self.output_path("report.md")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        def load(path):
            return json.loads(Path(path.get_path()).read_text())

        summary = [dict(row, score=load(row["score"])) for row in self.rows]
        coverage = [dict(row, metrics=load(row["metrics"])) for row in self.coverage]
        generation = [dict(row, metrics=load(row["metrics"])) for row in self.generation]
        lines = [
            "# Bidirectional Speech Qwen 1.7B rescoring",
            "",
            f"WER in %. Up to {N} distinct transcripts per utterance. Scale 0 is the generator-only baseline.",
            "Oracle selects by reference errors. Positive scales select by first-pass cost plus scaled auxiliary cost.",
            "Reverse scores use CTC alignment sums, RNN-T/TDT best alignments, or Cohere teacher-forced costs.",
            "Only the auxiliary score uses model-specific casing and punctuation. WER uses the generator's text.",
            "The Cohere runs use one encoder context per development utterance, with a 45-second threshold.",
            "When no candidate has a finite auxiliary score, selection uses the first-pass result.",
            "",
        ]
        for dataset in dict.fromkeys(row["dataset"] for row in summary):
            lines.extend(
                (
                    f"## {dataset}",
                    "",
                    "| Generator | Rescorer | Scale | WER | Ins / Del / Sub |",
                    "|---|---|---:|---:|---:|",
                )
            )
            for row in summary:
                if row["dataset"] == dataset:
                    score = row["score"]
                    lines.append(
                        f"| {row['generator']} | {row['rescorer']} | {row['condition']} | "
                        f"{score['wer_percent']:.3f} | {score['ins']} / {score['del']} / {score['sub']} |"
                    )
            lines.extend(
                (
                    "",
                    "| Rescorer on Speech Qwen output | Scoreable candidates | Segments back to first pass | Scoring RTF |",
                    "|---|---:|---:|---:|",
                )
            )
            for row in coverage:
                if row["dataset"] == dataset:
                    metrics = row["metrics"]
                    lines.append(
                        f"| {row['model']} | {metrics['scoreable']}/{metrics['candidates']} | "
                        f"{metrics['segments_with_no_scoreable_candidates']} | {metrics['rtf']:.3f} |"
                    )
            lines.append("")
            lines.extend(
                (
                    "| New generator | Mean distinct candidates | Empty first-best outputs | Generation RTF |",
                    "|---|---:|---:|---:|",
                )
            )
            for row in generation:
                if row["dataset"] == dataset:
                    metrics = row["metrics"]
                    lines.append(
                        f"| {row['model']} | {metrics['mean_candidates']:.2f} | "
                        f"{metrics['empty_first_pass']} | {metrics['rtf']:.3f} |"
                    )
            lines.append("")
        Path(self.out_report.get_path()).write_text("\n".join(lines))
        Path(self.out_summary.get_path()).write_text(
            json.dumps(dict(rows=summary, coverage=coverage, generation=generation), indent=2) + "\n"
        )


def run(output_root):
    from i6_core.rasr.config import WriteRasrConfigJob
    from i6_core.tools.compile import CMakeJob
    from i6_core.tools.git import CloneGitRepositoryJob

    from ...data import huggingface
    from . import (
        benchmark_scoring,
        hf_benchmark,
        parakeet_tdt,
        parakeet_tdt_experiments,
        qwen_speech,
        speech_qwen_round,
    )
    from .benchmark_scoring import ScoreBenchmarkJob
    from .speech_qwen_round import OracleSpeechNbestJob, ScoreSpeechNbestJob

    protocol, _, digest = frozen_protocol(output_root)
    version = analysis_version(
        runtime, parakeet_tdt, parakeet_tdt_experiments, hf_benchmark, qwen_speech, speech_qwen_round,
        compatible_modules=(parakeet_tdt_experiments,),
    )
    scoring_version = analysis_version(benchmark_scoring)
    prefix = "bidirectional-rescoring/" + digest[:16]
    speech = "Speech Qwen 1.7B"
    selected = protocol["selected"]
    dev = selected[f"{speech} | None | standalone | fast"]["dev"]
    stages = {
        dataset: next(s["parameters"] for s in record["stages"] if s["job_type"] == "SpeechQwenBeamJob")
        for dataset, record in dev.items()
    }
    (speech_repo,) = {s["speech_repo"] for s in stages.values()}
    speech_repo = completed(speech_repo)
    gate_dataset = completed(stages["librispeech-dev-other"]["dataset"])

    def download(model, revision, alias, patterns):
        job = huggingface.DownloadHuggingFaceRepoJob(model, revision=revision, allow_patterns=patterns)
        job.add_alias(alias)
        return job.out_repo_dir

    with ExperimentContext(prefix + "/tools"):
        patterns = ["*.json", "*.safetensors", "*.model", "*.txt"]
        stock = download(parakeet_tdt_experiments.MODEL, parakeet_tdt_experiments.REVISION, "parakeet-tdt", patterns)
        fluid = download(*FLUID, "fluid-speech", patterns)
        cohere = download(*COHERE, "cohere", ["*.json", "*.safetensors", "*.txt", "*.model"])
        checkout = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/rasr.git",
            commit=parakeet_tdt_experiments.RASR_COMMIT,
            checkout_folder_name="rasr",
        )
        checkout.add_alias("rasr-source")
        build = CMakeJob(
            source_folder=checkout.out_repository,
            num_processes=8,
            mem_rqmt=8,
            cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMODULE_ONNX=On", "-DMARCH=x86-64-v3"],
        )
        build.add_alias("rasr-cmake")
        tdt_configs = {}
        for name, repo, eos in (("parakeet-tdt", stock, parakeet_tdt.EOS), ("fluid-speech", fluid, None)):
            with ExperimentContext(name + "/search"):
                export = parakeet_tdt.ExportParakeetTdtJob(
                    repo, gate_dataset, version=analysis_version(parakeet_tdt, hf_benchmark)
                )
                export.add_alias("onnx-export")
                config = parakeet_tdt_experiments.recognition_config(export, beam=32, kind="am", eos=eos)
                config.lib_rasr.search_algorithm.score_threshold = [14.0]
                config = WriteRasrConfigJob(config, None).out_config
                tdt_configs[name] = dict(
                    binary=build.out_install_dir, config=config, gate=export.out_gate, vocab=export.out_vocab, eos=eos
                )
                tk.register_output("onnx-parity.json", export.out_gate)
    repos = {"parakeet-tdt": stock, "fluid-speech": fluid, "cohere": cohere}

    rows, coverage, generation_metrics = [], [], []
    for dataset in protocol["dev_datasets"]:
        record = dev[dataset]
        if record["status"] != "complete" or sha(record["references"]) != record["reference_sha256"]:
            raise ValueError("Frozen development references changed")
        refs = completed(record["references"])
        audio = completed(stages[dataset]["dataset"])

        def score(hyp, generator, rescorer, condition, alias):
            with ExperimentContext(alias):
                job = ScoreBenchmarkJob(refs, hyp, code_version=scoring_version)
                job.add_alias("scoring")
                tk.register_output("hypotheses.jsonl", hyp)
                tk.register_output("summary.json", job.out_summary)
                rows.append(
                    dict(
                        dataset=dataset,
                        generator=generator,
                        rescorer=rescorer,
                        condition=condition,
                        score=job.out_summary,
                    )
                )

        def oracle(candidates, generator):
            job = OracleSpeechNbestJob(refs, candidates, n=N, code_version=version)
            score(job.out_hypotheses, generator, "—", "Oracle", "oracle")

        def select(candidates, generator, rescorer, scales, policy=None, *, baseline=True):
            job = runtime.SelectBidirectionalJob(
                candidates, n=N, scales=scales, speech_policy=policy, code_version=version
            )
            job.add_alias("selection")
            for scale in scales:
                if scale or baseline:
                    score(
                        job.out_hypotheses[scale],
                        generator,
                        rescorer if scale else "—",
                        f"{scale:g}",
                        f"scale-{scale:g}",
                    )

        with ExperimentContext(f"{prefix}/{dataset}"):
            with ExperimentContext("speech-qwen-primary"):
                qwen = runtime.GenerateFusionNbestJob(audio, speech_repo, family="qwen", n=N, code_version=version)
                qwen.add_alias("generate")
                tk.register_output("candidates.jsonl", qwen.out_candidates)
                tk.register_output("generation-runtime.json", qwen.out_runtime)
                generation_metrics.append(dict(dataset=dataset, model=speech, metrics=qwen.out_runtime))
                oracle(qwen.out_candidates, speech)
                select(qwen.out_candidates, speech, "—", (0.0,))

            for slug, model, family in MODELS:
                with ExperimentContext(f"{slug}-primary"):
                    if family in ("ctc", "rnnt"):
                        entry = selected[f"{model} | {speech} | two-pass | fast"]["dev"][dataset]
                        if entry["status"] != "complete" or entry["reference_sha256"] != record["reference_sha256"]:
                            raise ValueError(f"Cached {model} rescoring has a different reference partition")
                        generation = next(
                            s["parameters"]
                            for s in entry["stages"]
                            if s["job_type"] in ("CompactGraniteNbestJob", "RnntSpeechNbestJob")
                        )
                        old = next(s["parameters"] for s in entry["stages"] if s["job_type"] == "SelectSpeechNbestJob")
                        auxiliary = next(
                            s["parameters"] for s in entry["stages"] if s["job_type"] == "ScoreSpeechNbestJob"
                        )
                        if (
                            completed(generation["dataset"]) != audio
                            or completed(auxiliary["speech_repo"]) != speech_repo
                        ):
                            raise ValueError(f"Cached {model} candidate audio or Speech Qwen checkpoint differs")
                        repo, scored, policy = (
                            completed(generation["repo"]),
                            completed(old["candidates"]),
                            old["policy"],
                        )
                        candidates = scored
                    else:
                        repo = repos[slug]
                        policy = "initial-capital-period" if family == "aed" else "original"
                        job = runtime.GenerateFusionNbestJob(
                            audio, repo, family=family, n=N, code_version=version, rasr=tdt_configs.get(slug)
                        )
                        job.add_alias("generate")
                        tk.register_output("candidates.jsonl", job.out_candidates)
                        tk.register_output("generation-runtime.json", job.out_runtime)
                        generation_metrics.append(dict(dataset=dataset, model=model, metrics=job.out_runtime))
                        candidates = job.out_candidates
                        speech_score = ScoreSpeechNbestJob(
                            audio, candidates, speech_repo, code_version=version, batch_size=4, policies=(policy,)
                        )
                        speech_score.add_alias("rescore-with-speech-qwen")
                        tk.register_output("rescoring-runtime.json", speech_score.out_runtime)
                        scored = speech_score.out_scores
                    oracle(candidates, model)
                    select(scored, model, speech, TO_QWEN, policy)

                with ExperimentContext("speech-qwen-primary/rescore-with-" + slug):
                    fixed = runtime.ScoreFixedFusionJob(
                        audio, qwen.out_candidates, repo, family=family, code_version=version
                    )
                    fixed.add_alias("rescore")
                    tk.register_output("rescored-candidates.jsonl", fixed.out_scores)
                    tk.register_output("rescoring-runtime.json", fixed.out_runtime)
                    coverage.append(dict(dataset=dataset, model=model, metrics=fixed.out_runtime))
                    select(fixed.out_scores, speech, model, FROM_QWEN[1:], baseline=False)

    with ExperimentContext(prefix):
        report = BidirectionalReportJob(
            rows, coverage, generation_metrics, code_version=analysis_version(runtime, benchmark_scoring)
        )
        report.add_alias("report")
        tk.register_output("summary.json", report.out_summary)
        tk.register_output("report.md", report.out_report)
    register_job_report("reports/asr-lm/bidirectional-rescoring.md", report.out_report, title="Bidirectional rescoring")
    return report
