"""Replay missing exact dev-selected configurations, never select on test WER.

Requires an already completed frozen selection artifact. Existing matching
results are reused; new inference jobs use the original reviewed runtime classes.
Only the fast operating point is scheduled. Unrestricted choices remain audit
references, not an excuse to launch expensive held-out sweeps.
"""

import hashlib
import importlib
from pathlib import Path

from sisyphus import gs, tk

from ...model_pipelines.common import controlled_comparison as cc
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob
from ...model_pipelines.common.job_reports import register_job_report
from . import scoring_version
from .runtime_budget import apply_runtime_budget


MODULES = {
    "HfBenchmarkRecogJob": "hf_benchmark", "TunedDynamicQwenJob": "dynamic_qwen_tuned",
    "DynamicGraniteQwenJob": "dynamic_llm_jobs", "QwenSpeechRecogJob": "qwen_speech",
    "SpeechQwenBeamJob": "pretrained_correction", "CorrectedGraniteSpeechJob": "pretrained_correction",
    "CorrectedGraniteQwenJob": "pretrained_correction", "CorrectedRnntQwenJob": "pretrained_correction",
    "RnntQwenRecogJob": "rnnt_llm", "RnntSpeechNbestJob": "speech_qwen_round",
    "CompactGraniteNbestJob": "qwen_benchmark", "ScoreSpeechNbestJob": "speech_qwen_round",
    "SelectSpeechNbestJob": "speech_qwen_round",
}
PATH_FIELDS = {"dataset", "repo", "speech_repo", "granite_repo", "qwen_repo", "auxiliary_repo", "vocabulary",
               "binary", "gate", "config", "rasr_config", "prior", "baseline", "speech_baseline", "candidates",
               "generation_runtime", "scoring_runtime"}


def instantiate(stage, **overrides):
    name = stage["job_type"]
    if name not in MODULES:
        raise ValueError(f"No audited replay implementation for {name}")
    for record in stage.get("files", {}).values():
        if cc.sha256(record["path"]) != record["sha256"]:
            raise ValueError("Frozen configuration file changed")
    module = importlib.import_module("i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common." + MODULES[name])
    kwargs = dict(stage["parameters"], **overrides)
    for key in PATH_FIELDS:
        if isinstance(kwargs.get(key), str):
            kwargs[key] = tk.Path(kwargs[key])
    return getattr(module, name)(**kwargs)


def make_entry(choice, dataset, *, score, segments, hypotheses, timing, references):
    source = choice["dev"][cc.DEV[0]]
    return dict({k: source[k] for k in ("model", "lm", "decoding", "lexicon", "group", "condition", "family")},
                dataset=dataset, score=score, segments=segments, hypotheses=hypotheses, runtime=timing,
                references=references, runtime_is_parent=False)


def run(protocol_file=None, *, datasets=None, output_root=None, alias_root=None, bootstrap_samples=10000):
    root = Path(output_root or gs.OUTPUT_DIR).absolute()
    # Resolve the output alias once: subsequent alias changes cannot change
    # this graph's selected protocol identity or inference dependencies.
    protocol_path = Path(protocol_file or root / "controlled-comparison/selection.json").resolve(strict=True)
    protocol = cc.read_json(protocol_path)
    if not protocol["ready"]:
        raise ValueError("Development selection is provisional; finish pending candidate runs first")
    digest = cc.sha256(protocol_path)
    frozen = tk.Path(str(protocol_path))
    if datasets is None:
        from .qwen import run as existing_suite
        # Reconstructs existing standard jobs only; no new tuning conditions.
        data = existing_suite(loquacious=True, two_pass=True)["data"]
        datasets = {k: v for k, v in data.items() if k not in cc.DEV and not k.startswith("loquacious-dev")}
    if set(datasets) & set(cc.DEV) or any(k.startswith("loquacious-dev") for k in datasets):
        raise ValueError("Selected evaluation must not contain development data")
    existing = cc.scan_exports(root, Path(alias_root or gs.ALIAS_DIR).absolute())
    lookup = {(r["group"], r["condition"], r["dataset"]): r for r in existing if r["status"] == "complete"}
    choices = [c for k, c in protocol["selected"].items() if k.endswith(" | fast")]
    choices.sort(key=lambda c: (c["dev"][cc.DEV[0]]["lm"] != "None", c["family"]))
    entries, jobs, baselines, generations = [], [], {}, {}
    prefix = f"controlled-evaluation/{digest[:16]}"

    def append_entry(choice, name, prepared, entry):
        entries.append(entry)
        if name != "loquacious-test":
            return
        from ...data.loquacious_benchmark import LoquaciousSubsetJob, FILTERS
        slug = choice["family"].replace(" | ", "__").replace(" ", "-")
        for subset in FILTERS:
            subset_name = name + "." + subset
            cached = lookup.get((choice["group"], choice["condition"], subset_name))
            if cached:
                sub = make_entry(choice, subset_name, score=tk.Path(cached["score"]), segments=tk.Path(cached["segments"]),
                    hypotheses=tk.Path(cached["hypotheses"]), timing=entry["runtime"], references=tk.Path(cached["references"]))
            else:
                with ExperimentContext(f"{prefix}/{slug}/{subset_name}"):
                    filtered = LoquaciousSubsetJob(prepared.out_references, entry["hypotheses"], subset,
                                                    manifest=prepared.out_manifest)
                    score = ScoreBenchmarkJob(filtered.out_references, filtered.out_hypotheses, code_version=scoring_version())
                    score.add_alias("scoring")
                    tk.register_output("scoring", score.out_summary)
                    tk.register_output("segments", score.out_segments)
                    sub = make_entry(choice, subset_name, score=score.out_summary, segments=score.out_segments,
                        hypotheses=filtered.out_hypotheses, timing=entry["runtime"], references=filtered.out_references)
            sub["runtime_is_parent"] = True
            entries.append(sub)

    def register(choice, name, prepared, job):
        # Job construction happens inside the selected family's scope below.
        apply_runtime_budget(job, name)
        job.rqmt.update(gpu_mem=48, mem=96)
        job.add_alias("recognition")
        score = ScoreBenchmarkJob(prepared.out_references, job.out_hypotheses, code_version=scoring_version())
        score.add_alias("scoring")
        for label, value in (("hypotheses", job.out_hypotheses), ("scoring", score.out_summary),
                             ("segments", score.out_segments), ("runtime", job.out_runtime)):
            tk.register_output(label, value)
        if hasattr(job, "out_log"):
            tk.register_output("rasr.log", job.out_log)
        return make_entry(choice, name, score=score.out_summary, segments=score.out_segments,
                          hypotheses=job.out_hypotheses, timing=job.out_runtime, references=prepared.out_references)

    for choice in choices:
        model, auxiliary, method = choice["family"].split(" | ")
        slug = choice["family"].replace(" | ", "__").replace(" ", "-")
        for name, prepared in datasets.items():
            current = lookup.get((choice["group"], choice["condition"], name))
            if current is not None:
                if current["normalizer"] != cc.NORMALIZER:
                    raise ValueError("Existing selected evaluation uses a different normalizer")
                entry = make_entry(choice, name, **dict(score=tk.Path(current["score"]), segments=tk.Path(current["segments"]),
                    hypotheses=tk.Path(current["hypotheses"]), timing=tk.Path(current["runtime"]) if current.get("runtime") else None,
                    references=tk.Path(current["references"])))
                append_entry(choice, name, prepared, entry)
                if auxiliary == "None":
                    baselines[model, name] = entry["score"]
                continue
            source = choice["dev"][cc.DEV[1] if name.startswith("loquacious-") else cc.DEV[0]]
            with ExperimentContext(f"{prefix}/{slug}/{name}"):
                if method == "two-pass":
                    if not auxiliary.startswith("Speech Qwen"):
                        raise ValueError("Missing text-Qwen selected evaluation requires an explicit audited pipeline")
                    selection_stage = source["stages"][0]
                    scoring_stage = next(s for s in source["stages"] if s["job_type"] == "ScoreSpeechNbestJob")
                    gen_stage = next(s for s in source["stages"] if s["job_type"] in {"CompactGraniteNbestJob", "RnntSpeechNbestJob"})
                    gen_key = model, name, gen_stage["parameters"].get("rasr_config", gen_stage["parameters"].get("config"))
                    if gen_key not in generations:
                        cached = None
                        # Reuse exact completed Granite first-pass artifacts,
                        # rather than generating the same candidates again.
                        for row in existing:
                            if row["status"] != "complete" or row["dataset"] != name or row["model"] != model:
                                continue
                            for stage in row["stages"]:
                                if stage["job_type"] == gen_stage["job_type"] and stage["files"] == gen_stage["files"]:
                                    base = Path(stage["job"]) / "output"
                                    candidate = base / "candidates.jsonl"
                                    timing = base / ("generation.json" if model == "Granite CTC" else "runtime.json")
                                    if candidate.exists() and timing.exists():
                                        cached = tk.Path(str(candidate)), tk.Path(str(timing))
                                        break
                            if cached:
                                break
                        if cached:
                            generations[gen_key] = cached
                        else:
                            gen = instantiate(gen_stage, dataset=prepared.out_dataset)
                            apply_runtime_budget(gen, name)
                            gen.rqmt.update(gpu_mem=48, mem=96)
                            gen.add_alias("generation")
                            timing = gen.out_report if hasattr(gen, "out_report") else gen.out_runtime
                            generations[gen_key] = gen.out_candidates, timing
                            jobs.append(gen)
                    candidates, gen_runtime = generations[gen_key]
                    lm = instantiate(scoring_stage, dataset=prepared.out_dataset, candidates=candidates)
                    apply_runtime_budget(lm, name)
                    lm.add_alias("speech-scoring")
                    tk.register_output("candidates", candidates)
                    tk.register_output("speech-scores", lm.out_scores)
                    job = instantiate(selection_stage, candidates=lm.out_scores, generation_runtime=gen_runtime,
                                      scoring_runtime=lm.out_runtime)
                    # Selection is CPU-only; do not apply GPU resource overrides.
                    job.add_alias("selection")
                    score = ScoreBenchmarkJob(prepared.out_references, job.out_hypotheses, code_version=scoring_version())
                    score.add_alias("scoring")
                    for label, value in (("hypotheses", job.out_hypotheses), ("scoring", score.out_summary),
                                         ("segments", score.out_segments), ("runtime", job.out_runtime)):
                        tk.register_output(label, value)
                    entry = make_entry(choice, name, score=score.out_summary, segments=score.out_segments,
                                       hypotheses=job.out_hypotheses, timing=job.out_runtime, references=prepared.out_references)
                    jobs += [lm, job]
                else:
                    stage = source["stages"][0]
                    overrides = dict(dataset=prepared.out_dataset)
                    if auxiliary.startswith("Speech Qwen"):
                        if (auxiliary, name) not in baselines:
                            raise ValueError("Missing dev-selected standalone Speech-Qwen control")
                        overrides["speech_baseline" if model == "Parakeet RNN-T" else "baseline"] = baselines[auxiliary, name]
                    job = instantiate(stage, **overrides)
                    if not hasattr(job, "out_hypotheses") and hasattr(job, "out_first_pass"):
                        job.out_hypotheses, job.out_runtime = job.out_first_pass, job.out_report
                    entry = register(choice, name, prepared, job)
                    jobs.append(job)
                append_entry(choice, name, prepared, entry)
                if auxiliary == "None":
                    baselines[model, name] = entry["score"]
    version = hashlib.sha256(cc.analysis_version().encode() + Path(__file__).read_bytes()).hexdigest()
    with ExperimentContext(prefix):
        evidence = cc.CollectSelectedEvaluationJob(entries, code_version=version)
        report = cc.ControlledComparisonJob(frozen, evidence.out_snapshot, bootstrap_samples=bootstrap_samples,
                                             operating_points=("fast",), code_version=version)
        # This evaluation intentionally schedules only fast choices. Avoid
        # representing absent slow-reference cells as missing required jobs.
        tk.register_output("frozen-selection.json", frozen)
        tk.register_output("evaluation.json", evidence.out_snapshot)
        tk.register_output("comparison.md", report.out_report)
        tk.register_output("comparison.json", report.out_results)
        tk.register_output("comparison.tsv", report.out_tsv)
        register_job_report("reports/asr-lm/controlled-selected-evaluation.md", report.out_report,
                            title="Controlled selected evaluation")
    from .error_complementarity import register_selected
    complementarity = register_selected(frozen, evidence.out_snapshot, existing, {r["dataset"] for r in entries},
        output_root=root, alias_root=Path(alias_root or gs.ALIAS_DIR).absolute(), prefix=prefix + "/error-complementarity")
    return dict(jobs=jobs, entries=entries, evidence=evidence, report=report, protocol=str(protocol_path),
                complementarity=complementarity)
