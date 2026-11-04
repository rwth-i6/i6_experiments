"""Step 2: CPU-only error complementarity using fixed and dev-selected panels."""

import hashlib
from pathlib import Path

from sisyphus import gs, tk

from ...model_pipelines.common import controlled_comparison as cc
from ...model_pipelines.common import error_complementarity as analysis
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.job_reports import register_job_report


def cohere_records(output_root, alias_root):
    """Cohere's fixed native baseline; never pick an AED setting using test WER."""
    output = Path(output_root) / "open-asr/cohere-aed/native"
    aliases = Path(alias_root) / "open-asr/cohere-aed/native"
    records = []
    names = sorted({p.name for base in (output, aliases) if base.is_dir() for p in base.iterdir() if p.is_dir()})
    for name in names:
        if name in cc.DEV or name.startswith("loquacious-dev"):
            continue
        exported = output / name
        score = exported / "scoring"
        if not score.exists():
            score = aliases / name / "scoring/output/summary.json"
        if not score.exists():
            continue
        score = score.resolve()
        summary = cc.read_json(score)
        params = cc.parameters_flat(score.parent.parent)
        ref = params.get("references")
        hyp = exported / "hypotheses"
        if not hyp.exists() and isinstance(params.get("hypotheses"), str):
            hyp = Path(params["hypotheses"])
        if not ref or not Path(ref).exists() or not hyp.exists():
            continue
        runtime = exported / "runtime"
        stages = cc.provenance(hyp.resolve().parent.parent)
        records.append(dict(model="Cohere AED", lm="None", decoding="Native", lexicon="None",
            group="cohere-aed", condition="native", dataset=name, family="Cohere AED | None | standalone",
            status="complete", score=str(score), score_sha256=cc.sha256(score),
            segments=str(score.parent / "segments.jsonl"), normalizer=summary.get("normalizer_revision"),
            references=ref, reference_sha256=cc.sha256(ref), hypotheses=str(hyp.resolve()),
            runtime=str(runtime.resolve()) if runtime.exists() else None,
            runtime_sha256=cc.sha256(runtime.resolve()) if runtime.exists() else None,
            runtime_is_parent=False, stages=stages,
            dataset_paths=sorted({str(Path(s["parameters"]["dataset"]).resolve()) for s in stages
                                 if isinstance(s["parameters"].get("dataset"), str)})))
    return records


def register_analysis(snapshots, protocol, datasets, *, cohere=None, prefix="error-complementarity"):
    version = hashlib.sha256(Path(analysis.__file__).read_bytes() + cc.analysis_version().encode()).hexdigest()
    if cohere is not None:
        snapshots = [*snapshots, cohere]
    jobs = {}
    with ExperimentContext(prefix):
        for dataset in sorted(set(datasets)):
            job = analysis.ErrorComplementarityJob(dataset, snapshots, protocol, code_version=version)
            job.add_alias(dataset)
            jobs[dataset] = job
            for label, value in (("summary.json", job.out_summary), ("report.md", job.out_report),
                                 ("units.jsonl.gz", job.out_units), ("examples.jsonl", job.out_examples)):
                tk.register_output(dataset + "/" + label, value)
        overview = analysis.ComplementarityOverviewJob({k: v.out_summary for k, v in jobs.items()}, code_version=version)
        overview.add_alias("overview")
        tk.register_output("index.md", overview.out_report)
        tk.register_output("pairs.tsv", overview.out_pairs)
    report_name = "error-complementarity.md" if prefix == "error-complementarity" else "error-complementarity-selected.md"
    register_job_report("reports/asr-lm/" + report_name, overview.out_report, title="Model error complementarity")
    return dict(jobs=jobs, overview=overview)


def cohere_snapshot(records, prefix):
    version = cc.analysis_version()
    with ExperimentContext(prefix):
        job = cc.SnapshotControlledResultsJob(records, partition="evaluation", code_version=version)
        job.add_alias("fixed-cohere-evidence")
        tk.register_output("cohere-evidence.json", job.out_snapshot)
    return job


def run(*, output_root=None, alias_root=None, protocol_file=None, selected_evidence=None):
    from .controlled_comparison import run as comparison
    root = Path(output_root or gs.OUTPUT_DIR).absolute()
    aliases = Path(alias_root or gs.ALIAS_DIR).absolute()
    source = comparison(output_root=root, alias_root=aliases)
    extra = cohere_snapshot(cohere_records(root, aliases), "error-complementarity")
    protocol = tk.Path(str(Path(protocol_file).resolve(strict=True))) if protocol_file else source["selection"].out_protocol
    snapshots = [source["dev"].out_snapshot, source["evaluation"].out_snapshot]
    # Optional late-stage evidence is explicit and frozen, not a filesystem poll
    # that would silently substitute whichever test configuration scores best.
    if selected_evidence:
        snapshots.append(tk.Path(str(Path(selected_evidence).resolve(strict=True))))
    records = source["dev"].records + source["evaluation"].records + extra.records
    return register_analysis(snapshots, protocol, [r["dataset"] for r in records], cohere=extra.out_snapshot)


def register_selected(protocol, selected_snapshot, records, datasets, *, output_root, alias_root, prefix):
    """Called by the selected-evaluation graph; adds CPU analysis, never ASR jobs."""
    version = cc.analysis_version()
    with ExperimentContext(prefix):
        raw = cc.SnapshotControlledResultsJob([r for r in records if r["dataset"] not in cc.DEV],
                                             partition="evaluation", code_version=version)
    extra = cohere_snapshot(cohere_records(output_root, alias_root), prefix)
    return register_analysis([raw.out_snapshot, extra.out_snapshot, selected_snapshot], protocol, datasets, prefix=prefix)
