"""CPU-only Steps 3/4; reuse frozen evidence, cached scores and audio lengths."""

import hashlib
from pathlib import Path

from sisyphus import gs, tk

from ...model_pipelines.common import controlled_comparison as cc
from ...model_pipelines.common import error_complementarity as ec
from ...model_pipelines.common import research_diagnostics as analysis
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.job_reports import register_job_report


def training_corpus_path(root, explicit=None):
    if explicit:
        source = Path(explicit).resolve(strict=True)
    else:
        # Reuse the already prepared training corpus, not an LM estimate or
        # evaluation-derived vocabulary. Do not schedule a new audio download.
        counts = Path(root) / "librispeech/datasets/LibriSpeech/lexicon_oov/training_word_frequencies"
        if not counts.exists():
            raise ValueError("Set ASR_RARITY_TRAINING_CORPUS to an existing train-other-960 Bliss XML")
        params = cc.parameters_flat(counts.resolve().parent.parent)
        value = params.get("bliss_corpus")
        if not isinstance(value, str):
            raise ValueError("Cannot verify training-corpus provenance")
        source = Path(value).resolve(strict=True)
    return tk.Path(str(source))


def run(
    *, which="both", output_root=None, alias_root=None, protocol_file=None, selected_evidence=None, training_corpus=None
):
    from .controlled_comparison import run as comparison
    from .error_complementarity import cohere_records, cohere_snapshot

    if which not in {"both", "ranking", "stratification"}:
        raise ValueError(which)
    root = Path(output_root or gs.OUTPUT_DIR).absolute()
    aliases = Path(alias_root or gs.ALIAS_DIR).absolute()
    source = comparison(output_root=root, alias_root=aliases)
    protocol = (
        tk.Path(str(Path(protocol_file).resolve(strict=True))) if protocol_file else source["selection"].out_protocol
    )
    version = hashlib.sha256(
        Path(analysis.__file__).read_bytes() + Path(ec.__file__).read_bytes() + cc.analysis_version().encode()
    ).hexdigest()
    result = dict(ranking={}, stratification={})
    if which in {"ranking", "both"}:
        for dataset in cc.DEV:
            with ExperimentContext("candidate-ranking/" + dataset):
                job = analysis.CandidateRankingDiagnosticsJob(
                    dataset, source["dev"].out_snapshot, protocol, code_version=version
                )
                job.add_alias("analysis")
                for label, value in (
                    ("summary.json", job.out_summary),
                    ("report.md", job.out_report),
                    ("units.jsonl.gz", job.out_units),
                    ("examples.jsonl", job.out_examples),
                ):
                    tk.register_output(label, value)
            register_job_report(
                "reports/asr-lm/candidate-ranking-" + dataset + ".md",
                job.out_report,
                title="Candidate coverage and ranking",
            )
            result["ranking"][dataset] = job
    if which in {"stratification", "both"}:
        extra = cohere_snapshot(cohere_records(root, aliases), "stratified-errors")
        snapshots = [source["dev"].out_snapshot, source["evaluation"].out_snapshot, extra.out_snapshot]
        if selected_evidence:
            snapshots.append(tk.Path(str(Path(selected_evidence).resolve(strict=True))))
        with ExperimentContext("stratified-errors/training-frequencies"):
            freq = analysis.TrainingWordFrequenciesJob(
                training_corpus_path(root, training_corpus), code_version=version
            )
            freq.add_alias("counts")
            tk.register_output("frequencies.json", freq.out_frequencies)
        records = source["dev"].records + source["evaluation"].records
        # A fixed native Granite run supplies audio lengths, independent of
        # evaluated model outputs. No waveform is loaded by these analyses.
        anchors = {
            r["dataset"]: r
            for r in records
            if r["group"] == "qwen-granite" and r["condition"] == "native" and r["status"] == "complete"
        }
        for dataset, r in sorted(anchors.items()):
            with ExperimentContext("stratified-errors/" + dataset):
                duration = analysis.ScoringUnitDurationsJob(
                    tk.Path(r["hypotheses"]), tk.Path(r["references"]), code_version=version
                )
                duration.add_alias("durations")
                tk.register_output("durations.json", duration.out_durations)
                job = analysis.StratifiedErrorsJob(
                    dataset, snapshots, protocol, duration.out_durations, freq.out_frequencies, code_version=version
                )
                job.add_alias("analysis")
                tk.register_output("summary.json", job.out_summary)
                tk.register_output("report.md", job.out_report)
            register_job_report(
                "reports/asr-lm/stratified-" + dataset + ".md", job.out_report, title="Stratified ASR errors"
            )
            result["stratification"][dataset] = job
        result["frequencies"] = freq
    return result
