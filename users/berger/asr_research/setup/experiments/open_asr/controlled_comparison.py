"""Step 1: freeze independent dev/eval evidence and register CPU-only reports."""

from pathlib import Path

from sisyphus import gs, tk

from ...model_pipelines.common import controlled_comparison as analysis
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.job_reports import register_job_report

_RUNS = {}


def run(*, output_root=None, alias_root=None, bootstrap_samples=10000):
    root = Path(output_root or gs.OUTPUT_DIR).absolute()
    aliases = Path(alias_root or gs.ALIAS_DIR).absolute()
    key = (id(tk.sis_graph), str(root), str(aliases), bootstrap_samples, gs.ALIAS_AND_OUTPUT_SUBDIR)
    if key in _RUNS:
        return _RUNS[key]
    records = analysis.scan_exports(root, aliases)
    version = analysis.analysis_version()
    with ExperimentContext("controlled-comparison"):
        dev = analysis.SnapshotControlledResultsJob([r for r in records if r["dataset"] in analysis.DEV],
                                                   partition="dev", code_version=version)
        evaluation = analysis.SnapshotControlledResultsJob([r for r in records if r["dataset"] not in analysis.DEV],
                                                          partition="evaluation", code_version=version)
        select = analysis.SelectControlledConfigsJob(dev.out_snapshot, code_version=version)
        report = analysis.ControlledComparisonJob(select.out_protocol, evaluation.out_snapshot,
                                                  bootstrap_samples=bootstrap_samples, code_version=version)
        for label, job in (("dev-evidence", dev), ("evaluation-evidence", evaluation),
                           ("dev-selection", select), ("paired-comparison", report)):
            job.add_alias(label)
        for label, path in (("dev-snapshot.json", dev.out_snapshot), ("evaluation-snapshot.json", evaluation.out_snapshot),
                            ("selection.json", select.out_protocol), ("comparison.md", report.out_report),
                            ("comparison.json", report.out_results), ("comparison.tsv", report.out_tsv),
                            ("missing-evaluations.json", report.out_missing)):
            tk.register_output(label, path)
    register_job_report("reports/asr-lm/controlled-comparison.md", report.out_report,
                        title="Controlled pretrained-ASR comparisons")
    result = dict(dev=dev, evaluation=evaluation, selection=select, report=report)
    _RUNS[key] = result
    return result
