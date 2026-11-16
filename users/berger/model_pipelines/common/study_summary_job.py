"""CPU-only publication of the frozen study tables; no inference dependencies."""

from pathlib import Path

from sisyphus import Job, Task, tk

from . import study_summary
from .job_reports import register_job_report


class ConsolidatePublicStudyJob(Job):
    def __init__(self, comparison, protocol, diagnostics):
        self.comparison = comparison
        self.protocol = protocol
        self.diagnostics = diagnostics
        self.code_version = study_summary.digest(study_summary.__file__)
        self.out_dir = self.output_path("study", directory=True)
        self.out_files = {name: self.output_path("study/" + name)
                          for name in ("tables.md", "metrics.md", "diagnostics.md", "attribution.md", "results.tsv", "provenance.json")}

    @classmethod
    def hash(cls, kwargs):
        return super().hash(dict(kwargs, code_version=study_summary.digest(study_summary.__file__)))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        study_summary.consolidate(self.comparison.get_path(), self.protocol.get_path(),
                                  {d: {k: p.get_path() for k, p in files.items()}
                                   for d, files in self.diagnostics.items()}, self.out_dir.get_path())


def register_public_study(output_root):
    comparison, protocol, diagnostics = study_summary.input_files(output_root)
    # Resolve completed output symlinks to immutable producer paths. Do not
    # reconstruct their recognition graphs just to publish small summaries.
    path = lambda p: tk.Path(str(Path(p).resolve(strict=True)))
    job = ConsolidatePublicStudyJob(path(comparison), path(protocol),
                                    {d: {k: path(p) for k, p in files.items()} for d, files in diagnostics.items()})
    job.add_alias("public-study/consolidation")
    tk.register_output("public-study", job.out_dir)
    for name in ("tables.md", "metrics.md", "diagnostics.md", "attribution.md"):
        register_job_report("reports/asr-lm/public-study-" + name, job.out_files[name], title="Frozen public study")
    return job
