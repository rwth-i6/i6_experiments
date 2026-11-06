from pathlib import Path

from sisyphus import Job, Task, tk

from diagnostics.compress_rasr_logs import main


class CompressRasrLogsJob(Job):
    def __init__(self, jobs_root: tk.Path, count: int | None = 100, workers: int = 4):
        self.jobs_root = jobs_root
        self.count = count
        self.workers = workers
        self.out_report = self.output_path("migration_report", directory=True)
        self.rqmt = {"cpu": workers, "mem": 2, "time": 24, "gpu": 0}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        directory = Path(self.out_report.get_path())
        directory.mkdir(exist_ok=True)
        main([
            "--jobs-root", self.jobs_root.get_path(),
            "--workers", str(self.workers),
            "--resume", str(directory),
            "--in-place",
        ] + (["--count", str(self.count)] if self.count is not None else []))


def py():
    job = CompressRasrLogsJob(
        jobs_root=tk.Path("/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/forward"),
        count=None,
        workers=4,
    )
    job_2 = CompressRasrLogsJob(
        jobs_root=tk.Path("/work/smt4/azim.javed/sisyphus_work/i6_experiments/users/azim_javed/common/jobs/returnn/"),
        count=None,
        workers=4,
    )
    job.add_alias("diagnostics/compress_rasr_logs")
    tk.register_output("diagnostics/compress_rasr_logs", job.out_report)
    job_2.add_alias("diagnostics/compress_rasr_logs_2")
    tk.register_output("diagnostics/compress_rasr_logs_2", job_2.out_report)
    return job, job_2
