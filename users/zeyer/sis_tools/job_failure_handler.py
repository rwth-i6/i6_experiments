"""
Use this for on_job_failure::

    def on_job_failure(job: Job):
        from i6_experiments.users.zeyer.sis_tools import job_failure_handler

        return job_failure_handler.get_cached_job_failure_handler()(job)

See https://github.com/rwth-i6/sisyphus/pull/205.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple
import os
import time
from ast import literal_eval

from sisyphus import Job, Task, tk
from sisyphus.engine import EngineBase, EngineSelector
from sisyphus.simple_linux_utility_for_resource_management_engine import SimpleLinuxUtilityForResourceManagementEngine

from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJobV2


def get_cached_job_failure_handler():
    """
    Creates if not exists, otherwise return existing
    """
    global _global_job_failure_handler
    if not _global_job_failure_handler:
        _global_job_failure_handler = JobFailureHandler()
    return _global_job_failure_handler


_global_job_failure_handler: Optional[JobFailureHandler] = None


class JobFailureHandler:
    def __init__(self):
        self.cached_engine = tk.cached_engine()
        self.failed_hosts = {}  # hostname -> most recent failure time

    def __call__(self, job: Job):
        # Filter out jobs. Currently only ReturnnTrainingJob on GPU with a specific GPU error.
        if isinstance(job, (ReturnnTrainingJob, ReturnnForwardJobV2)) and job.rqmt["gpu"] > 0:
            if not _is_returnn_cuda_error(_get_returnn_log_filename(job)):
                return  # do nothing
        else:
            return  # do nothing

        # Register failure.
        try:
            task, task_id = _get_failed_task(job)
        except _FailedTaskNotFoundError:
            print(f"{job} failed but no failed task found, maybe already cleaned, skipping...")
            return
        usage_file = task.get_process_logging_path(task_id)
        last_usage = literal_eval(open(usage_file).read())
        failed_host = last_usage["host"]
        print(f"{job} failed on {failed_host}, excluding host, clearing error...")
        self._clean_failed_hosts()
        self.failed_hosts[failed_host] = time.time()

        # Update Slurm engine default rqmt with maybe added excluded hosts.
        engine = _get_slurm_engine(cached_engine=self.cached_engine, task=task)
        self._update_slurm_engine_default_rqmt(engine=engine)

        # Cleanup (triggers restart)
        self._cleanup_job(job)

    def _clean_failed_hosts(self):
        """
        Remove hosts where the last failure was more than 24h ago
        """
        now = time.time()
        for host, last_failure in list(self.failed_hosts.items()):
            if now - last_failure > 24 * 60 * 60:
                del self.failed_hosts[host]

    def _update_slurm_engine_default_rqmt(self, engine: SimpleLinuxUtilityForResourceManagementEngine):
        if engine.default_rqmt.get("sbatch_args"):
            sbatch_args = list(engine.default_rqmt["sbatch_args"])
        else:
            sbatch_args = []
        exclude_hosts = ",".join([host.split(".")[0] for host in self.failed_hosts.keys()])
        if "-x" in sbatch_args:
            i = sbatch_args.index("-x")
            assert i + 1 < len(sbatch_args)
            if exclude_hosts:
                sbatch_args[i + 1] = exclude_hosts
            else:
                del sbatch_args[i : i + 2]
        elif exclude_hosts:
            sbatch_args.extend(["-x", exclude_hosts])
        engine.default_rqmt["sbatch_args"] = sbatch_args

    def _cleanup_job(self, job: Job):
        assert isinstance(job, (ReturnnTrainingJob, ReturnnForwardJobV2))  # only implemented for this case currently...
        base_path = job._sis_path()
        for fn in [base_path + "/error.run.1", base_path + "/submit_log.run"]:
            os.remove(fn)


def _get_returnn_log_filename(job: Union[ReturnnTrainingJob, ReturnnForwardJobV2]) -> str:
    return job._sis_path() + "/log.run.1"


def _is_returnn_cuda_error(log_filename: str) -> bool:
    if not os.path.exists(log_filename):
        return False
    # Example error (all a single line, wrapped here):
    #   torch.cuda.init() failed: RuntimeError Unexpected error from cudaGetDeviceCount().
    #   Did you run some cuda functions before calling NumCudaDevices() that might have already set an error?
    #   Error 805: MPS client failed to connect to the MPS control daemon or the MPS server
    # Another example error:
    #   ERROR: torch.cuda.is_available(): Timeout handler after 30 seconds, killing proc 51656.
    # Open in binary mode to avoid some UTF8 encoding problems,
    # e.g. due to some escape codes.
    f = open(log_filename, "rb")
    # Seek to end - 10k bytes
    f.seek(0, os.SEEK_END)
    f.seek(max(0, f.tell() - 10_000), os.SEEK_SET)
    lines = f.readlines()
    for line in lines:
        if line.startswith(b"torch.cuda.init() failed:"):
            if b"MPS client" in line:
                return True
        if line.startswith(b"ERROR: torch.cuda.is_available(): Timeout handler"):
            return True
    return False


class _FailedTaskNotFoundError(Exception):
    pass


def _get_failed_task(job: Job) -> Tuple[Task, int]:
    for task in job._sis_tasks():
        task: Task
        for task_id in task.task_ids():
            if task.error(task_id):
                return task, task_id
    raise _FailedTaskNotFoundError(f"No failed task for job {job}?")


def _get_slurm_engine(
    *, cached_engine: Optional[EngineBase] = None, task: Optional[Task] = None
) -> SimpleLinuxUtilityForResourceManagementEngine:
    engine = cached_engine or tk.cached_engine()
    if isinstance(engine, EngineSelector):
        if task is None:
            engine = engine.engines[engine.default_engine]
        else:
            engine = engine.get_used_engine_by_rqmt(task.rqmt())
    assert isinstance(engine, SimpleLinuxUtilityForResourceManagementEngine)
    return engine
