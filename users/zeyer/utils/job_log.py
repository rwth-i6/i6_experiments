from typing import Optional, Union, Tuple, TextIO, Iterator
import os
import re
import tarfile
from io import TextIOWrapper
from sisyphus import Job
from contextlib import contextmanager
from .job_dir import get_job_base_dir


@contextmanager
def open_recent_job_log(
    job: Union[str, Job], task: str = "run", index: int = 1
) -> Tuple[Optional[TextIO], Optional[str]]:
    """
    Opens the job dir / log.<task>.<index> or the log inside the finished.tar.gz.
    This is the most recent log in case there were multiple restarts.
    Also see :func:`open_job_logs`.

    :param job: dir or Job object
    :param task:
    :param index:
    """
    job_dir = get_job_base_dir(job)
    log_base_fn = f"log.{task}.{index}"
    log_fn = f"{job_dir}/{log_base_fn}"
    if os.path.exists(log_fn):
        yield open(log_fn), log_fn
        return
    tar_fn = f"{job_dir}/finished.tar.gz"
    if os.path.exists(tar_fn):
        with tarfile.open(tar_fn) as tarf:
            while True:
                f = tarf.next()
                if not f:
                    break
                if f.name == log_base_fn:
                    f = tarf.extractfile(f)
                    yield TextIOWrapper(f), f"{tar_fn}:{log_base_fn}"
                    return
    yield None, None


def open_job_logs(job: Union[str, Job], task: str = "run", index: int = 1) -> Iterator[Tuple[TextIO, str]]:
    """
    Opens all logs of a job, iterating over the files in `engine`.
    Also checks the logs inside the finished.tar.gz.

    :param job: dir or Job object
    :param task:
    :param index:
    :return: generator of (file, filename). filename can also be a reference inside a tar file
    """
    job_dir = get_job_base_dir(job)
    log_fn_engine_pattern = rf"(.*)\.{re.escape(task)}\.(\d+)\.{index}"  # job_name.task.engine_id.index
    count = 0

    if os.path.exists(f"{job_dir}/engine"):
        for name in os.listdir(f"{job_dir}/engine"):
            if re.match(f"^{log_fn_engine_pattern}$", name):
                log_fn = f"{job_dir}/engine/{name}"
                yield open(log_fn), log_fn
                count += 1

    tar_fn = f"{job_dir}/finished.tar.gz"
    if os.path.exists(tar_fn):
        with tarfile.open(tar_fn) as tarf:
            while True:
                f = tarf.next()
                if not f:
                    break
                if re.match(f"^engine/{log_fn_engine_pattern}$", f.name):
                    f = tarf.extractfile(f)
                    yield TextIOWrapper(f), f"{tar_fn}:{f.name}"
                    count += 1

    assert count > 0, f"No logs found for {job} {task} {index}"
