from typing import Union, IO, Tuple, Optional
import os
from contextlib import contextmanager
import tarfile

from sisyphus import Job

from .job_dir import get_job_base_dir


def read_job_file(job: Union[str, Job], filename: str) -> bytes:
    """
    Reads a file from a job dir or from the finished.tar.gz.

    :param job: dir or Job object
    :param filename: in the job dir, e.g. "job.save"
    """
    with open_job_file_io(job, filename) as (f, fn):
        return f.read()


@contextmanager
def open_job_file_io(job: Union[str, Job], filename: str) -> Tuple[Optional[IO], Optional[str]]:
    """
    Opens the job dir / filename or the filename inside the finished.tar.gz.

    :param job: dir or Job object
    :param filename: in the job dir, e.g. "job.save"
    """
    job_dir = get_job_base_dir(job)
    fn_ = f"{job_dir}/{filename}"
    if os.path.exists(fn_):
        yield open(fn_, "rb"), fn_
        return
    tar_fn = f"{job_dir}/finished.tar.gz"
    if os.path.exists(tar_fn):
        with tarfile.open(tar_fn) as tarf:
            while True:
                f = tarf.next()
                if not f:
                    break
                if f.name == filename:
                    f_ = tarf.extractfile(f)
                    assert f_ is not None
                    yield f_, tar_fn
    raise FileNotFoundError(f"File {filename} not found in {job_dir} or {tar_fn}")
