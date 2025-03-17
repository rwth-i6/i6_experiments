from typing import Optional, List
import os
import re
import tarfile
from glob import glob
from .job_dir import get_job_base_dir


def get_job_aliases(job: str) -> Optional[List[str]]:
    """
    Read the job aliases from the log files.
    Also looks into log files inside the finished.tar.gz file if it exists.

    Note: This is not necessarily needed. We can also read the aliases from the job info file.
    See :func:`job_aliases_from_info.get_job_aliases`.

    :param job: without "work/" prefix
    """
    job_dir = get_job_base_dir(job)
    if os.path.exists(f"{job_dir}/finished.tar.gz"):
        with tarfile.open(f"{job_dir}/finished.tar.gz") as tarf:
            while True:
                f = tarf.next()
                if not f:
                    break
                if f.name.startswith("log.") and f.isfile():
                    f = tarf.extractfile(f)
                    res = _read_job_log_check_for_alias(f)
                    if res:
                        return res
    for logfn in glob(f"{job_dir}/log.*"):
        with open(logfn, "rb") as f:
            res = _read_job_log_check_for_alias(f)
            if res:
                return res
    return None


def _read_job_log_check_for_alias(f) -> Optional[List[str]]:
    for line in f.read(3000).splitlines():
        if b" INFO: " not in line:
            continue
        if b"Start Job: Job<" not in line:
            continue
        line = line.decode("utf8")
        m = re.search("Start Job: Job<(.*)> Task: ", line)
        assert m, f"unexpected line: {line!r} in file {f.name}"
        return m.group(1).split()
    return None
