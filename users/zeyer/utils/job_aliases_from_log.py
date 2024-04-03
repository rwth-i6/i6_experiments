from typing import Optional, List
import os
import re
import tarfile
from glob import glob
from . import sis_path


def get_job_aliases(job: str) -> Optional[List[str]]:
    """
    :param job: without "work/" prefix
    """
    work_dir_prefix = sis_path.get_work_dir_prefix()
    if os.path.exists(f"{work_dir_prefix}{job}/finished.tar.gz"):
        with tarfile.open(f"{work_dir_prefix}{job}/finished.tar.gz") as tarf:
            while True:
                f = tarf.next()
                if not f:
                    break
                if f.name.startswith("log.") and f.isfile():
                    f = tarf.extractfile(f)
                    res = _read_job_log_check_for_alias(f)
                    if res:
                        return res
    for logfn in glob(f"{work_dir_prefix}{job}/log.*"):
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
