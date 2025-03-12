"""
Get job aliases from job info file
"""

from typing import List
from .job_dir import get_job_base_dir


def get_job_aliases(job: str) -> List[str]:
    """
    Get job aliases from job info file

    :param job: without "work/" prefix
    """
    job_dir = get_job_base_dir(job)
    aliases = []
    # See Job._sis_setup_directory.
    with open(f"{job_dir}/info") as f:
        for line in f:
            if line.startswith("ALIAS: "):
                aliases.append(line[len("ALIAS: ") :].strip())
    return aliases
