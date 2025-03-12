"""
Get job aliases from job info file
"""

from typing import List
from .job_dir import get_job_base_dir


def get_job_aliases(job: str) -> List[str]:
    """
    Get job aliases from job info file

    :param job: without "work/" prefix
    :return: list of aliases. those are paths all with "alias/" prefix
    """
    job_dir = get_job_base_dir(job)
    aliases = []
    # See Job._sis_setup_directory.
    with open(f"{job_dir}/info") as f:
        for line in f:
            if line.startswith("ALIAS: "):
                # Add "alias/" prefix to make it a valid path,
                # and also to have it consistent to job_aliases_from_log.
                aliases.append("alias/" + line[len("ALIAS: ") :].strip())
    return aliases
