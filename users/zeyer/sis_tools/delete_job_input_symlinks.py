"""
In gs.JOB_INPUT ("input") directory, Sisyphus keeps symlinks to input files.
Those are purely for the user and not really necessary.
Also, the same info can be found in the ``info`` file,
so the symlinks are purely redundant.
"""

import sys
import os
import argparse
import time
import shutil
import stat
from collections import defaultdict
from functools import reduce


_my_dir = os.path.dirname(__file__)
_base_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)


_setup()


def main():
    from sisyphus import gs

    arg_parser = argparse.ArgumentParser(description=f"{__doc__}", formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument("work_dir", default=gs.WORK_DIR)
    arg_parser.add_argument("--mode", default="dryrun", help="dryrun (default), remove")
    args = arg_parser.parse_args()

    work_dir = args.work_dir

    prev_report_time = time.monotonic()
    count_all_jobs = 0
    count_cleaned_jobs = 0

    for root, dirs, files in os.walk(work_dir):
        if time.monotonic() - prev_report_time > 10:
            prev_report_time = time.monotonic()
            print(f"Jobs found so far: {count_all_jobs}, cleaned: {count_cleaned_jobs}, current dir: {root!r}")

        exclude_recurse_dirs = set()
        for dir_name in dirs:
            job_dir = root + "/" + dir_name

            if _is_job_dir(job_dir):
                count_all_jobs += 1
                exclude_recurse_dirs.add(dir_name)

                job_input_dir = job_dir + "/input"
                if os.path.isdir(job_input_dir):
                    if args.mode == "dryrun":
                        counts_per_type = defaultdict(int)
                        with os.scandir(job_input_dir) as it:
                            for entry in it:
                                file_type = stat.S_IFMT(entry.stat(follow_symlinks=False).st_mode)
                                file_type_str = {stat.S_IFLNK: "l", stat.S_IFREG: "f", stat.S_IFDIR: "d"}.get(
                                    file_type, f"other({file_type})"
                                )
                                counts_per_type[file_type_str] += 1
                        counts_str = ", ".join(f"{k}={v}" for k, v in counts_per_type.items())
                        print(f"[dryrun] would remove input dir in job {job_dir!r},) contents: {counts_str}")

                    elif args.mode == "remove":
                        shutil.rmtree(job_input_dir)
                        print(f"Removed input dir in job {job_dir!r}")

                    else:
                        raise ValueError(f"Unknown mode: {args.mode!r}")

                    count_cleaned_jobs += 1

            elif "." in dir_name:
                # exclude directories with dots in their names (probably not job dirs)
                exclude_recurse_dirs.add(dir_name)

        if exclude_recurse_dirs:
            # don't visit those directories
            dirs[:] = [d for d in dirs if d not in exclude_recurse_dirs]

    print(f"Total jobs found: {count_all_jobs}, cleaned: {count_cleaned_jobs}")


def _is_job_dir(job_dir: str) -> bool:
    """
    :param job_dir: absolute or relative path to job directory
    """
    base_name = os.path.basename(job_dir)
    if "." not in base_name:
        return False
    if not os.path.isfile(job_dir + "/info"):
        return False
    if not os.path.isdir(job_dir + "/output"):
        return False
    return True


if __name__ == "__main__":
    main()
