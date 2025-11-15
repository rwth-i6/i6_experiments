from multiprocessing.pool import ThreadPool
import os
import sys
import functools
from argparse import ArgumentParser

_my_dir = os.path.dirname(__file__)
_base_dir = functools.reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
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


from sisyphus import tools, gs, Job
from sisyphus.job import job_finished
from i6_experiments.common.utils.fake_job import make_fake_job


def clean(*, work_dir: str, dry_run: bool = False, stop_after_n_jobs: int = -1) -> None:
    """
    Adapted code from :class:`sisyphus.manager.JobCleaner`.
    """
    gs.WORK_DIR = work_dir

    count_all_jobs = 0
    count_cleaned_jobs = 0

    for root, dirs, files in os.walk(work_dir):
        job_dir_names = set()
        for dir_name in dirs:
            job_dir = root + "/" + dir_name
            if _is_job_dir(job_dir):
                count_all_jobs += 1
                job_dir_names.add(dir_name)

                assert job_dir.startswith(work_dir + "/")
                parts = job_dir[len(work_dir) + 1 :].split("/")
                module_s = ".".join(parts[:-1])
                cls_name, sis_hash = parts[-1].split(".", 1)

                job = make_fake_job(module=module_s, name=cls_name, sis_hash=sis_hash)
                assert job._sis_path() == job_dir
                job._sis_finished = functools.partial(_job_sis_finished_simple, job)

                if job._sis_cleanable():
                    if dry_run:
                        print(f"Clean: {job._sis_id()!r} in {job_dir!r} (dry-run)")
                    else:
                        print(f"Clean: {job._sis_id()!r} in {job_dir!r}")
                        job._sis_cleanup()
                        count_cleaned_jobs += 1
                        if 0 < stop_after_n_jobs <= count_cleaned_jobs:
                            break

        if 0 < stop_after_n_jobs <= count_cleaned_jobs:
            break

        if job_dir_names:
            # don't visit job directories
            dirs[:] = [d for d in dirs if d not in job_dir_names]

    print(f"Total jobs found: {count_all_jobs}, cleaned: {count_cleaned_jobs}")


def _is_job_dir(job_dir: str) -> bool:
    """
    :param job_dir: absolute or relative path to job directory
    """
    if not os.path.isfile(job_dir + "/info"):
        return False
    if not os.path.isdir(job_dir + "/output"):
        return False
    return True


def _job_sis_finished_simple(self: Job) -> bool:
    """Return True if job or task is finished"""
    if self._sis_is_finished:
        return True

    if job_finished(self._sis_path()):
        # Job is already marked as finished, skip check next time
        self._sis_is_finished = True
        return True

    return False


def main():
    """main"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("work_dir", default="work")
    arg_parser.add_argument("--mode", default="dryrun", help="dryrun (default), run")
    arg_parser.add_argument("--stop-after-n-jobs", type=int, default=-1, help="Stop after cleaning N jobs.")
    args = arg_parser.parse_args()
    assert os.path.isdir(args.work_dir), f"Work dir {args.work_dir!r} does not exist or is not a directory."
    assert args.mode in ("dryrun", "run"), f"Unknown mode {args.mode!r}."
    clean(work_dir=args.work_dir, dry_run=(args.mode == "dryrun"), stop_after_n_jobs=args.stop_after_n_jobs)
    if args.mode == "dryrun":
        print("Dry run. Use `--mode run` to actually clean jobs.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(1)
