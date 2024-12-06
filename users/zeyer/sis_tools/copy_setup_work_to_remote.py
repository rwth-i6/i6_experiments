"""
Copy relevant jobs from the setup work directory to some remote machine.
"""

import sys
import os
import argparse
from subprocess import check_call
from functools import reduce
from typing import Optional, List, Set

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

    try:
        import better_exchook

        better_exchook.install()

    except ImportError:
        pass


_setup()


from sisyphus.loader import config_manager
from sisyphus import gs, tk, Job


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", required=True, help="sis config")
    arg_parser.add_argument("--dest", help="dest dir (e.g. host:...")
    arg_parser.add_argument("--dry-run", action="store_true", help="just print what would be copied")
    args = arg_parser.parse_args()

    config_manager.load_config_file(args.config)

    # Collect all non-finished jobs.
    jobs = tk.sis_graph.get_jobs_by_status(skip_finished=True)
    jobs_: List[Job] = reduce(lambda a, b: (a + sorted(b[1])) if not b[0].startswith("input_") else a, jobs.items(), [])

    # Collect all finished inputs of those jobs.
    inputs_visited: Set[tk.Path] = set()
    inputs_finished: List[tk.Path] = []
    for job in jobs_:
        # noinspection PyProtectedMember
        for input_path in sorted(job._sis_inputs):
            input_path: tk.Path
            if input_path in inputs_visited:
                continue
            inputs_visited.add(input_path)
            if input_path.available():
                inputs_finished.append(input_path)

    # Collect jobs of those inputs.
    jobs_visited: Set[Job] = set()
    jobs_finished: List[Job] = []
    for input_path in inputs_finished:
        job: Optional[Job] = input_path.creator
        if not job:
            continue
        if job in jobs_visited:
            continue
        jobs_visited.add(job)
        # noinspection PyProtectedMember
        if job._sis_finished():
            # noinspection PyProtectedMember
            print("Finished input job:", job)
            jobs_finished.append(job)
        else:
            print("Input job not finished:", job)

    if not args.dest:
        print("No --dest given. Will not copy anything. Quitting.")
        return

    # Note: We might want to add some blacklist, to exclude things like:
    #     --exclude "i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.*"
    #     --exclude "i6_core/returnn/oggzip/BlissToOggZipJob.*/work"
    #     --exclude "i6_core/audio/encoding/BlissChangeEncodingJob.*/output/audio"

    assert args.dest.endswith("/work/")  # sanity check
    cmd = ["rsync", "-avRP"]
    if args.dry_run:
        cmd.append("--dry-run")
    for job in jobs_finished:
        # noinspection PyProtectedMember
        job_path = job._sis_path()
        assert job_path.startswith("work/")
        # This is the rync way to specify the root dir.
        job_path = "work/./" + job_path[len("work/") :]
        assert os.path.isdir(job_path)
        cmd.append(job_path)
    cmd.append(args.dest)
    print("$", " ".join(cmd))
    check_call(cmd)

    # We might also want to fix symlinks on the dest side now...


def _get_sis_job_name(job: Job) -> str:
    cls = type(job)
    module_name = cls.__module__
    recipe_prefix = gs.RECIPE_PREFIX + "."
    if module_name.startswith(recipe_prefix):
        sis_name = module_name[len(recipe_prefix) :]
    else:
        sis_name = module_name
    sis_name = os.path.join(sis_name.replace(".", os.path.sep), cls.__name__)
    assert job.job_id().startswith(sis_name + ".")
    return sis_name


if __name__ == "__main__":
    main()
