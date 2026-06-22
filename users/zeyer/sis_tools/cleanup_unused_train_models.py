#!/usr/bin/env python3

"""
Cleanup unused train model checkpoints in the work dir,
and also from active finished train jobs.

You provide the Sisyphus config(s),
which are used to determine whether a train job is active or unused.
"""

import os
import re
import sys
import argparse
import logging
import time
from functools import reduce
from typing import TypeVar, Optional

_my_dir = os.path.dirname(__file__)
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_sis_dir = _setup_base_dir + "/tools/sisyphus"
_returnn_dir = _setup_base_dir + "/tools/returnn"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)
        if _returnn_dir not in sys.path:
            sys.path.append(_returnn_dir)


_setup()


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument("config", nargs="+")
    arg_parser.add_argument("--log-level", type=int, default=20)
    arg_parser.add_argument("--mode", default="dryrun", help="dryrun (default), remove")
    arg_parser.add_argument(
        "--filter-work-dir-fs",
        nargs="*",
        help="if set, only consider jobs where the realpath of the work dir is a prefix of the realpath of this",
    )
    arg_parser.add_argument(
        "--keep-last-n",
        type=int,
        default=11,
        help="for active finished trainings, keep the best epochs plus the last N epochs (default 11)",
    )
    arg_parser.add_argument(
        "--drop-optimizer",
        action="store_true",
        help="also remove optimizer state (*.opt.pt) of active finished trainings"
        " (resume-only state, safe to drop once a training is done)",
    )
    args = arg_parser.parse_args()

    args.filter_work_dir_fs = (
        [os.path.realpath(fs) + "/" for fs in args.filter_work_dir_fs] if args.filter_work_dir_fs else None
    )

    def _ignore_work_dir(d: str, *, is_realpath: bool = False) -> bool:
        if not is_realpath:
            d = os.path.realpath(d)
        if args.filter_work_dir_fs:
            for fs in args.filter_work_dir_fs:
                if d.startswith(fs):
                    return False
            return True
        return False

    # See Sisyphus __main__ for reference.

    import sisyphus.logging_format
    from sisyphus.loader import config_manager
    from sisyphus import graph
    from sisyphus import gs
    from i6_core.returnn.training import ReturnnTrainingJob
    from i6_experiments.users.zeyer.utils import job_aliases_from_info
    from i6_experiments.users.zeyer.utils.set_insert_order import SetInsertOrder
    from i6_experiments.users.zeyer.returnn.training import (
        get_relevant_epochs_from_training_learning_rate_scores,
        GetRelevantEpochsFromTrainingLearningRateScoresException,
    )
    from returnn.util import better_exchook
    from returnn.util.basic import human_bytes_size

    better_exchook.install()

    # HACK: Replace the set() by SetInsertOrder() to make the order deterministic.
    graph.graph._targets = SetInsertOrder()

    gs.WARNING_ABSPATH = False
    gs.GRAPH_WORKER = 1  # makes the order deterministic, easier to reason about

    sisyphus.logging_format.add_coloring_to_logging()
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=args.log_level)

    print("Loading Sisyphus configs...")
    start = time.time()
    config_manager.load_configs(args.config)
    print("Loading Sisyphus configs done, took %.3f sec." % (time.time() - start))

    print("Checking active train jobs of the Sisyphus graph...")
    active_train_job_paths_dict = {}  # job path -> job object
    active_train_job_finished_list = []  # list of job objects
    for job in graph.graph.jobs(update_graph=False):
        job: ReturnnTrainingJob
        # noinspection PyProtectedMember
        job_path: str = job._sis_path()
        # Note: no isinstance(job, ReturnnTrainingJob) check here,
        # to also catch fake jobs (via dependency_boundary).
        if not job_path.startswith("work/i6_core/returnn/training/ReturnnTrainingJob."):
            continue
        if _ignore_work_dir(job_path):
            continue
        # print("active train job:", job._sis_path())
        if os.path.isdir(job_path):
            print("Active train job:", job)
            # Resolve symlinks, to only store the path which is actually used for storage
            # (even if that might be some outdated incorrect hash),
            # because that makes the matching easier when we scan the work dir.
            while os.path.islink(job_path):
                job_path_ = os.readlink(job_path)
                job_path__ = _rel_job_path(job_path_)
                if job_path == job_path__:  # same name, just on different disk; stop
                    break
                print("  symlink ->", job_path_, "resolved to", job_path__)
                job_path = job_path__
            if job_path in active_train_job_paths_dict:
                if isinstance(job, ReturnnTrainingJob):  # real job, not fake job
                    # Expect that the prev job is a fake job.
                    # (Note: fake job == via :func:`make_fake_job`)
                    assert not isinstance(active_train_job_paths_dict[job_path], ReturnnTrainingJob), (
                        f"Duplicate active train job path {job_path}:\n"
                        f"Previous: {active_train_job_paths_dict[job_path]}\n"
                        f"New: {job}"
                    )
                    active_train_job_paths_dict[job_path] = job  # update
                # if this is a fake job, just keep the first one
            else:
                active_train_job_paths_dict[job_path] = job
            # noinspection PyProtectedMember
            if isinstance(job, ReturnnTrainingJob) and job._sis_finished():  # If finished, and also no fake job.
                active_train_job_finished_list.append(job)
        else:
            print("Active train job not created yet:", job)
    print("Num active train jobs:", len(active_train_job_paths_dict))

    print("Now checking all train jobs in work dir to find unused train jobs...")
    own_work_realpath = os.path.realpath("work") + "/"
    print("Building alias->job reverse map from the alias/ dir (to find aliases missing from info files)...")
    alias_reverse_map = _build_alias_reverse_map()  # realpath(train job dir) -> [alias paths]
    print("  found train aliases for", len(alias_reverse_map), "distinct job dirs.")
    # Realpaths of the configs we were given; used to skip jobs created by a different config.
    config_file_realpaths = {os.path.realpath(c) for c in args.config}
    # 'Unused' jobs whose real storage is in ANOTHER setup (imported symlink, e.g. the shared LM):
    # listed for transparency but NEVER collected for removal (deleting them corrupts the other setup).
    imported_unused = []  # list of (alias-or-basename, fn, realpath)
    total_model_size_to_remove = 0
    total_train_job_count = 0
    train_job_with_models_to_remove = []
    unused_train_jobs = {}  # key: alias (or basename as fallback), value: job path filename
    model_fns_to_remove = []
    found_active_fns = set()  #  as a sanity check.
    covered_real_job_paths = set()
    # Coverage accounting: total checkpoint (*.pt) bytes per bucket,
    # to show how much of the train data this run reasons about vs leaves untouched (and why).
    cov_size = {
        "active_finished": 0,
        "active_running": 0,
        "unused_own_config": 0,
        "skipped_diff_recipe": 0,
        "skipped_imported": 0,
        "skipped_no_ckpt": 0,
    }
    cov_count = {k: 0 for k in cov_size}
    diff_recipe_size = {}  # recipe basename -> checkpoint bytes (for the breakdown)
    for basename in os.listdir("work/i6_core/returnn/training"):
        if not basename.startswith("ReturnnTrainingJob."):
            continue
        fn = "work/i6_core/returnn/training/" + basename

        if fn not in active_train_job_paths_dict and os.path.islink(fn):
            try:
                link = _rel_job_path(os.readlink(fn))
            except FileNotFoundError:
                pass  # resolves to some non-existing work dir; just keep using it
            else:
                if link != fn:
                    continue  # skip, will be handled when we reach the real name

        # Avoid duplicates, due to symlinks or so.
        # This potentially covers a bit more cases than the _rel_job_path logic above.
        realpath = os.path.realpath(fn)
        if realpath in covered_real_job_paths:
            assert fn not in active_train_job_paths_dict
            continue
        covered_real_job_paths.add(realpath)
        if _ignore_work_dir(realpath, is_realpath=True):
            continue

        total_train_job_count += 1

        if fn in active_train_job_paths_dict:
            found_active_fns.add(fn)
            job_obj = active_train_job_paths_dict[fn]
            bucket = (
                "active_finished"
                if isinstance(job_obj, ReturnnTrainingJob) and job_obj._sis_finished()
                else "active_running"
            )
            cov_size[bucket] += _model_ckpt_size(fn)
            cov_count[bucket] += 1
            continue

        model_dir = fn + "/output/models"
        if not os.path.isdir(model_dir):
            cov_count["skipped_no_ckpt"] += 1  # early error at file creation; no models dir
            continue  # can happen when there was an early error, e.g. at file creation

        # Jobs that errored (error.run.N exists) and have no checkpoint at all
        # have nothing to clean up -- skip them silently (no alias resolution, no listing).
        if not any(m.endswith(".pt") for m in os.listdir(model_dir)) and any(
            b.startswith("error.run.") for b in os.listdir(fn)
        ):
            cov_count["skipped_no_ckpt"] += 1
            continue

        # Detect the recipe/config that created this job: the first non-Sisyphus stack frame
        # in the info file. If it is not among the configs we were given, this job belongs to a
        # different setup/config (e.g. exp2026_05_27_chunked_ctc_ls.py); skip it -- not ours to clean.
        recipe_file = _recipe_file_from_info(fn)
        if recipe_file is not None and recipe_file not in config_file_realpaths:
            sz = _model_ckpt_size(fn)
            cov_size["skipped_diff_recipe"] += sz
            cov_count["skipped_diff_recipe"] += 1
            rec_name = os.path.basename(recipe_file)
            diff_recipe_size[rec_name] = diff_recipe_size.get(rec_name, 0) + sz
            print("Skipping (created by a different config):", rec_name, "->", basename)
            continue

        # Aliases recorded in this job's own info file (assigned when it was created).
        info_aliases = job_aliases_from_info.get_job_aliases(fn)
        # Aliases that CURRENTLY resolve to this exact dir; the alias/ dir is authoritative.
        # An alias is missing from the info file if it was re-assigned later or by another config;
        # conversely the info alias may now point at a newer hash -- see the stale re-hash case.
        current_aliases = alias_reverse_map.get(realpath, [])
        if current_aliases:
            # This dir is the live target of its alias(es).
            name = current_aliases[0]
        elif info_aliases:
            # The alias in our info file now points at a newer hash:
            # this dir is a stale re-hash orphan of that training (its checkpoints are outdated).
            # Display it by its actual hash so it is never confused with the current training,
            # and name the alias it is a stale copy of.
            # (Several stale re-hashes of one training all carry the same ALIAS line.)
            name = f"{basename} (stale re-hash; alias now points to a newer job hash: {info_aliases[0]})"
            print("Stale re-hash:", name)
        else:
            name = basename
            print("No aliases found for train job:", fn)

        # Jobs whose real storage lives in ANOTHER setup (imported symlink, e.g. the shared LM)
        # must never be collected for removal -- deleting them corrupts the source setup.
        # List them for transparency and move on.
        if not realpath.startswith(own_work_realpath):
            cov_size["skipped_imported"] += _model_ckpt_size(fn)
            cov_count["skipped_imported"] += 1
            imported_unused.append((name, fn, realpath))
            continue

        # First collect all, and then go through them in sorted order below.
        # We do this because here the listdir order is totally arbitrary
        # (due to FS, but sorting by hash also would not help),
        # and to inspect the output, it's much more helpful when this is sorted in some way.
        if name in unused_train_jobs:
            name = f"{name} [{basename}]"
        unused_train_jobs[name] = fn

    print("Collecting model checkpoint files from unused train jobs to remove...")
    # Now go sorted.
    for name, fn in sorted(unused_train_jobs.items()):
        model_dir = fn + "/output/models"
        model_count = 0
        model_size = 0
        with os.scandir(model_dir) as it:
            for model_base_fn in it:
                model_base_fn: os.DirEntry
                if not model_base_fn.name.endswith(".pt"):
                    print("Unexpected model file:", model_base_fn.name)
                    continue
                model_fns_to_remove.append(model_base_fn.path)
                model_size += model_base_fn.stat().st_size
                model_count += 1
        if model_count == 0:
            continue
        print("Unused train job:", name, "model size:", human_bytes_size(model_size))
        total_model_size_to_remove += model_size
        cov_size["unused_own_config"] += model_size
        cov_count["unused_own_config"] += 1
        train_job_with_models_to_remove.append(name)

    print(
        "Active-finished keep policy:",
        f"best epochs + last {args.keep_last_n} epochs,",
        "dropping optimizer states" if args.drop_optimizer else "keeping optimizer states",
    )
    print("Collecting model checkpoint files from active finished train jobs to remove...")
    for job in active_train_job_finished_list:
        job: ReturnnTrainingJob
        name = job.get_one_alias() or job.job_id()
        try:
            relevant_epochs = get_relevant_epochs_from_training_learning_rate_scores(
                model_dir=job.out_model_dir,
                scores_and_learning_rates=job.out_learning_rates,
                allow_all_removed=False,
                log_stream=None,
            )
        except GetRelevantEpochsFromTrainingLearningRateScoresException as exc:
            print(f"  Job {name}, warning: {exc}, skipping.")
            continue
        # Relevant epochs so far only contains the best from the learning rate scores.
        # Those are not necessarily e.g. the final epochs, or other fixed kept epochs.
        # Also keep the last N epochs (default 11; --keep-last-n).
        last_epoch = max(job.out_checkpoints.keys())
        relevant_epochs.extend(range(last_epoch - (args.keep_last_n - 1), last_epoch + 1))
        model_dir = job.out_model_dir.get_path()
        model_fns_to_remove_ = []
        model_size = 0
        epochs_to_keep = set()
        epochs_to_delete = set()
        with os.scandir(model_dir) as it:
            for model_base_fn in it:
                model_base_fn: os.DirEntry
                if not model_base_fn.name.endswith(".pt"):
                    print("Unexpected model file:", model_base_fn.name)
                    continue
                if model_base_fn.name.endswith(".opt.pt"):
                    if args.drop_optimizer:
                        model_fns_to_remove_.append(model_base_fn.path)
                        model_size += model_base_fn.stat().st_size
                    continue  # else keep optimizer state (resume-only; this is the last epoch)
                epoch = int(re.match("epoch\\.([0-9]+)\\.pt", model_base_fn.name).group(1))
                if epoch in relevant_epochs:
                    epochs_to_keep.add(epoch)
                    continue
                epochs_to_delete.add(epoch)
                model_fns_to_remove_.append(model_base_fn.path)
                model_size += model_base_fn.stat().st_size
        if not model_fns_to_remove_:
            continue
        if not epochs_to_keep:
            print("Warning: Active finished train job with no relevant epochs found:", name)
            continue  # better skip this
        model_fns_to_remove.extend(model_fns_to_remove_)
        print(
            "Active finished train job with checkpoints to clean:",
            name,
            "cleanup model size:",
            human_bytes_size(model_size),
            "epochs to delete:",
            sorted(epochs_to_delete),
            "epochs to keep:",
            sorted(epochs_to_keep),
        )
        total_model_size_to_remove += model_size
        train_job_with_models_to_remove.append(f"partial: {name}")

    print("Total train job count:", total_train_job_count)
    print("Total train job with models to remove count:", len(train_job_with_models_to_remove))
    print("List of train jobs with models to remove:")
    for alias in train_job_with_models_to_remove:
        print(f" {alias}")
    if not train_job_with_models_to_remove:
        print(" (none)")
    print("Can remove total model size:", human_bytes_size(total_model_size_to_remove))

    # Coverage report: of all train-job checkpoint (*.pt) data in the work dir,
    # how much this run reasons about ("covered") vs leaves untouched, and why.
    covered = cov_size["active_finished"] + cov_size["active_running"] + cov_size["unused_own_config"]
    not_covered = cov_size["skipped_diff_recipe"] + cov_size["skipped_imported"] + cov_size["skipped_no_ckpt"]
    _hb = human_bytes_size
    print("")
    print("=== Checkpoint (*.pt) coverage report ===")
    print("Total checkpoint data scanned:", _hb(covered + not_covered))
    print("Covered (this run reasons about these):", _hb(covered))
    print("  active, finished: ", _hb(cov_size["active_finished"]), f"({cov_count['active_finished']} jobs)")
    print("  active, running:  ", _hb(cov_size["active_running"]), f"({cov_count['active_running']} jobs)")
    print(
        "  unused (our cfg): ",
        _hb(cov_size["unused_own_config"]),
        f"({cov_count['unused_own_config']} jobs) -> fully removable",
    )
    print("Not covered (left untouched):", _hb(not_covered))
    print("  different recipe: ", _hb(cov_size["skipped_diff_recipe"]), f"({cov_count['skipped_diff_recipe']} jobs)")
    print("  imported elsewhere:", _hb(cov_size["skipped_imported"]), f"({cov_count['skipped_imported']} jobs)")
    print("  no checkpoint:    ", _hb(cov_size["skipped_no_ckpt"]), f"({cov_count['skipped_no_ckpt']} jobs)")
    if diff_recipe_size:
        print("  different-recipe breakdown (by recipe):")
        for rec_name, sz in sorted(diff_recipe_size.items(), key=lambda kv: -kv[1]):
            print("    ", rec_name, _hb(sz))
    print("Of covered, removable now:", _hb(total_model_size_to_remove))
    print("")

    if len(found_active_fns) != len(active_train_job_paths_dict):
        print("ERROR: Did not find some active jobs:")
        for fn in active_train_job_paths_dict:
            if fn not in found_active_fns:
                print(" ", fn)
        raise Exception("Did not find some active jobs.")

    if args.mode == "remove":
        for fn in model_fns_to_remove:
            print("Remove model:", fn)
            os.remove(fn)
    elif args.mode == "dryrun":
        print("Dry-run mode, not removing. (use --mode remove to actually remove)")
    else:
        raise ValueError("invalid mode: %r" % args.mode)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _model_ckpt_size(job_dir: str) -> int:
    """
    Sum the size of all model checkpoint files (``*.pt``, incl. ``*.opt.pt``)
    under ``<job_dir>/output/models``.
    Returns 0 if the models dir does not exist yet.
    """
    model_dir = job_dir + "/output/models"
    total = 0
    try:
        with os.scandir(model_dir) as it:
            for e in it:
                if e.name.endswith(".pt"):
                    total += e.stat().st_size
    except FileNotFoundError:
        pass
    return total


def _recipe_file_from_info(job_dir: str) -> Optional[str]:
    """
    Parse the job's info file STACKTRACE and return the first non-Sisyphus source file --
    the recipe/config entry point that created the job -- as an absolute realpath,
    or None if it cannot be determined.
    """
    try:
        with open(job_dir + "/info") as f:
            lines = f.readlines()
    except OSError:
        return None
    in_trace = False
    for line in lines:
        if line.startswith("STACKTRACE:"):
            in_trace = True
            continue
        if not in_trace:
            continue
        clean = _ANSI_RE.sub("", line)
        m = re.search(r'File "([^"]+)", line ', clean)
        if not m:
            continue
        path = m.group(1)
        # Skip Sisyphus framework frames: the 'sis' launcher and everything under tools/sisyphus/.
        if "/sisyphus/" in path or os.path.basename(path) == "sis":
            continue
        return os.path.realpath(path)
    return None


def _build_alias_reverse_map() -> dict:
    """
    Scan the ``alias/`` dir for symlinks pointing at ReturnnTrainingJob dirs,
    and build a map ``realpath(job dir) -> [alias paths]``.

    The ``info`` file of a job only lists the alias(es) assigned by the config that created it;
    aliases assigned later, or by a different config, are missing there.
    The ``alias/`` dir is the authoritative source for which job an alias currently points at.
    """
    reverse_map = {}
    if not os.path.isdir("alias"):
        return reverse_map
    for root, dirs, files in os.walk("alias", followlinks=False):
        for name in dirs + files:
            path = os.path.join(root, name)
            if not os.path.islink(path):
                continue
            target = os.path.realpath(path)
            if "/ReturnnTrainingJob." not in target:
                continue
            reverse_map.setdefault(target, []).append(path)
    return reverse_map


def _rel_job_path(job_path: str) -> str:
    if not job_path.startswith("/"):  # is already relative
        assert job_path.startswith("work/")
        return job_path
    p = -1
    while True:
        p = job_path.find("/work/", p + 1)
        if p < 0:
            raise FileNotFoundError(f"Cannot find relative work path in {job_path!r}")
        rel_path = job_path[p + 1 :]
        assert rel_path.startswith("work/")
        if os.path.exists(rel_path):
            return rel_path


if __name__ == "__main__":
    main()
