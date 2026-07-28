#!/usr/bin/env python3

"""
Delete dangling symlinks and empty directories, to clean up cruft from graph building
and cross-setup job sharing.

Under the given work dir, it finds:
- broken symlinks (target does not exist),
- symlinks whose target dir is recursively empty (no files anywhere under it),
  e.g. the eagerly-created empty share targets of ``setup_job_symlinks``,
- real directories that are recursively empty,
  e.g. leftover ``engine``/``output``/``work`` skeletons of ``.cleared`` jobs.

For a symlink to a recursively-empty target, BOTH the symlink AND the target dir are removed
-- even when the target lives outside the given work dir (e.g. in a shared/other setup's work),
because otherwise the empty job dir would be left behind and the job would not actually be cleaned.
Only ever removes recursively-empty dirs (no files anywhere under them), so this never deletes data.
A shared target removed here may leave another setup's symlink to it dangling -- that symlink was
pointing at an empty dir anyway, and this same tool cleans broken symlinks.

Dry-run by default; pass ``--mode remove`` to actually delete.
"""

import os
import shutil
import argparse


def _is_recursively_empty(path: str) -> bool:
    """True if ``path`` (a dir) contains no files and no symlinks anywhere under it."""
    for root, dirs, files in os.walk(path, followlinks=False):
        if files:  # regular files and symlinks-to-files both show up here
            return False
        for d in dirs:
            if os.path.islink(os.path.join(root, d)):  # symlink-to-dir counts as content
                return False
    return True


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument("work_dir")
    arg_parser.add_argument("--mode", default="dryrun", help="dryrun (default), remove")
    args = arg_parser.parse_args()
    assert os.path.isdir(args.work_dir), f"work_dir {args.work_dir!r} is not a directory"
    assert args.mode in ("dryrun", "remove"), f"unknown mode {args.mode!r}"
    remove = args.mode == "remove"

    broken_links = []  # symlinks with a missing target
    empty_target_links = []  # (symlink, target_realpath) where target dir is recursively empty
    empty_dirs = []  # real, recursively-empty dirs (top-level; not descended into)

    for root, dirs, files in os.walk(args.work_dir, topdown=True, followlinks=False):
        pruned = []
        for name in list(dirs) + files:
            p = os.path.join(root, name)
            if os.path.islink(p):
                rp = os.path.realpath(p)
                if not os.path.exists(rp):
                    broken_links.append(p)
                elif os.path.isdir(rp) and _is_recursively_empty(rp):
                    empty_target_links.append((p, rp))
                continue
            if name in files:
                continue
            # real subdir: if recursively empty, take the whole subtree and don't descend into it
            if _is_recursively_empty(p):
                empty_dirs.append(p)
                pruned.append(name)
        if pruned:
            dirs[:] = [d for d in dirs if d not in pruned]

    # Distinct dirs to remove: the empty targets (possibly external, possibly shared by several
    # symlinks) plus the local empty dirs. Dedup so a shared target is removed once.
    dirs_to_remove = {rp for _, rp in empty_target_links} | set(empty_dirs)
    external = sum(1 for _, rp in empty_target_links if not rp.startswith(os.path.realpath(args.work_dir) + "/"))

    print(f"work_dir: {args.work_dir}")
    print(f"broken symlinks:                 {len(broken_links)}")
    print(f"symlinks -> recursively-empty:   {len(empty_target_links)}")
    print(f"recursively-empty directories:   {len(empty_dirs)}")
    print(f"distinct dirs to remove:         {len(dirs_to_remove)} ({external} target(s) outside work_dir)")

    if not remove:
        for p in broken_links[:5]:
            print("  [broken-symlink]", p)
        for p, rp in empty_target_links[:5]:
            print("  [empty-target]", p, "->", rp)
        for p in empty_dirs[:5]:
            print("  [empty-dir]", p)
        print("Dry run. Use `--mode remove` to actually delete.")
        return

    for p in broken_links:
        os.unlink(p)
    for p, _ in empty_target_links:
        if os.path.islink(p):
            os.unlink(p)
    n_dirs = 0
    for d in dirs_to_remove:
        # A parent removed earlier may already have taken a nested one; re-check emptiness too.
        if os.path.isdir(d) and not os.path.islink(d) and _is_recursively_empty(d):
            shutil.rmtree(d)
            n_dirs += 1
    print(f"Removed {len(broken_links) + len(empty_target_links)} symlinks and {n_dirs} empty dirs.")


if __name__ == "__main__":
    main()
