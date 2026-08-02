"""
Remove cruft dirs under a work dir: ``__pycache__``, ``crash_dir``, ``*.cleared*`` and ``*.broken*`` job dirs.

These are safe to remove:
- ``__pycache__``: Python bytecode, regenerated on demand.
- ``crash_dir``: crash dumps from failed job runs.
- ``*.cleared*`` / ``*.broken*``: superseded / failed job dirs Sisyphus renamed aside.

Pure ``os.walk`` (no external tools). Matched dirs are pruned from the walk and removed whole (dryrun by default).
"""

import os
import shutil
import argparse
from collections import defaultdict


def _cat(name: str):
    if name == "__pycache__":
        return "__pycache__"
    if name == "crash_dir":
        return "crash_dir"
    if ".cleared" in name:
        return ".cleared"
    if ".broken" in name:
        return ".broken"
    return None


def _count_inodes(path: str) -> int:
    n = 1  # the dir itself
    for _root, dirs, files in os.walk(path):
        n += len(dirs) + len(files)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dir")
    ap.add_argument("--mode", default="dryrun", choices=["dryrun", "remove"])
    args = ap.parse_args()
    base = os.path.abspath(args.work_dir.rstrip("/"))
    assert os.path.isdir(base), f"not a dir: {base!r}"

    cruft = []  # (cat, dir, inode_count)
    for dirpath, dirnames, _files in os.walk(base, topdown=True):
        keep = []
        for d in dirnames:
            cat = _cat(d)
            full = os.path.join(dirpath, d)
            if cat and not os.path.islink(full):
                cruft.append((cat, full, _count_inodes(full)))
                # matched: leave out of `keep` so os.walk does not descend into it
            else:
                keep.append(d)
        dirnames[:] = keep

    per_cat = defaultdict(lambda: [0, 0])  # cat -> [num dirs, num inodes]
    for cat, _full, n in cruft:
        per_cat[cat][0] += 1
        per_cat[cat][1] += n
    for cat, (ndirs, ninodes) in sorted(per_cat.items(), key=lambda x: -x[1][1]):
        print(f"{cat:>12}: {ndirs} dirs, {ninodes} inodes")
    total = sum(n for _cat_name, _full, n in cruft)
    print(f"TOTAL: {len(cruft)} cruft dirs, {total} inodes")

    if args.mode == "dryrun":
        for cat, full, n in cruft[:10]:
            print(f"  [dryrun] would rm -rf {full}  ({n} inodes)")
        print("Dry run. Use --mode remove to delete.")
        return

    removed = 0
    for _cat_name, full, _n in cruft:
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full)
            removed += 1
    print(f"Removed {removed} cruft dirs, ~{total} inodes.")


if __name__ == "__main__":
    main()
