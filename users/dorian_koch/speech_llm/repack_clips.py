"""Repack finished ``{i}.wav`` clip dirs into arrow datasets, in place (backlog E13a).

Sisyphus hashes a job's *inputs*, never its output content, so this reclaims inodes from work
already done without changing a single job id, re-running anything, or altering a scored number.
``clip_store.open_clips`` reads either layout, so consumers do not notice.

    # look first -- this is the default and it writes nothing
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/repack_clips.py

    # then, deliberately
    ... repack_clips.py --apply --limit 1

**Order matters.** ``MergeMoshiOutputsViaSymlinks`` joins shards by symlinking their wavs, so a
merged dir's entries point *into* shard dirs. Repack the **merges first** (``--kind merge``): that
is where most of the inodes are and it leaves the shards untouched. Only once no merge references a
shard is that shard safe to repack (``--kind shard``) -- otherwise the merge's links dangle, and
``os.symlink`` dangles silently.

Every dir is verified clip-by-clip against the originals before anything is deleted; a mismatch
aborts that dir with its wavs intact. Idempotent: an already-arrow dir is skipped, so re-running
over a partly-done list is safe.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

from i6_experiments.users.dorian_koch.speech_llm.clip_store import repack_wav_dir  # noqa: E402

WORK = SETUP / "work/i6_experiments/users/dorian_koch/speech_llm"

#: Where clip dirs actually live, by kind. ``merge`` dirs hold symlinks into ``shard`` dirs, which
#: is why they are separate selections rather than one glob -- see the ordering note above.
KINDS = {
    "merge": ["knowledge_benchmark/MergeMoshiOutputsViaSymlinks.*/output/*"],
    "shard": ["speech_inference/SpeechInference.*/output/*", "knowledge_benchmark/MoshiInference.*/output/*"],
    "tts": ["knowledge_benchmark/ChatterboxSingleSpeakerInference.*/output/*"],
}


def is_finished(job_dir: Path) -> bool:
    return (job_dir / "finished").exists() or (job_dir / "finished.tar.gz").exists()


def candidates(kinds):
    for kind in kinds:
        for pat in KINDS[kind]:
            for d in sorted(WORK.glob(pat)):
                if not d.is_dir():
                    continue
                # output/<name>/ -> the job dir is three levels up. Only finished jobs: a running
                # job's output dir is being written right now.
                job_dir = d.parent.parent
                if not is_finished(job_dir):
                    continue
                yield kind, d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", action="append", choices=sorted(KINDS), help="repeatable; default: merge only")
    p.add_argument("--apply", action="store_true", help="actually repack (default is a dry run)")
    p.add_argument("--limit", type=int, default=0, help="stop after N dirs (0 = no limit)")
    p.add_argument("--dir", action="append", help="repeatable explicit dir, instead of the globs")
    args = p.parse_args()

    kinds = args.kind or ["merge"]
    if args.dir:
        work = [("explicit", Path(d)) for d in args.dir]
    else:
        work = list(candidates(kinds))

    totals = {"dirs": 0, "clips": 0, "freed": 0, "skipped": 0, "failed": 0}
    for kind, d in work:
        if args.limit and totals["dirs"] >= args.limit:
            break
        try:
            # A merge dir is symlinks by construction; that is the case we are here for.
            res = repack_wav_dir(d, dry_run=not args.apply, follow_symlinks=True)
        except Exception as exc:  # noqa: BLE001 -- one bad dir must not abandon the rest
            totals["failed"] += 1
            print(f"[FAIL] {d}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if res["action"] in ("already-arrow", "no-clips", "missing"):
            totals["skipped"] += 1
            continue
        totals["dirs"] += 1
        totals["clips"] += res["clips"]
        totals["freed"] += res["freed"]
        # flush: stdout is a file under sbatch, so without this a run that is killed (a login
        # node reaping it, a walltime) loses every line it had printed and looks like it did nothing.
        print(f"[{res['action']}] {kind:7s} {res['clips']:5d} clips, {res['freed']:5d} inodes  {d}", flush=True)

    verb = "freed" if args.apply else "would free"
    print(
        f"\n{totals['dirs']} dirs, {totals['clips']} clips, {verb} ~{totals['freed']} inodes"
        f"  (skipped {totals['skipped']}, failed {totals['failed']})"
    )
    if not args.apply:
        print("dry run -- nothing was written. Re-run with --apply.")
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
