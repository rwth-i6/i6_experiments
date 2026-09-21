"""
Pack the outputs of finished jobs which we never use into a tar.gz, to save inodes.

E.g. :class:`i6_core.recognition.scoring.ScliteJob` writes 15 outputs, of which we only ever
read ``reports`` and ``wer``; with a few thousand such jobs that is ~30k inodes for nothing::

    python3 -m i6_experiments.users.schmitt.sis_tools.pack_unused_job_outputs \
        work/i6_core/recognition/scoring/ScliteJob. --keep reports,wer --mode dryrun
    python3 -m i6_experiments.users.schmitt.sis_tools.pack_unused_job_outputs \
        work/i6_core/recognition/scoring/ScliteJob. --keep reports,wer --mode pack

All job dirs matching the given prefix are handled. Everything in ``output/`` except the kept
names goes into ``<job dir>/packed_outputs.tar.gz``, with member names ``output/<name>``, i.e.
``tar -xzf packed_outputs.tar.gz`` in the job dir undoes it, as does ``--mode restore``.
The outputs are only deleted after they were verified to be in the archive.
"""

import os
import argparse
import shutil
import tarfile
from glob import glob

from tqdm import tqdm


ARCHIVE_NAME = "packed_outputs.tar.gz"


def main():
    arg_parser = argparse.ArgumentParser(description=f"{__doc__}", formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument("job_dir_prefix", help="e.g. work/i6_core/recognition/scoring/ScliteJob.")
    arg_parser.add_argument("--keep", required=True, help="comma-separated outputs not to pack, e.g. reports,wer")
    arg_parser.add_argument("--mode", default="dryrun", help="dryrun (default), pack, restore")
    args = arg_parser.parse_args()
    assert args.mode in ("dryrun", "pack", "restore"), f"unknown mode {args.mode!r}"

    keep = set(args.keep.split(","))
    job_dirs = sorted(glob(args.job_dir_prefix + "*"))
    print(f"Job dirs matching {args.job_dir_prefix!r}: {len(job_dirs)}, mode: {args.mode}, keeping: {sorted(keep)}")

    num_jobs = 0
    num_outputs = 0
    num_skipped = 0
    for job_dir in tqdm(job_dirs, unit=" jobs"):
        archive_path = job_dir + "/" + ARCHIVE_NAME
        # Symlinked job dirs point into another setup, not ours to touch.
        if os.path.islink(job_dir) or not os.path.isdir(job_dir + "/output"):
            num_skipped += 1
            continue
        if not any(os.path.exists(job_dir + "/" + name) for name in ["finished", "finished.tar.gz"]):
            num_skipped += 1
            continue

        if args.mode == "restore":
            if not os.path.isfile(archive_path):
                num_skipped += 1
                continue
            num_jobs += 1
            num_outputs += _restore(job_dir, archive_path)
            continue

        if os.path.isfile(archive_path):
            # Never overwrite an archive: it might hold the only copy of some output.
            num_skipped += 1
            continue
        names = [name for name in sorted(os.listdir(job_dir + "/output")) if name not in keep]
        if not names:
            num_skipped += 1
            continue

        if args.mode == "pack":
            _pack(job_dir, names=names, archive_path=archive_path)
        else:
            tqdm.write(f"[dryrun] would pack in {job_dir!r}: {', '.join(names)}")
        num_jobs += 1
        num_outputs += len(names)

    verb = {"dryrun": "would be packed", "pack": "packed", "restore": "restored"}[args.mode]
    print(f"Jobs: {num_jobs} handled, {num_skipped} skipped. Outputs {verb}: {num_outputs}.")
    if args.mode != "restore":
        print(f"Inodes {verb.replace('packed', 'freed')}: {num_outputs - num_jobs} (minus {num_jobs} archives)")


def _pack(job_dir: str, *, names: list, archive_path: str):
    tmp_path = archive_path + ".tmp"
    with tarfile.open(tmp_path, "w:gz") as tar:
        for name in names:
            tar.add(job_dir + "/output/" + name, arcname="output/" + name)
    with tarfile.open(tmp_path, "r:gz") as tar:
        packed = {member.name: member for member in tar}
    for name in names:
        member = packed.get("output/" + name)
        assert member, f"{name!r} missing in {tmp_path!r}"
        if member.isreg():
            assert member.size == os.path.getsize(job_dir + "/output/" + name), f"{name!r} size mismatch"
    os.replace(tmp_path, archive_path)

    for name in names:
        path = job_dir + "/output/" + name
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _restore(job_dir: str, archive_path: str) -> int:
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
        tar.extractall(job_dir, filter="data")
    os.remove(archive_path)
    return len(names)


if __name__ == "__main__":
    main()
