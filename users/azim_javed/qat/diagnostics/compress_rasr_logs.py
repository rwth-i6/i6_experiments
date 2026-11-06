import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def compress(source, target, in_place=False):
    before = source.stat()
    digest = hashlib.sha256()
    descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as raw, source.open("rb") as reader:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as writer:
                while chunk := reader.read(1024 * 1024):
                    digest.update(chunk)
                    writer.write(chunk)
            raw.flush()
            os.fsync(raw.fileno())
        verified = hashlib.sha256()
        with gzip.open(temporary, "rb") as reader:
            while chunk := reader.read(1024 * 1024):
                verified.update(chunk)
        if digest.digest() != verified.digest():
            raise RuntimeError(f"Integrity verification failed: {source}")
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError(f"Source changed during compression: {source}")
        if in_place and not (source.parent.parent / "finished").is_file():
            raise RuntimeError(f"Job is no longer finished: {source}")
        shutil.copystat(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"job": source.parent.parent.name, "source": str(source), "target": str(target),
            "original_bytes": before.st_size, "original_allocated": before.st_blocks * 512,
            "compressed_bytes": target.stat().st_size, "compressed_allocated": target.stat().st_blocks * 512,
            "sha256": digest.hexdigest(), "in_place": in_place}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-root", type=Path, default=Path(
        "/work/smt4/azim.javed/sisyphus_work/i6_core/returnn/forward"))
    parser.add_argument("--count", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--in-place", action="store_true")
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--destination", type=Path)
    location.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.resume is not None:
        run = args.resume.resolve(strict=True)
    else:
        args.destination.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="rasr-migration-", dir=args.destination))
    print(run, flush=True)
    totals = {"jobs": 0, "original_allocated": 0, "compressed_allocated": 0}
    with (run / "results.jsonl").open("a+b") as report:
        fcntl.flock(report, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report.seek(0)
        data = report.read()
        data = data[:data.rfind(b"\n") + 1]
        report.truncate(len(data))
        completed = {}
        for line in data.splitlines():
            result = json.loads(line)
            if result.get("in_place", False) != args.in_place:
                raise ValueError("Cannot change migration mode when resuming")
            assert Path(result["target"]).is_file(), result["target"]
            completed[result["source"]] = result
            totals["jobs"] += 1
            for key in ("original_allocated", "compressed_allocated"):
                totals[key] += result[key]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for job in sorted(args.jobs_root.glob("ResumableReturnnForwardJobV2.*")):
                if args.count is not None and totals["jobs"] + len(futures) >= args.count:
                    break
                source = job / "output/rasr.recog.log"
                if str(source) in completed:
                    continue
                if not (job / "finished").is_file() or not source.is_file():
                    continue
                if args.in_place and (job.is_symlink() or source.is_symlink()):
                    continue
                with source.open("rb") as reader:
                    if reader.read(2) == b"\x1f\x8b":
                        continue
                target = source if args.in_place else run / job.name / "rasr.recog.log"
                if not args.in_place:
                    target.parent.mkdir(exist_ok=True)
                futures.append(pool.submit(compress, source, target, args.in_place))
            for future in as_completed(futures):
                result = future.result()
                report.write((json.dumps(result) + "\n").encode())
                report.flush()
                os.fsync(report.fileno())
                if args.count is not None and args.count <= 100:
                    print(f"{'Migrated' if args.in_place else 'Copied'}: {Path(result['source']).parent.parent}", flush=True)
                totals["jobs"] += 1
                for key in ("original_allocated", "compressed_allocated"):
                    totals[key] += result[key]
                if totals["jobs"] % 10 == 0:
                    print(json.dumps(totals), flush=True)
    (run / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")


if __name__ == "__main__":
    main()
