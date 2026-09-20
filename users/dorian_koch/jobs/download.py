"""Resumable download of one large archive, verified by checksum.

Written for the Open Yap corpus (360 GB), but deliberately generic: a vendor download URL, a
resumable fetch, a checksum, and an output path. Nothing dataset-specific belongs here.

⚠ Why a job and not a shell loop: a 360 GB transfer outlives any interactive session and any single
walltime. Sisyphus reschedules an interrupted `resume=` task on its own, and `curl -C -` picks the
byte range back up, so the download survives preemption without anyone watching it. A login-node
`nohup` would be reaped.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from sisyphus import Job, Task


class DownloadArchive(Job):
    """Fetch one file, resuming until its checksum matches.

    ``hash_overwrite`` is the point of this class's hash policy: the URL carries a vendor token that
    can be reissued, and the attempt/rqmt knobs are pure scheduling. If any of those reached the
    hash, re-downloading 360 GB would be one URL refresh away. Pin the identity to what the file IS
    -- its checksum -- not to how we happened to fetch it.
    """

    def __init__(
        self,
        *,
        url: str,
        sha256: str,
        filename: str,
        hash_overwrite: str | None = None,
        attempts: int = 100,
        retry_sleep_sec: int = 5,
        rqmt: dict | None = None,
    ):
        self.url = url
        self.sha256 = sha256
        self.filename = filename
        self.hash_overwrite = hash_overwrite
        self.attempts = int(attempts)
        self.retry_sleep_sec = int(retry_sleep_sec)
        # 24 h is the cap on this cluster; the task resumes, so a slow transfer simply takes more
        # than one allocation rather than failing.
        self.rqmt = rqmt or {"cpu": 2, "mem": 4, "time": 24}
        self.out_dir = self.output_path("archive", directory=True)
        self.out_file = self.output_path(os.path.join("archive", filename))

    @classmethod
    def hash(cls, parsed_args):
        if parsed_args.get("hash_overwrite") is not None:
            return parsed_args["hash_overwrite"]
        d = dict(parsed_args)
        # HOW we fetch is not WHAT we fetch. The checksum is the identity.
        for k in ("url", "attempts", "retry_sleep_sec", "rqmt", "hash_overwrite"):
            d.pop(k, None)
        return super().hash(d)

    def tasks(self):
        # resume=: an interrupted transfer is rescheduled by Sisyphus itself and continues from the
        # bytes already on disk. Per CLAUDE.md, do NOT clear this with `hpc-rerun --include-interrupted`.
        yield Task("run", resume="run", rqmt=self.rqmt)

    def _sha256(self, path: str) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(16 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def run(self):
        # Download into the WORK dir, not the output dir. A partial file sitting in output/ would be
        # indistinguishable from a finished one to anything downstream -- it only moves once the
        # checksum matches.
        # CWD is this job's work directory while the task runs (the idiom used elsewhere in
        # this tree); `Job.job_dir()` does not exist.
        part = os.path.join(os.getcwd(), f"{self.filename}.part")
        final = self.out_file.get_path()
        os.makedirs(os.path.dirname(final), exist_ok=True)

        if os.path.exists(final):
            print(f"[download] {final} already present, verifying", flush=True)
        else:
            for attempt in range(1, self.attempts + 1):
                have = os.path.getsize(part) if os.path.exists(part) else 0
                print(
                    f"[download] attempt {attempt}/{self.attempts}, {have / 1e9:.2f} GB on disk",
                    flush=True,
                )
                rc = subprocess.run(  # --no-progress-meter: curl's default meter writes a line a second into log.run.N, which for a
                    # multi-hour transfer buries everything else in it.
                    ["curl", "-fL", "-C", "-", "--retry", "3", "--no-progress-meter", "-o", part, self.url]
                ).returncode
                if rc == 0:
                    break
                print(f"[download] curl exit {rc}; retrying", flush=True)
                import time

                time.sleep(self.retry_sleep_sec)
            else:
                raise RuntimeError(f"download failed after {self.attempts} attempts: {self.url}")

            got = self._sha256(part)
            if got != self.sha256:
                # Do NOT delete the partial file: on a truncated transfer the bytes are still good
                # and a later resume can finish it. Raise so a human looks.
                raise RuntimeError(
                    f"checksum mismatch for {self.filename}\n  expected {self.sha256}\n  got      {got}\n"
                    f"  ({os.path.getsize(part) / 1e9:.2f} GB at {part})"
                )
            print("[download] checksum OK, moving into place", flush=True)
            shutil.move(part, final)

        got = self._sha256(final)
        if got != self.sha256:
            raise RuntimeError(f"checksum mismatch after move: expected {self.sha256}, got {got}")
        print(f"[download] {final}: {os.path.getsize(final) / 1e9:.1f} GB, sha256 OK", flush=True)
