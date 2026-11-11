"""Validated prefix checkpoints for independent-segment inference, not within-audio resume."""

import hashlib
import json
import math
import os
from pathlib import Path


class SegmentResumeState:
    def __init__(self, hypotheses, ids, identity):
        self.path = Path(hypotheses)
        self.ids = tuple(str(i) for i in ids)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Resume requires unique segment IDs")
        self.start_index = 0
        self.elapsed_seconds = self.audio_seconds = 0.0
        self.validation = {}
        self.metadata = dict(
            schema=1,
            identity=identity,
            ids_sha256=hashlib.sha256(json.dumps(self.ids, ensure_ascii=False).encode()).hexdigest(),
        )
        sidecar = self.path.with_name(self.path.name + ".resume.json")
        if sidecar.exists():
            if json.loads(sidecar.read_text()) != self.metadata:
                raise ValueError("Resume identity/config/dataset changed; refusing to append")
        else:
            if self.path.exists() and self.path.stat().st_size:
                raise ValueError("Existing hypotheses have no resume identity; refusing to append")
            temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
            temporary.write_text(json.dumps(self.metadata, indent=2))
            os.replace(temporary, sidecar)
        if not self.path.exists():
            return
        with self.path.open("rb+") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                # A torn final write is discarded, never treated as a completed segment.
                if not line.endswith(b"\n"):
                    stream.truncate(offset)
                    break
                row = json.loads(line)
                if self.start_index >= len(self.ids) or row["id"] != self.ids[self.start_index]:
                    raise ValueError("Resume hypotheses are not an ordered dataset prefix")
                elapsed, audio = row["elapsed_seconds"], row["_resume_audio_seconds"]
                if not math.isfinite(elapsed) or not math.isfinite(audio) or elapsed < 0 or audio <= 0:
                    raise ValueError("Invalid checkpoint timing")
                self.start_index += 1
                self.elapsed_seconds += elapsed
                self.audio_seconds += audio
                self.validation.update(row.get("_resume_validation", {}))
