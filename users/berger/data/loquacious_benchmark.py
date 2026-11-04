"""Native Loquacious evaluation audio and reusable subset scoring metadata."""

import json
import os
import re

from sisyphus import Job, Task
from i6_core.util import uopen

from .open_asr import reference_rows

LOQUACIOUS_REVISION = "0e84cdb9e4b826afaabca5d33ec9453b11aacef3"
FILTERS = {"commonvoice": "common_voice", "librispeech": "^[0-9-]*$", "voxpopuli": "PLENARY", "yodas": r"\.wav$"}


class PrepareLoquaciousBenchmarkJob(Job):
    def __init__(self, split, *, revision=LOQUACIOUS_REVISION, code_version):
        if split not in {"dev", "test"}:
            raise ValueError(split)
        self.split, self.revision, self.code_version = split, revision, code_version
        self.out_dataset = self.output_path("dataset", directory=True)
        self.out_references = self.output_path("references.jsonl")
        self.out_manifest = self.output_path("manifest.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=2, mem=16, time=8))

    def run(self):
        os.environ["HF_HOME"] = "/work/asr4/berger/huggingface"
        os.environ["HF_HUB_CACHE"] = "/work/asr4/berger/huggingface/hub"
        os.environ["HF_DATASETS_CACHE"] = "/work/asr4/berger/huggingface/datasets"
        from datasets import Audio, load_dataset

        ds = load_dataset("speechbrain/LoquaciousSet", "small", split=self.split, revision=self.revision)
        ds = ds.rename_column("ID", "id").rename_column("wav", "audio")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        rows = reference_rows(ds.remove_columns("audio"))
        excluded_ids = sorted(set(map(str, ds["id"])) - {r["id"] for r in rows})
        ds = ds.select([r.pop("source_index") for r in rows]).select_columns(["id", "audio", "text"])
        ds.save_to_disk(self.out_dataset.get_path())
        with open(self.out_references.get_path(), "w") as dst:
            for row in rows:
                dst.write(json.dumps(row) + "\n")
        with open(self.out_manifest.get_path(), "w") as dst:
            json.dump(
                dict(
                    repo="speechbrain/LoquaciousSet",
                    config="small",
                    split=self.split,
                    revision=self.revision,
                    num_segments=len(rows),
                    sample_rate=16000,
                    audio_reencoded=False,
                    cache="/work/asr4/berger/huggingface",
                    excluded_empty_reference_ids=excluded_ids,
                ),
                dst,
                indent=2,
            )


class LoquaciousSubsetJob(Job):
    def __init__(self, references, hypotheses, subset, *, segments=None, manifest=None):
        if subset not in set(FILTERS) | {"short"} or (subset == "short" and segments is None):
            raise ValueError(subset)
        self.references, self.hypotheses, self.subset, self.segments = references, hypotheses, subset, segments
        self.manifest = manifest
        self.out_references = self.output_path("references.jsonl")
        self.out_hypotheses = self.output_path("hypotheses.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        ids = None
        if self.segments is not None:
            with uopen(self.segments.get_path(), "rt", encoding="utf-8") as src:
                ids = {line.strip().rsplit("/", 2)[-2] for line in src if line.strip()}
        chosen = set()
        with open(self.references.get_path()) as src, open(self.out_references.get_path(), "w") as dst:
            for line in src:
                row = json.loads(line)
                if (row["id"] in ids) if ids is not None else re.search(FILTERS[self.subset], row["id"]):
                    chosen.add(row["id"])
                    dst.write(line)
        excluded = set()
        if self.manifest is not None:
            with open(self.manifest.get_path()) as src:
                excluded = set(json.load(src)["excluded_empty_reference_ids"])
        if not chosen or (ids is not None and chosen != ids - excluded):
            raise ValueError("Missing/empty Loquacious subset")
        seen = set()
        with open(self.hypotheses.get_path()) as src, open(self.out_hypotheses.get_path(), "w") as dst:
            for line in src:
                key = json.loads(line)["id"]
                if key in chosen:
                    if key in seen:
                        raise ValueError("Duplicate hypothesis")
                    seen.add(key)
                    dst.write(line)
        if seen != chosen:
            raise ValueError("Subset hypothesis coverage mismatch")
