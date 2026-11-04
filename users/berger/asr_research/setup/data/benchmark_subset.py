"""Select benchmark utterances before recognition, rather than only at scoring."""

import json

from sisyphus import Job, Task
from i6_core.util import uopen


class PrepareBenchmarkSubsetJob(Job):
    def __init__(self, dataset, references, manifest, segments, *, code_version):
        self.dataset, self.references, self.manifest = dataset, references, manifest
        self.segments, self.code_version = segments, code_version
        self.out_dataset = self.output_path("dataset", directory=True)
        self.out_references = self.output_path("references.jsonl")
        self.out_manifest = self.output_path("manifest.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=8, time=2))

    def run(self):
        from datasets import load_from_disk

        with uopen(self.segments.get_path(), "rt", encoding="utf-8") as src:
            wanted = {line.strip().rsplit("/", 2)[-2] for line in src if line.strip()}
        with open(self.manifest.get_path()) as src:
            manifest = json.load(src)
        wanted -= set(manifest.get("excluded_empty_reference_ids", []))
        with open(self.references.get_path()) as src:
            references = [row for line in src if (row := json.loads(line))["id"] in wanted]
        if not wanted or len(references) != len(wanted) or {r["id"] for r in references} != wanted:
            raise ValueError("Missing/duplicate references for benchmark subset")
        dataset = load_from_disk(self.dataset.get_path())
        ids = list(map(str, dataset["id"]))
        indices = [i for i, seq_id in enumerate(ids) if seq_id in wanted]
        if len(indices) != len(wanted):
            raise ValueError("Missing/duplicate audio for benchmark subset")
        dataset.select(indices).save_to_disk(self.out_dataset.get_path())
        with open(self.out_references.get_path(), "w") as dst:
            for row in references:
                dst.write(json.dumps(row) + "\n")
        manifest.update(num_segments=len(indices), subset="dev.short")
        with open(self.out_manifest.get_path(), "w") as dst:
            json.dump(manifest, dst)
