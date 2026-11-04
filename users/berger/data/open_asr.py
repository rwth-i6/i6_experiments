"""Pinned public English short-form Open ASR benchmark (2026-09-29).

No evaluation text is used to build a lexicon or train an LM. Audio stays in its
published encoding; all recognizers consume the same saved HF dataset.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from sisyphus import Job, Task

from ..model_pipelines.common.hf_scoring import normalize_hf_english


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    repo: str
    revision: str
    config: str | None = None
    split: str = "test"
    parent_repo: str | None = None
    parent_revision: str | None = None


BENCHMARK_REVISION = "7ce09ca6c5fb4aac32b4a528d906356a21ad5399"
DATASET_REPO = "hf-audio/open-asr-leaderboard"
DATASET_REVISION = "b6bdcd0beb34f8975dc659796176d88f43aff502"
DATASETS = (
    BenchmarkDataset("librispeech-test-clean", DATASET_REPO, DATASET_REVISION, "librispeech", "test.clean"),
    BenchmarkDataset("librispeech-test-other", DATASET_REPO, DATASET_REVISION, "librispeech", "test.other"),
    BenchmarkDataset("ami-cleaned", DATASET_REPO, DATASET_REVISION, "ami_cleaned"),
    BenchmarkDataset("gigaspeech-cleaned", DATASET_REPO, DATASET_REVISION, "gigaspeech_cleaned"),
    BenchmarkDataset("spgispeech", DATASET_REPO, DATASET_REVISION, "spgispeech"),
    BenchmarkDataset("voxpopuli-cleaned", DATASET_REPO, DATASET_REVISION, "voxpopuli_cleaned_aa"),
    BenchmarkDataset("monsoon-en-in", "VoiceArena/Monsoon_en_IN_test", "bc1da7b42ef6e2853123c97bf6d22067e4802d11"),
    BenchmarkDataset(
        "earnings22-cleaned",
        "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked",
        "27f7d0056d593231974a752ce03c1e965da69b3f",
        parent_repo="ArtificialAnalysis/Earnings22-Cleaned-AA",
        parent_revision="5999936ee3d8d3cde160e3c65a8cc6ff5b7c5276",
    ),
)


def reference_rows(metadata):
    """Validate identity and session grouping without ever inspecting audio."""
    seen, groups = set(), {}
    result = []
    for index, sample in enumerate(metadata):
        seq_id = str(sample["id"])
        if not seq_id or seq_id in seen:
            raise ValueError(f"Empty/duplicate benchmark id: {seq_id!r}")
        seen.add(seq_id)
        text = sample["text"]
        if not isinstance(text, str):
            raise ValueError(f"Non-string reference for {seq_id}")
        if not normalize_hf_english(text).strip():
            # Same reference-only empty-text filter as the official English runner.
            continue
        parent = str(sample.get("parent_id", seq_id))
        chunk = int(sample.get("chunk_index", 0))
        entry = groups.setdefault(parent, {"text": text, "chunks": set()})
        if entry["text"] != text or chunk in entry["chunks"]:
            raise ValueError(f"Inconsistent parent transcript/duplicate chunk: {parent}/{chunk}")
        entry["chunks"].add(chunk)
        result.append(dict(id=seq_id, source_index=index, group_id=parent, chunk_index=chunk, reference=text))
    if not result:
        raise ValueError("Benchmark has no nonempty references")
    for group, entry in groups.items():
        indices = sorted(entry["chunks"])
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError(f"Missing chunk indices for {group}: {indices}")
    return result


class PrepareOpenAsrDatasetJob(Job):
    def __init__(self, dataset: BenchmarkDataset, *, code_version: str):
        self.dataset = dataset
        self.code_version = code_version
        self.out_dataset = self.output_path("dataset", directory=True)
        self.out_references = self.output_path("references.jsonl")
        self.out_manifest = self.output_path("manifest.json")
        self.rqmt = {"cpu": 2, "mem": 12, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from dataclasses import asdict
        from datasets import Audio, load_dataset
        from huggingface_hub import hf_hub_download

        spec = self.dataset
        ds = load_dataset(spec.repo, spec.config, split=spec.split, revision=spec.revision)
        if spec.parent_repo:
            parents = load_dataset(spec.parent_repo, split=spec.split, revision=spec.parent_revision)
            if "audio" in parents.column_names:
                parents = parents.remove_columns("audio")
            parent_text = {str(s["id"]): s["transcript"] for s in parents}
            paths, texts = [], []
            for sample in ds:
                # Metadata-only dataset; its audio is in separate, pinned files.
                filename = sample["file_name"]
                if Path(filename).name != filename:
                    raise ValueError(f"Unexpected audio filename: {filename!r}")
                paths.append(
                    hf_hub_download(spec.repo, f"audio/{filename}", repo_type="dataset", revision=spec.revision)
                )
                texts.append(parent_text[str(sample["parent_id"])])
            ds = ds.add_column("audio", paths).add_column("text", texts)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        # Access metadata without decoding or copying audio into Python lists.
        rows = reference_rows(ds.remove_columns("audio"))
        ds = ds.select([r.pop("source_index") for r in rows])
        ds = ds.select_columns([k for k in ("id", "audio", "text", "parent_id", "chunk_index") if k in ds.column_names])
        ds.save_to_disk(self.out_dataset.get_path())
        with open(self.out_references.get_path(), "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(self.out_manifest.get_path(), "w") as f:
            json.dump(
                dict(
                    dataset=asdict(spec),
                    num_segments=len(rows),
                    num_scoring_units=len({r["group_id"] for r in rows}),
                    sample_rate=16000,
                    normalization="updated-hf",
                    audio_reencoded=False,
                ),
                f,
                indent=2,
            )
