"""Sisyphus jobs: Seamless Interaction (arXiv 2606.11167 data) -> durable audio -> per-axis Mimi codes.

Four-stage pipeline (see ``projects/2026-01-speech-llm/rl_alignment.md``, Workstream 1), parameterized by
``(label, split)`` so it scales from ``improvised``/``dev`` to naturalistic + train later:

  SeamlessInteractionIndexJob  -- filelist.csv -> paired interactions -> shards/<k>.json
  SeamlessAudioDownloadJob     -- (sharded) hf download -> extract wav+json -> 24k mono FLAC (durable)
  MergeSeamlessAudio           -- symlink-merge shard audio/transcripts + concat manifests
  SeamlessSegmentEncodeJob     -- (GPU) audio+transcripts -> per-axis mimi.encode -> codes + index.jsonl

Downloads use only HuggingFace (``hf_hub_download``, reachable from compute nodes); tars stream into
node-local ``/tmp`` and are deleted after extraction (only the resampled audio is kept). The encode reuses
the base-Moshi mimi (``kyutai/moshiko-pytorch-bf16``) so codes match the policy. Workers live in the
moshi_family lib (``rl/seamless_*_main.py``) and run in ``moshi_family_venv``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import urllib.request
from collections import defaultdict

from sisyphus import Job, Task, tk

from .common import run_worker_script
from .inference_harness import _moshi_family_pythonpath

FILELIST_URL = "https://raw.githubusercontent.com/facebookresearch/seamless_interaction/main/assets/filelist.csv"


class SeamlessInteractionIndexJob(Job):
    """Select fully-paired interactions for ``(label, split)`` and pack them into ``num_shards`` shards.

    Reads the official ``filelist.csv`` (file_id -> label/split/batch_idx/archive_idx), keeps interactions
    whose BOTH participants are present, caps to ``max_interactions``, and round-robins them into shards.
    Emits ``shards/<k>.json`` (each: label/split + its interactions with per-file archive locations)."""

    __sis_hash_exclude__ = {"seed": 0, "rqmt": None, "filelist_url": FILELIST_URL}

    def __init__(
        self,
        *,
        label: str = "improvised",
        split: str = "dev",
        num_shards: int = 4,
        max_interactions: int | None = None,
        seed: int = 0,
        filelist_url: str = FILELIST_URL,
        rqmt: dict | None = None,
    ):
        self.label = label
        self.split = split
        self.num_shards = num_shards
        self.max_interactions = max_interactions
        self.seed = seed
        self.filelist_url = filelist_url
        self.out_dir = self.output_path("shards", directory=True)
        self.rqmt = rqmt or {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        data = urllib.request.urlopen(self.filelist_url, timeout=120).read().decode()
        by_key: dict[str, list[dict]] = defaultdict(list)
        for row in csv.DictReader(io.StringIO(data)):
            if row["label"] != self.label or row["split"] != self.split:
                continue
            fid = row["file_id"]
            key = "_".join(fid.split("_")[:3])  # V.._S.._I.. (interaction)
            by_key[key].append(
                {"file_id": fid, "batch_idx": int(row["batch_idx"]), "archive_idx": int(row["archive_idx"])}
            )
        paired = [
            {"key": key, "files": sorted(files, key=lambda f: f["file_id"])[:2]}
            for key, files in by_key.items()
            if len(files) >= 2
        ]
        paired.sort(key=lambda it: it["key"])
        random.Random(self.seed).shuffle(paired)
        if self.max_interactions is not None:
            paired = paired[: self.max_interactions]
        shards: list[list[dict]] = [[] for _ in range(self.num_shards)]
        for i, it in enumerate(paired):
            shards[i % self.num_shards].append(it)
        out = self.out_dir.get()
        for k, interactions in enumerate(shards):
            with open(os.path.join(out, f"{k}.json"), "w") as f:
                json.dump({"label": self.label, "split": self.split, "interactions": interactions}, f)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(
                {"num_shards": self.num_shards, "n_interactions": len(paired), "per_shard": [len(s) for s in shards]},
                f,
            )
        print(f"[index] {self.label}/{self.split}: {len(paired)} paired interactions -> {self.num_shards} shards")


class SeamlessAudioDownloadJob(Job):
    """Download one shard's archives, extract+resample both participants' audio -> durable 24k FLAC + json."""

    __sis_hash_exclude__ = {"sample_rate": 24000, "rqmt": None}

    def __init__(
        self,
        *,
        venv_python_path,
        shards_dir: tk.Path,
        shard_index: int,
        sample_rate: int = 24000,
        rqmt: dict | None = None,
    ):
        self.venv_python_path = venv_python_path
        self.shards_dir = shards_dir
        self.shard_index = shard_index
        self.sample_rate = sample_rate
        self.out_dir = self.output_path("data", directory=True)
        self.rqmt = rqmt or {"cpu": 4, "mem": 16, "time": 12}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        lib_parent = _moshi_family_pythonpath()
        script = os.path.join(lib_parent, "moshi_family", "rl", "seamless_download_main.py")
        shard_json = os.path.join(self.shards_dir.get(), f"{self.shard_index}.json")
        args = [
            "--shard_json",
            shard_json,
            "--out_dir",
            self.out_dir.get(),
            "--sample_rate",
            self.sample_rate,
        ]
        run_worker_script(
            self.venv_python_path.get(),
            script,
            args,
            log_label=f"Seamless download shard {self.shard_index}",
            with_hf_home=True,
            extra_env={"PYTHONPATH": lib_parent},
        )


class MergeSeamlessAudio(Job):
    """Symlink-merge sharded download outputs (audio/ + transcripts/) + concatenate manifests."""

    def __init__(self, *, in_dirs: list[tk.Path]):
        self.in_dirs = in_dirs
        self.out_dir = self.output_path("data", directory=True)

    def tasks(self):
        yield Task("merge", mini_task=True)

    def merge(self):
        out = self.out_dir.get()
        for sub in ("audio", "transcripts"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        manifest_lines: list[str] = []
        for ind in self.in_dirs:
            d = ind.get()
            for sub in ("audio", "transcripts"):
                sd = os.path.join(d, sub)
                if not os.path.isdir(sd):
                    continue
                for name in os.listdir(sd):
                    link = os.path.join(out, sub, name)
                    if not os.path.exists(link):
                        os.symlink(os.path.join(sd, name), link)
            mf = os.path.join(d, "manifest.jsonl")
            if os.path.exists(mf):
                manifest_lines.extend(ln for ln in open(mf).read().splitlines() if ln.strip())
        with open(os.path.join(out, "manifest.jsonl"), "w") as f:
            for ln in manifest_lines:
                f.write(ln + "\n")
        print(f"[merge] {len(manifest_lines)} interactions merged")


class SeamlessSegmentEncodeJob(Job):
    """Per-axis Mimi-encode the merged Seamless audio -> ``codes/`` + ``index.jsonl`` (the RL data_dir)."""

    __sis_hash_exclude__ = {"sample_rate": 24000, "vad_backend": "silero", "seed": 0, "rqmt": None}

    def __init__(
        self,
        *,
        venv_python_path,
        audio_dir: tk.Path,
        hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        per_axis_cap: int = 2000,
        both_directions: bool = True,
        vad_backend: str = "silero",
        sample_rate: int = 24000,
        seed: int = 0,
        rqmt: dict | None = None,
    ):
        self.venv_python_path = venv_python_path
        self.audio_dir = audio_dir
        self.hf_repo = hf_repo
        self.per_axis_cap = per_axis_cap
        self.both_directions = both_directions
        self.vad_backend = vad_backend
        self.sample_rate = sample_rate
        self.seed = seed
        self.out_dir = self.output_path("data", directory=True)
        self.rqmt = rqmt or {"gpu": 1, "cpu": 6, "mem": 32, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        lib_parent = _moshi_family_pythonpath()
        script = os.path.join(lib_parent, "moshi_family", "rl", "seamless_encode_main.py")
        args = [
            "--audio_dir",
            self.audio_dir.get(),
            "--out_dir",
            self.out_dir.get(),
            "--hf_repo",
            self.hf_repo,
            "--per_axis_cap",
            self.per_axis_cap,
            "--vad_backend",
            self.vad_backend,
            "--seed",
            self.seed,
        ]
        if self.both_directions:
            args.append("--both_directions")
        run_worker_script(
            self.venv_python_path.get(),
            script,
            args,
            log_label="Seamless segment encode",
            with_hf_home=True,
            extra_env={"PYTHONPATH": lib_parent},
        )


def seamless_codes_data_dir(
    *,
    venv_python_path,
    label: str = "improvised",
    split: str = "dev",
    num_shards: int = 4,
    max_interactions: int | None = None,
    per_axis_cap: int = 2000,
    both_directions: bool = True,
    vad_backend: str = "silero",
    hf_repo: str = "kyutai/moshiko-pytorch-bf16",
) -> tk.Path:
    """Build index -> sharded download -> merge -> encode and return the per-axis codes ``data_dir``
    (consumed by ``rl.segments.CodesSegmentSource`` via ``RLFinetune(segment_source="codes")``)."""
    index = SeamlessInteractionIndexJob(
        label=label, split=split, num_shards=num_shards, max_interactions=max_interactions
    )
    tk.register_output(f"seamless/{label}_{split}/shards", index.out_dir)
    dls = [
        SeamlessAudioDownloadJob(venv_python_path=venv_python_path, shards_dir=index.out_dir, shard_index=k)
        for k in range(num_shards)
    ]
    merged = MergeSeamlessAudio(in_dirs=[d.out_dir for d in dls])
    tk.register_output(f"seamless/{label}_{split}/audio", merged.out_dir)
    encode = SeamlessSegmentEncodeJob(
        venv_python_path=venv_python_path,
        audio_dir=merged.out_dir,
        hf_repo=hf_repo,
        per_axis_cap=per_axis_cap,
        both_directions=both_directions,
        vad_backend=vad_backend,
    )
    tk.register_output(f"seamless/{label}_{split}/codes", encode.out_dir)
    return encode.out_dir
