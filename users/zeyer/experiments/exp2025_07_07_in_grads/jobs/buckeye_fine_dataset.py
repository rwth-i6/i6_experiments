"""Convert the alexwengg/buckeye fine-resolution forced-align benchmark to our pipeline schema.

The raw repo is manifest.json + audio/*.wav:
fine float-ms word timestamps, hand-corrected, pre-segmented into 2-25 s utterances.
This job rewrites it as a parquet HF dataset with the schema the rest of the pipeline reads
(``audio{array, sampling_rate}`` + ``word_detail{utterance, start, stop}``),
laid out as a hub-cache snapshot so ``load_dataset`` / ``get_content_dir_from_hub_cache_dir``
consume it unchanged.

start/stop are stored as SAMPLE indices (``round(start_ms * sr / 1000)``),
so ``dataset_offset_factors=1`` like TIMIT -- 62.5 us resolution, not the 62.5 ms of nh0znoisung.
Non-word markers (``<SIL>``/``<IVER>``/``<VOCNOISE>``/``{...}``) are dropped.
Audio is stored as a plain float array (no datasets.Audio() feature) to avoid any codec dependency.
"""

from typing import Optional
from sisyphus import Job, Task, tk


class BuildBuckeyeFineDatasetJob(Job):
    """alexwengg/buckeye manifest+wav -> parquet dataset in our schema. See module docstring."""

    def __init__(
        self,
        *,
        raw_dir: tk.Path,
        split_name: str = "test",
        num_shards: int = 8,
        max_segments: Optional[int] = None,
        max_duration_s: Optional[float] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.raw_dir = raw_dir
        self.split_name = split_name
        self.num_shards = num_shards
        # Keep only short segments (duration <= max_duration_s): the full set is mostly capped at
        # ~25 s, which (a) at char-level exceeds Whisper's 448-token decoder context and (b) makes the
        # per-token-backward extracts too slow. <=20 s segments have <=385 chars (char-level fits) and
        # are much cheaper. max_segments is an additional first-N cap.
        self.max_duration_s = max_duration_s
        self.max_segments = max_segments
        self.returnn_root = returnn_root
        self.rqmt = {"cpu": 4, "mem": 48, "time": 4}
        self.out_hub_cache_dir = self.output_path("hub_cache", directory=True)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import json

        from i6_experiments.users.zeyer.external_models.huggingface import (
            set_hf_offline_mode,
            get_content_dir_from_hub_cache_dir,
        )

        set_hf_offline_mode()

        import numpy as np
        import soundfile as sf
        import datasets

        content = get_content_dir_from_hub_cache_dir(self.raw_dir)
        with open(os.path.join(content, "manifest.json")) as f:
            manifest = json.load(f)
        samples = manifest["samples"]
        if self.max_duration_s is not None:
            samples = [s for s in samples if s["duration_s"] <= self.max_duration_s]
        if self.max_segments is not None:
            samples = samples[: self.max_segments]
        print(f"kept {len(samples)} segments after filters", flush=True)
        print(f"{len(samples)} segments in manifest", flush=True)

        feats = datasets.Features(
            {
                "id": datasets.Value("string"),
                "speaker": datasets.Value("string"),
                "audio": {
                    "array": datasets.Sequence(datasets.Value("float32")),
                    "sampling_rate": datasets.Value("int64"),
                },
                "word_detail": {
                    "utterance": datasets.Sequence(datasets.Value("string")),
                    "start": datasets.Sequence(datasets.Value("int64")),
                    "stop": datasets.Sequence(datasets.Value("int64")),
                },
            }
        )

        # hub-cache snapshot layout: <hub>/datasets--local--buckeye_fine/{refs/main, snapshots/<ref>/data/*.parquet}
        ref = "buckeyefine0"
        root = os.path.join(self.out_hub_cache_dir.get_path(), "datasets--local--buckeye_fine")
        data_dir = os.path.join(root, "snapshots", ref, "data")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(root, "refs"), exist_ok=True)
        with open(os.path.join(root, "refs", "main"), "w") as f:
            f.write(ref)

        n = len(samples)
        shard_size = (n + self.num_shards - 1) // self.num_shards
        n_words = n_segs = n_skipped = 0
        for shard in range(self.num_shards):
            recs = []
            for s in samples[shard * shard_size : (shard + 1) * shard_size]:
                wav, sr = sf.read(os.path.join(content, s["audio"]))
                wav = np.asarray(wav, dtype=np.float32)
                if wav.ndim > 1:
                    wav = wav[:, 0]
                words = [w for w in s["words"] if not (w["word"].startswith("<") or w["word"].startswith("{"))]
                if not words:
                    n_skipped += 1
                    continue
                utt = [w["word"] for w in words]
                start = [int(round(w["start_ms"] * sr / 1000.0)) for w in words]
                stop = [int(round(w["end_ms"] * sr / 1000.0)) for w in words]
                n_words += len(utt)
                n_segs += 1
                recs.append(
                    {
                        "id": s["id"],
                        "speaker": s.get("speaker", ""),
                        "audio": {"array": wav, "sampling_rate": int(sr)},
                        "word_detail": {"utterance": utt, "start": start, "stop": stop},
                    }
                )
            ds = datasets.Dataset.from_list(recs, features=feats)
            out_pq = os.path.join(data_dir, f"{self.split_name}-{shard:05d}-of-{self.num_shards:05d}.parquet")
            ds.to_parquet(out_pq)
            print(f"shard {shard}: {len(recs)} segs -> {os.path.basename(out_pq)}", flush=True)

        print(f"done: {n_segs} segs, {n_words} words, {n_skipped} skipped (no real words)", flush=True)
