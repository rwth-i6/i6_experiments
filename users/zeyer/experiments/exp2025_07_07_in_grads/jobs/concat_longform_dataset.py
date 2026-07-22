"""Concatenate short dataset entries into longer pseudo-recordings,
for long-form chunk-align experiments (:mod:`chunk_segmentation`).

Generic: works on any dataset in our standard schema
(``audio{array,sampling_rate}`` + ``word_detail{utterance,start,stop}``),
e.g. TIMIT or :class:`buckeye_fine_dataset.BuildBuckeyeFineDatasetJob`'s output.
Groups consecutive same-speaker entries (in original dataset order)
until reaching ``target_duration_s``,
concatenates their audio with a fixed silence gap,
and shifts word offsets accordingly.
Never splits an individual entry,
so groups can slightly exceed the target when a single entry is already close to it.

Output uses ``dataset_offset_factors=1`` (start/stop stored directly as sample indices),
like TIMIT and the fine-resolution Buckeye dataset.
"""

from typing import Optional
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class ConcatenateForLongFormJob(Job):
    """See module docstring."""

    __sis_version__ = 1

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        speaker_key: str = "speaker",
        target_duration_s: float = 120.0,
        silence_gap_s: float = 0.5,
        num_shards: int = 8,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param dataset_dir: hub cache dir (e.g. from BuildBuckeyeFineDatasetJob or a TIMIT download).
        :param dataset_key: e.g. "val", "test".
        :param speaker_key: dataset column used to group entries --
            only concatenate same-speaker entries, in original dataset order.
            E.g. "speaker" (fine-resolution Buckeye), "speaker_id" (TIMIT).
        :param target_duration_s: keep appending same-speaker entries to the current group
            until adding the next one would exceed this; then start a new group.
        :param silence_gap_s: silence inserted between concatenated entries,
            so the chunker doesn't see a hard word-to-word splice with zero gap.
        :param num_shards:
        :param returnn_root:
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.speaker_key = speaker_key
        self.target_duration_s = target_duration_s
        self.silence_gap_s = silence_gap_s
        self.num_shards = num_shards
        self.returnn_root = returnn_root

        self.rqmt = {"cpu": 4, "mem": 16, "time": 2}
        self.out_hub_cache_dir = self.output_path("hub_cache", directory=True)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import datasets
        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        ds_split = ds[self.dataset_key]
        print(f"Input: {len(ds_split)} entries")

        # Group indices by speaker, preserving original within-speaker order.
        by_speaker = {}
        for i, spk in enumerate(ds_split[self.speaker_key]):
            by_speaker.setdefault(spk, []).append(i)

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

        records = []

        def flush(group_entries, speaker):
            if not group_entries:
                return
            sr = group_entries[0]["audio"]["sampling_rate"]
            gap = np.zeros(int(round(self.silence_gap_s * sr)), dtype=np.float32)
            audio_parts = []
            words, starts, stops = [], [], []
            offset = 0
            for k, e in enumerate(group_entries):
                a = np.asarray(e["audio"]["array"], dtype=np.float32)
                if k > 0:
                    audio_parts.append(gap)
                    offset += len(gap)
                audio_parts.append(a)
                words.extend(e["word_detail"]["utterance"])
                starts.extend(int(s) + offset for s in e["word_detail"]["start"])
                stops.extend(int(s) + offset for s in e["word_detail"]["stop"])
                offset += len(a)
            audio = np.concatenate(audio_parts)
            records.append(
                {
                    "id": f"{speaker}-{len(records)}",
                    "speaker": speaker,
                    "audio": {"array": audio, "sampling_rate": sr},
                    "word_detail": {"utterance": words, "start": starts, "stop": stops},
                }
            )

        for spk, idxs in by_speaker.items():
            group_entries, group_dur = [], 0.0
            for i in idxs:
                e = ds_split[i]
                dur = len(e["audio"]["array"]) / e["audio"]["sampling_rate"]
                if group_entries and group_dur + dur > self.target_duration_s:
                    flush(group_entries, spk)
                    group_entries, group_dur = [], 0.0
                group_entries.append(e)
                group_dur += dur + self.silence_gap_s
            flush(group_entries, spk)

        durs = [len(r["audio"]["array"]) / r["audio"]["sampling_rate"] for r in records]
        print(f"Built {len(records)} long-form recordings from {len(ds_split)} entries ({len(by_speaker)} speakers)")
        print(f"Duration: min={min(durs):.1f}s max={max(durs):.1f}s mean={sum(durs) / len(durs):.1f}s")

        # hub-cache snapshot layout, same convention as BuildBuckeyeFineDatasetJob.
        ref = "concatlongform0"
        root = os.path.join(self.out_hub_cache_dir.get_path(), "datasets--local--concat_longform")
        data_dir = os.path.join(root, "snapshots", ref, "data")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(root, "refs"), exist_ok=True)
        with open(os.path.join(root, "refs", "main"), "w") as f:
            f.write(ref)

        n = len(records)
        shard_size = max(1, (n + self.num_shards - 1) // self.num_shards)
        for shard in range(self.num_shards):
            chunk = records[shard * shard_size : (shard + 1) * shard_size]
            if not chunk:
                continue
            ds_out = datasets.Dataset.from_list(chunk, features=feats)
            out_pq = os.path.join(data_dir, f"{self.dataset_key}-{shard:05d}-of-{self.num_shards:05d}.parquet")
            ds_out.to_parquet(out_pq)
            print(f"shard {shard}: {len(chunk)} recs -> {os.path.basename(out_pq)}")

        print("done")
