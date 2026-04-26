from pathlib import Path
from sisyphus import Job, Task, tk
import os
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict,
    concatenate_datasets,
)
import json


class HfDownloadSplit(Job):
    def __init__(self, *, dataset_name: str, split: str):
        self.dataset_name = dataset_name
        self.split = split
        self.out_hf = self.output_path("hf_split", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        dataset = load_dataset(self.dataset_name, split=self.split)
        dataset.save_to_disk(self.out_hf.get())


class HfMergeShards(Job):
    def __init__(self, *, shard_paths: list[tk.Path], add_shard_to_id: bool = False):
        self.shard_paths = shard_paths
        self.add_shard_to_id = add_shard_to_id
        self.out_hf = self.output_path("merged_dataset", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        if "add_shard_to_id" in d and not d["add_shard_to_id"]:
            del d["add_shard_to_id"]
        else:
            d["version"] = 2
        return super().hash(d)

    def run(self):
        datasets = [load_from_disk(p.get()) for p in self.shard_paths]
        if self.add_shard_to_id:
            for i, p in enumerate(self.shard_paths):
                ds = datasets[i]
                assert "id" in ds.column_names
                # Prefix the existing ID with the shard index
                datasets[i] = ds.map(
                    lambda x: {"id": f"shard_{i}_{x['id']}"}, num_proc=4
                )

        merged = concatenate_datasets(datasets)
        merged.save_to_disk(self.out_hf.get())
