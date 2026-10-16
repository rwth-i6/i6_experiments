import json
import os

from sisyphus import Job, Task, tk
import pyarrow as pa
from datasets import (
    Dataset,
    load_dataset,
    load_from_disk,
    concatenate_datasets,
)
from datasets.table import InMemoryTable
from datasets.utils.py_utils import asdict


def save_linked(merged: Dataset, parts: list[Dataset], out_path: str) -> bool:
    """Write ``merged`` (= ``concatenate_datasets(parts)``) to ``out_path`` WITHOUT copying data: True if done.

    The result is what ``merged.save_to_disk(out_path)`` writes -- the same ``state.json`` (fingerprint, format,
    split) and ``dataset_info.json`` (datasets' own serialisation, ``arrow_dataset.Dataset.save_to_disk``) -- except
    that its ``data-*.arrow`` files are relative symlinks to the parts' own arrow files instead of a re-written copy,
    so ``load_from_disk`` reads the identical rows, features and fingerprint (guard ``check_zero_copy_merge.py``).
    User 2026-10-16: the merge copies were 1.8 TiB of duplicates; "new behaviour exactly matches old behaviour".

    Returns False -- and writes nothing -- whenever that equality is not guaranteed; the caller then runs the old
    ``save_to_disk`` copy: a part with an indices mapping, a part whose table is not exactly its own saved arrow files,
    or arrow schemas that ``concatenate_datasets`` would have to unify. Never link to a part that lives in a temp dir:
    the links must outlive this call (callers pass other jobs' outputs only).
    """
    files = []
    for p in parts:
        if p._indices is not None or not p.cache_files:
            return False
        if p.data.schema != merged.data.schema:
            return False
        own = [c["filename"] for c in p.cache_files]
        state = os.path.join(os.path.dirname(own[0]), "state.json")
        try:
            listed = [f["filename"] for f in json.load(open(state))["_data_files"]]
        except (OSError, ValueError, KeyError):
            return False
        if [os.path.basename(f) for f in own] != listed or len({os.path.dirname(f) for f in own}) != 1:
            return False  # the table is not exactly this part's saved files (in order)
        if not all(os.path.isfile(f) for f in own):
            return False
        files += own
    if merged.data.schema != parts[0].data.schema or len(merged) != sum(len(p) for p in parts):
        return False
    os.makedirs(out_path, exist_ok=True)
    out_real = os.path.realpath(out_path)
    n = len(files)
    names = [f"data-{k:05d}-of-{n:05d}.arrow" for k in range(n)]
    for name, f in zip(names, files):
        link = os.path.join(out_path, name)
        if os.path.lexists(link):
            os.unlink(link)  # a rerun of this job: replace our own previous link
        os.symlink(os.path.relpath(os.path.realpath(f), out_real), link)
    # exactly datasets 4.8.4's save_to_disk serialisation of the two JSON files
    state = {
        key: merged.__dict__[key]
        for key in ["_fingerprint", "_format_columns", "_format_kwargs", "_format_type", "_output_all_columns"]
    }
    state["_split"] = str(merged.split) if merged.split is not None else merged.split
    state["_data_files"] = [{"filename": name} for name in names]
    with open(os.path.join(out_path, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    info = asdict(merged._info)  # datasets own asdict (drops None fields), as save_to_disk uses
    with open(os.path.join(out_path, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump({key: info[key] for key in sorted(info)}, f, indent=2)
    back = load_from_disk(out_path)
    assert len(back) == len(merged) and back.features == merged.features, out_path
    # (not the fingerprint: load_from_disk derives a fresh one on reload, for this and for a save_to_disk copy alike;
    # state.json carries merged._fingerprint exactly as save_to_disk writes it)
    assert back.data.schema == merged.data.schema, out_path
    return True


class HfDownloadSplit(Job):
    def __init__(self, *, dataset_name: str, split: str, config_name: str | None = None):
        self.dataset_name = dataset_name
        self.split = split
        self.config_name = config_name
        self.out_hf = self.output_path("hf_split", directory=True)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        if not d.get("config_name"):
            d.pop("config_name", None)
        return super().hash(d)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        args = [self.config_name] if self.config_name else []
        dataset = load_dataset(self.dataset_name, *args, split=self.split)
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
                # Prefix the existing ID with the shard index. Rewrite only that column: ds.map() rewrote every row
                # (audio included) into cache-*.arrow inside the INPUT shard's output dir (562 GiB by 2026-10-15;
                # guard check_hf_merge_no_cache.py).
                ids = pa.array([f"shard_{i}_{x}" for x in ds["id"]], type=pa.string())
                # on the plain pyarrow table: a multi-file shard is a ConcatenationTable, whose set_column raises
                # NotImplementedError (check_zero_copy_merge, multi-file shards). Buffers stay memory-mapped.
                table = ds.data.table.set_column(ds.column_names.index("id"), "id", ids)
                datasets[i] = Dataset(InMemoryTable(table), info=ds.info.copy(), split=ds.split)

        merged = concatenate_datasets(datasets)
        # Zero-copy when the merged rows are exactly the shards' saved files (always, without the id rewrite: the
        # rewritten id column only exists in memory); otherwise the old copy.
        if self.add_shard_to_id or not save_linked(merged, datasets, self.out_hf.get()):
            merged.save_to_disk(self.out_hf.get())
