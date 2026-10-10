from typing import Optional

from sisyphus import Job, Task, tk


class ExtractTextFromArrowDatasetJob(Job):
    """Extract a text column from an on-disk HF dataset dir to gzipped text."""

    def __init__(
        self,
        dataset_dir: tk.Path,
        *,
        split: Optional[str] = "train",
        column_name: str = "text",
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.column_name = column_name
        self.rqmt = {"cpu": 2, "mem": 16, "time": 6}
        self.out_text = self.output_path("text.txt.gz")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        import gzip
        from datasets import load_from_disk

        ds = load_from_disk(self.dataset_dir.get_path())
        if self.split is not None and hasattr(ds, "keys"):
            assert self.split in ds, f"split {self.split!r} not in {list(ds.keys())}"
            ds = ds[self.split]
        assert self.column_name in ds.column_names, (
            f"column {self.column_name!r} not in {ds.column_names}"
        )
        ds = ds.select_columns([self.column_name])
        with gzip.open(self.out_text.get_path(), "wt", encoding="utf-8") as f:
            for item in ds:
                f.write(item[self.column_name])
                f.write("\n")
