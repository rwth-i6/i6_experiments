from typing import Dict, Optional, Sequence

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


class ArrowDatasetToTextDictJob(Job):
    """
    Writes the texts of splits of an on-disk HF dataset as a TextDict, the id column as the sequence tags.
    """

    def __init__(
        self, dataset_dir: tk.Path, *, splits: Sequence[str], id_column: str = "id", text_column: str = "text"
    ):
        """
        :param dataset_dir: the dataset saved by ``save_to_disk``
        :param splits: the splits whose texts the TextDict holds
        :param id_column: the column of the sequence tags
        :param text_column: the column of the texts
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.splits = list(splits)
        self.id_column = id_column
        self.text_column = text_column
        self.rqmt = {"cpu": 2, "mem": 16, "time": 6}
        self.out_text_dict = self.output_path("text_dict.py.gz")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        import gzip
        from datasets import load_from_disk

        dataset = load_from_disk(self.dataset_dir.get_path())
        texts: Dict[str, str] = {}
        for split in self.splits:
            for item in dataset[split].select_columns([self.id_column, self.text_column]):
                tag = str(item[self.id_column])
                assert tag not in texts, f"the id {tag} is in two splits"
                texts[tag] = item[self.text_column]
        with gzip.open(self.out_text_dict.get_path(), "wt", encoding="utf-8") as f:
            f.write("{\n")
            for tag, text in texts.items():
                f.write(f"{tag!r}: {text!r},\n")
            f.write("}\n")
