"""
Reference text-dict for WER scoring: reads {id: transcript} directly from the local HF parquet
(offline, text column only, no audio). Output is a RETURNN text-dict (``{seq_tag: text}`` python
repr) consumable by ``TextDictToStmJob``. Seq tags ('id') match the HuggingFaceDataset seq tags, so
they align with the greedy decoder's ``search_out.py`` hyps.
"""

from __future__ import annotations

from typing import Sequence

from sisyphus import Job, Task

from .datasets import parquet_files


class HFTextDictJob(Job):
    def __init__(self, src_splits: Sequence[str]):
        self.src_splits = tuple(src_splits)
        self.out_text_dict = self.output_path("text_dict.py")
        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import pyarrow.parquet as pq

        files = parquet_files(self.src_splits)
        with open(self.out_text_dict.get_path(), "wt", encoding="utf-8") as f:
            f.write("{\n")
            for pf in files:
                t = pq.read_table(pf, columns=["id", "text"])
                ids = t.column("id").to_pylist()
                txts = t.column("text").to_pylist()
                for i, x in zip(ids, txts):
                    f.write("%s: %s,\n" % (repr(i), repr(x)))
            f.write("}\n")
