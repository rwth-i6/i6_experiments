"""TextDict utilities for the no-bliss (dataset/store-reference) scoring paths."""

import ast
import gzip

from sisyphus import Job, Task, tk

__all__ = ["IntersectTextDictJob"]


def _read_text_dict(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


class IntersectTextDictJob(Job):
    """Filter a TextDict to the seq_tags present in another TextDict.

    sclite counts every hyp-less reference utterance as pure deletions, so an unfiltered superset
    reference silently inflates WER.
    """

    def __init__(self, *, text_dict: tk.Path, keys_from: tk.Path):
        self.text_dict = text_dict
        self.keys_from = keys_from
        self.out_txt = self.output_path("out.txt.gz")
        self.out_num_dropped = self.output_var("num_dropped")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1}, mini_task=True)

    def run(self):
        ref = _read_text_dict(self.text_dict.get_path())
        keys = set(_read_text_dict(self.keys_from.get_path()).keys())
        kept = {k: v for k, v in ref.items() if k in keys}
        missing = keys - set(kept)
        # a hyp without reference cannot be scored at all, fail loud instead of skewing wer
        assert not missing, (
            f"{len(missing)} hyp seqs missing from the reference, e.g. {sorted(missing)[:5]}"
        )
        with gzip.open(self.out_txt.get_path(), "wt", encoding="utf-8") as f:
            f.write("{\n")
            for k, v in kept.items():
                f.write(f"{k!r}: {v!r},\n")
            f.write("}\n")
        self.out_num_dropped.set(len(ref) - len(kept))
