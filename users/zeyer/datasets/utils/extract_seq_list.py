"""
Extract/generate seq lists.
"""

from typing import Optional, Union, Any, Dict
from sisyphus import tk, Task, Job
from i6_core.util import uopen


class ExtractSeqListJob(Job):
    """
    Takes any dataset dict, and extracts all seq tags from it.
    """

    def __init__(
        self,
        *,
        returnn_dataset: Dict[str, Any],  # to get all seq tags
        returnn_root: Optional[tk.Path] = None,
    ):
        self.returnn_dataset = returnn_dataset
        self.returnn_root = returnn_root

        self.out_seq_list = self.output_path("out_seq_list.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        import sys
        import os
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(1, returnn_root.get_path())

        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset
        from returnn.log import log

        config = Config()
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 4
        log.init_by_config(config)

        import tree

        dataset_dict = self.returnn_dataset
        dataset_dict = tree.map_structure(lambda x: x.get_path() if isinstance(x, tk.Path) else x, dataset_dict)
        print("RETURNN dataset dict:", dataset_dict)
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)

        with open(self.out_seq_list.get_path(), "w") as f:
            for seq_tag in dataset.get_all_tags():
                print(seq_tag, file=f)


class ExtractNumLinesFromTextFileJob(Job):
    """
    Extracts the number of lines from a (potentially gzipped) text file.
    """

    def __init__(self, *, text_file: tk.Path, skip_empty_lines: bool = True):
        self.text_file = text_file
        self.skip_empty_lines = skip_empty_lines
        self.out_num_lines = self.output_var("out_num_lines.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        num_lines = 0
        with uopen(self.text_file.get_path(), "rt") as f:
            for line in f:
                if self.skip_empty_lines and not line.strip():
                    continue
                num_lines += 1
        self.out_num_lines.set(num_lines)


class WriteLmDatasetSeqListJob(Job):
    """
    LmDataset has simple seq tags like "line-{line_nr}".
    So to write out the seq list, we just need to know the number of lines.
    """

    def __init__(self, *, num_lines: Union[int, tk.Variable]):
        super().__init__()
        assert isinstance(num_lines, (int, tk.Variable))
        self.num_lines = num_lines
        self.out_seq_list = self.output_path("out_seq_list.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        num_lines = self.num_lines
        if isinstance(num_lines, tk.Variable):
            num_lines = num_lines.get()
        assert isinstance(num_lines, int)
        with open(self.out_seq_list.get_path(), "w") as f:
            for line_nr in range(num_lines):
                print(f"line-{line_nr}", file=f)


class WriteSeqListFromShuffledJob(Job):
    """
    Assuming that some dataset was split and shuffled using
    :class:`i6_core.corpus.segments.ShuffleAndSplitSegmentsJob`,
    and then merged again (via some more complex pipeline,
    e.g. involving :class:`i6_core.returnn.oggzip.BlissToOggZipJob`),
    we might get sequence tags like `"librispeech-lm-part{split_key}/recording_{split_seq_idx}/line_{split_seq_idx}"`.

    We write those seq tags out to a seq list file.
    This file could be used for :class:`returnn.datasets.meta.MetaDataset` ``seq_list_file``.

    The code is based on :class:`i6_core.corpus.segments.ShuffleAndSplitSegmentsJob`.
    """

    def __init__(
        self,
        *,
        seq_tag_template: str,
        num_seqs: Union[int, tk.Variable],
        split: Dict[str, float],
        shuffle=True,
        shuffle_seed=0x3C5EA3E47D4E0077,
    ):
        assert isinstance(split, dict)
        assert all(s > 0 for s in split.values())
        assert abs(sum(split.values()) - 1.0) < 1e-10

        self.seq_tag_template = seq_tag_template
        self.num_seqs = num_seqs
        self.split = split
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        self.out_segments = {k: self.output_path(f"{k}.segments") for k in self.split.keys()}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import random
        import itertools as it

        n = self.num_seqs
        if isinstance(n, tk.Variable):
            n = n.get()
        assert isinstance(n, int)
        segments = list(range(n))

        if self.shuffle:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(segments)

        ordered_keys = sorted(self.split.keys())
        split_idx = [0] + [int(n * c) for c in it.accumulate(self.split[k] for k in ordered_keys)]
        split_idx[-1] = n  # just in case we get numeric errors that drop the last element

        for i, split_key in enumerate(ordered_keys):
            with uopen(self.out_segments[split_key].get_path(), "wt", encoding="utf-8") as f:
                for split_seq_idx in range(split_idx[i], split_idx[i + 1]):
                    f.write(self.seq_tag_template.format(split_key=split_key, split_seq_idx=split_seq_idx) + "\n")
