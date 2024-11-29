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
