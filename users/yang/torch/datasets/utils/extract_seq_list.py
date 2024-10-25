from typing import Optional, Any, Dict
from sisyphus import tk, Task, Job


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
