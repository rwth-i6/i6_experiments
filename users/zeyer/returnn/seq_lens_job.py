"""
Extract seq lens job from RETURNN dataset.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from sisyphus import Job, Task
from i6_core.returnn import ReturnnConfig


class ExtractSeqLensJob(Job):
    """
    Extracts sequence lengths from a dataset for one specific key.

    Also see: :class:`i6_experiments.users.schmitt.corpus.statistics.GetSeqLenFileJob`

    TODO this here can be removed once https://github.com/rwth-i6/i6_core/pull/522 is merged
    """

    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        dataset: Dict[str, Any],
        post_dataset: Optional[Dict[str, Any]] = None,
        *,
        key: str,
        format: str,
        returnn_config: Optional[ReturnnConfig] = None,
    ):
        """
        :param dataset: dict for :func:`returnn.datasets.init_dataset`
        :param post_dataset: extension of the dataset dict, which is not hashed
        :param key: e.g. "data", "classes" or whatever the dataset provides
        :param format: "py" or "txt"
        :param returnn_config: for the RETURNN global config.
            This is optional and only needed if you use any custom functions (e.g. audio pre_process)
            which expect some configuration in the global config.
        """
        super().__init__()
        self.dataset = dataset
        self.post_dataset = post_dataset
        self.key = key
        assert format in {"py", "txt"}
        self.format = format
        self.returnn_config = returnn_config

        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_file = self.output_path(f"seq_lens.{format}")

        self.rqmt = {"gpu": 0, "cpu": 1, "mem": 4, "time": 1}

    @classmethod
    def hash(cls, parsed_args):
        """hash"""
        parsed_args = parsed_args.copy()
        parsed_args.pop("post_dataset")
        if not parsed_args["returnn_config"]:
            parsed_args.pop("returnn_config")
        return super().hash(parsed_args)

    def tasks(self):
        """tasks"""
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        """create files"""
        config = self.returnn_config or ReturnnConfig({})
        assert "dataset" not in config.config and "dataset" not in config.post_config
        dataset_dict = self.dataset.copy()
        if self.post_dataset:
            # The modification to the config here is not part of the hash anymore,
            # so merge dataset and post_dataset now.
            dataset_dict.update(self.post_dataset)
        config.config["dataset"] = dataset_dict
        config.write(self.out_returnn_config_file.get_path())

    def run(self):
        """run"""
        import tempfile
        import shutil
        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset

        config = Config()
        config.load_file(self.out_returnn_config_file.get_path())
        set_global_config(config)

        dataset_dict = config.typed_value("dataset")
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)
        dataset.init_seq_order(epoch=1)

        with tempfile.NamedTemporaryFile("w") as tmp_file:
            if self.format == "py":
                tmp_file.write("{\n")

            seq_idx = 0
            while dataset.is_less_than_num_seqs(seq_idx):
                dataset.load_seqs(seq_idx, seq_idx + 1)
                seq_tag = dataset.get_tag(seq_idx)
                seq_len = dataset.get_seq_length(seq_idx)
                assert self.key in seq_len.keys()
                seq_len_ = seq_len[self.key]
                if self.format == "py":
                    tmp_file.write(f"{seq_tag!r}: {seq_len_},\n")
                elif self.format == "txt":
                    tmp_file.write(f"{seq_len_}\n")
                else:
                    raise ValueError(f"{self}: invalid format {self.format!r}")
                seq_idx += 1

            if self.format == "py":
                tmp_file.write("}\n")
            tmp_file.flush()

            shutil.copyfile(tmp_file.name, self.out_file.get_path())
