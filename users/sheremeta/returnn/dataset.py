from typing import Any

from sisyphus import Job, Task, tk

from i6_core.returnn.config import ReturnnConfig


def _load_config_dataset(returnn_config: ReturnnConfig, dataset_key: str, returnn_root: tk.Path) -> Any:
    """
    Writes the config, loads it as RETURNN does, its prolog run first, and builds its dataset.

    :param returnn_config: the config, its prolog defines what its dataset needs
    :param dataset_key: the config key of the dataset
    :param returnn_root: the RETURNN checkout that loads the config and the dataset
    :return: the dataset
    """
    import sys

    sys.path.insert(0, returnn_root.get_path())
    from returnn.config import Config, set_global_config
    from returnn.datasets import init_dataset
    from returnn.log import log

    returnn_config.write("returnn.config")
    config = Config()
    config.load_file("returnn.config")
    set_global_config(config)
    config.typed_dict.setdefault("log_verbosity", 4)
    log.init_by_config(config)
    return init_dataset(config.typed_value(dataset_key))


class ReturnnConfigDatasetToHdfJob(Job):
    """Writes one sparse data key of the dataset of a RETURNN config to an HDF, with the config's prolog run first."""

    def __init__(
        self, *, returnn_config: ReturnnConfig, data_key: str, returnn_root: tk.Path, dataset_key: str = "forward_data"
    ):
        """
        :param returnn_config: the config, its prolog defines what its dataset needs
        :param data_key: the data key to write, one label per frame
        :param returnn_root: the RETURNN checkout that loads the config and the dataset
        :param dataset_key: the config key of the dataset
        """
        super().__init__()
        self.returnn_config = returnn_config
        self.data_key = data_key
        self.returnn_root = returnn_root
        self.dataset_key = dataset_key
        self.out_hdf = self.output_path("data.hdf")

    @classmethod
    def hash(cls, parsed_args):
        # the dataset is defined in the prolog, all of which is hashed, also the part added after the config was built
        return super().hash({**parsed_args, "python_prolog": parsed_args["returnn_config"].python_prolog})

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 2})

    def run(self):
        import numpy as np

        # the dataset as the forward reads it
        dataset = _load_config_dataset(self.returnn_config, self.dataset_key, self.returnn_root)
        from returnn.datasets.hdf import SimpleHDFWriter

        dataset.init_seq_order(epoch=1)
        writer = SimpleHDFWriter(self.out_hdf.get_path(), dim=dataset.get_data_dim(self.data_key), ndim=1)
        seq_idx = 0
        while dataset.is_less_than_num_seqs(seq_idx):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            data = np.asarray(dataset.get_data(seq_idx, self.data_key), dtype=np.int32).reshape(-1)
            writer.insert_batch(data[None, :], seq_len={0: [data.shape[0]]}, seq_tag=[dataset.get_tag(seq_idx)])
            seq_idx += 1
        writer.close()
