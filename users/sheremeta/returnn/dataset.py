from typing import Any, Dict

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


class ReturnnConfigDatasetDroppedLabelsJob(Job):
    """
    Counts per sequence of the dataset of a RETURNN config the labels its text codes leave out, with the config's
    prolog run first. The text codes hold the labels in order, one per frame, and the blank on every other frame.
    """

    def __init__(
        self,
        *,
        returnn_config: ReturnnConfig,
        codes_key: str,
        labels_key: str,
        blank: int,
        returnn_root: tk.Path,
        dataset_key: str = "forward_data",
        cpu_rqmt: int = 2,
        mem_rqmt: float = 8,
        time_rqmt: float = 2,
    ):
        """
        :param returnn_config: the config, its prolog defines what its dataset needs
        :param codes_key: the data key of the text codes, one code per frame
        :param labels_key: the data key of the labels
        :param blank: the code of a frame without a label
        :param returnn_root: the RETURNN checkout that loads the config and the dataset
        :param dataset_key: the config key of the dataset
        :param cpu_rqmt: CPUs of the job, the workers of the dataset among them
        :param mem_rqmt: memory of the job in GB
        :param time_rqmt: hours of the job
        """
        super().__init__()
        self.returnn_config = returnn_config
        self.codes_key = codes_key
        self.labels_key = labels_key
        self.blank = int(blank)
        self.returnn_root = returnn_root
        self.dataset_key = dataset_key
        self.rqmt = {"cpu": cpu_rqmt, "mem": mem_rqmt, "time": time_rqmt}
        self.out_summary = self.output_path("summary.json")
        self.out_report = self.output_path("report.txt")
        self.out_dropped = self.output_path("dropped.jsonl")

    @classmethod
    def hash(cls, parsed_args):
        # the dataset is defined in the prolog, all of which is hashed, the resources are not
        hashed = {key: value for key, value in parsed_args.items() if not key.endswith("_rqmt")}
        return super().hash({**hashed, "python_prolog": parsed_args["returnn_config"].python_prolog})

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import json

        import numpy as np

        dataset = _load_config_dataset(self.returnn_config, self.dataset_key, self.returnn_root)
        dataset.init_seq_order(epoch=1)

        # the labels the codes of every sequence place, which must be its first labels in order, the rest dropped
        num_seqs = num_not_in_order = num_labels = num_dropped = 0
        dropped_histogram: Dict[int, int] = {}
        with open(self.out_dropped.get_path(), "wt") as out:
            while dataset.is_less_than_num_seqs(num_seqs):
                dataset.load_seqs(num_seqs, num_seqs + 1)
                tag = dataset.get_tag(num_seqs)
                labels = np.asarray(dataset.get_data(num_seqs, self.labels_key)).reshape(-1)
                codes = np.asarray(dataset.get_data(num_seqs, self.codes_key)).reshape(-1)
                placed = codes[codes != self.blank]
                num_seqs += 1
                if num_seqs % 10000 == 0:
                    print(f"{num_seqs} sequences, {num_dropped} of {num_labels} labels dropped", flush=True)

                if placed.size > labels.size or not np.array_equal(placed, labels[: placed.size]):
                    num_not_in_order += 1
                    out.write(json.dumps({"seq_tag": tag, "labels": int(labels.size), "not_in_order": True}) + "\n")
                    continue
                dropped = int(labels.size - placed.size)
                num_labels += int(labels.size)
                num_dropped += dropped
                if dropped:
                    dropped_histogram[dropped] = dropped_histogram.get(dropped, 0) + 1
                    row = {"seq_tag": tag, "labels": int(labels.size), "dropped": dropped, "frames": int(codes.size)}
                    out.write(json.dumps(row) + "\n")

        num_counted = num_seqs - num_not_in_order
        num_with_drops = sum(dropped_histogram.values())
        summary = {
            "n_seqs": num_seqs,
            "n_seqs_not_in_order": num_not_in_order,
            "n_labels": num_labels,
            "n_dropped": num_dropped,
            "share_dropped": num_dropped / num_labels if num_labels else None,
            "n_seqs_with_drops": num_with_drops,
            "share_seqs_with_drops": num_with_drops / num_counted if num_counted else None,
            "dropped_histogram": {str(key): dropped_histogram[key] for key in sorted(dropped_histogram)},
        }
        with open(self.out_summary.get_path(), "wt") as f:
            json.dump(summary, f, indent=1)
        with open(self.out_report.get_path(), "wt") as f:
            f.write(
                f"sequences={num_seqs}  with dropped labels={num_with_drops}  labels={num_labels}  "
                f"dropped={num_dropped}  codes not the labels in order={num_not_in_order}\n"
            )
