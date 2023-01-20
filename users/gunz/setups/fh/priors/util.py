__all__ = ["chunks", "EstimatePriorsJob"]

import h5py
import itertools
import numpy as np
import typing

from sisyphus import tk, Job, Task

import i6_core.returnn as returnn


def chunks(lst: typing.List, n: int):
    """Yield successive n-sized chunks from lst."""

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class EstimatePriorsJob(Job):
    def __init__(
        self,
        *,
        graph: tk.Path,
        model: returnn.Checkpoint,
        data_paths: typing.List[tk.Path],
        dataset_indices: typing.List[int],
        native_lstm_library: typing.Optional[typing.Union[str, tk.Path]],
        batch_size: int,
        gpu: typing.Union[int, bool],
        mem: int,
        time: int,
    ):
        super().__init__()

        assert batch_size > 0

        self.batch_size = batch_size
        self.data_paths = data_paths
        self.dataset_indices = dataset_indices
        self.graph = graph
        self.model = model
        self.native_lstm_library = native_lstm_library

        self.rqmt = {
            "cpu": 2 if gpu else 8,
            "gpu": int(gpu),
            "mem": mem,
            "time": float(time if gpu else (time * 8)),
        }

    def get_run_args(self):
        return range(1, (len(self.dataset_indices) + 1))

    def tasks(self):
        yield Task(
            "run",
            resume="run",
            rqmt=self.rqmt,
            args=self.get_run_args(),
        )

    def get_segment_features_from_hdf(
        self,
        data_index: int,
        split_index: typing.Optional[int] = None,
        num_splits: typing.Optional[int] = None,
    ):
        with h5py.File(tk.uncached_path(self.data_paths[data_index]), "r") as hf:
            entries = hf["streams"]["features"]["data"]

            rows = (row for data in entries.values() for row in data[:])

            if split_index is not None and num_splits is not None:
                assert split_index < num_splits

                shape = np.sum([data.shape for data in entries.values()], axis=0)
                total_len = shape[-2]

                shard_len = total_len // num_splits
                start = split_index * shard_len
                end = start + shard_len

                print(
                    f"Processing features {start} to {end} from {total_len} {shape} total"
                )

                rows = itertools.islice(rows, start, end)

            while True:
                next_batch = list(itertools.islice(rows, self.batch_size))
                if len(next_batch) == 0:
                    break
                yield np.vstack(next_batch)

    def calculate_mean_posteriors(self, session, task_id: int):
        assert False, "implement this method"

    def dump_means(self, task_id: int):
        assert False, "implement this method"

    def run(self, task_id: int):
        import tensorflow as tf

        if self.rqmt["gpu"] > 0:
            assert tf.test.is_gpu_available(), "TF failed GPU init"

        if self.native_lstm_library is not None:
            tf.load_op_library(self.native_lstm_library)

        mg = tf.compat.v1.MetaGraphDef()
        with open(tk.uncached_path(self.graph), "rb") as f:
            mg.ParseFromString(f.read())

        tf.import_graph_def(mg.graph_def, name="")

        opts = {}
        opts.setdefault("intra_op_parallelism_threads", self.rqmt["cpu"])
        opts.setdefault("inter_op_parallelism_threads", self.rqmt["cpu"])

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(**opts)) as s:
            s.run(
                mg.saver_def.restore_op_name,
                feed_dict={
                    mg.saver_def.filename_tensor_name: tk.uncached_path(
                        self.model.ckpt_path
                    )
                },
            )
            self.calculate_mean_posteriors(s, task_id)

        self.dump_means(task_id)
