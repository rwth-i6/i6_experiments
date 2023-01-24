__all__ = [
    "EstimateMonophonePriorsJob",
    "EstimateCartPriorsJob",
    "DumpXmlForMonophoneJob",
    "MonophoneTensorMap",
]

import logging
import math
import pickle
import time
import typing

import numpy as np

from sisyphus import tk, Job, Task

import i6_core.returnn as returnn

from .util import EstimatePriorsJob


class MonophoneTensorMap(typing.TypedDict, total=False):
    in_data: str
    in_seq_length: str

    out_center_state: str


def default_mono_tensor_map() -> MonophoneTensorMap:
    return {
        "in_data": "extern_data/placeholders/data/data:0",
        "in_seq_length": "extern_data/placeholders/centerState/centerState_dim0_size:0",
        "out_center_state": "center__output/output_batch_major:0",
    }


class EstimateMonophonePriorsJob(EstimatePriorsJob):
    """
    Estimates the priors of monophone states.
    """

    def __init__(
        self,
        graph: tk.Path,
        model: returnn.Checkpoint,
        data_paths: typing.List[tk.Path],
        dataset_indices: typing.List[int],
        num_states: typing.Union[int, tk.Variable],
        native_lstm_path: typing.Optional[typing.Union[str, tk.Path]] = None,
        batch_size=10000,
        gpu=1,
        mem=16,
        time=2,
        tensor_map: typing.Optional[MonophoneTensorMap] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            data_paths=data_paths,
            dataset_indices=dataset_indices,
            graph=graph,
            model=model,
            native_lstm_library=native_lstm_path,
            gpu=gpu,
            mem=mem,
            time=time,
        )

        self.center_phoneme_means: np.ndarray = np.zeros(0)
        self.num_segments = [
            self.output_path(f"segmentLength.{index}", cached=False)
            for index in self.dataset_indices
        ]
        self.num_states = num_states
        self.prior_files = [
            self.output_path(f"centerPhonemeMeans.{index:d}", cached=False)
            for index in self.dataset_indices
        ]
        self.tensor_map = (
            {**default_mono_tensor_map(), **tensor_map}
            if tensor_map is not None
            else default_mono_tensor_map()
        )

    def get_posteriors(self, session, feature_vector: np.ndarray):
        b, t, *_ = feature_vector.shape
        return session.run(
            [self.tensor_map["out_center_state"]],
            feed_dict={
                self.tensor_map["in_data"]: feature_vector.reshape(1, b, t),
                self.tensor_map["in_seq_length"]: [b],
            },
        )

    def calculate_mean_posteriors(self, session, task_id: int):
        self.center_phoneme_means = np.zeros(
            self.num_states
            if isinstance(self.num_states, int)
            else self.num_states.get()
        )

        last_print = time.monotonic()
        sample_count = 0

        for i, batch in enumerate(
            self.get_segment_features_from_hdf(self.dataset_indices[task_id - 1])
        ):
            now = time.monotonic()
            if now - last_print > 60:
                logging.info(f"{max(i - 1, 0)} batches done")
                last_print = now

            batch_size = len(batch)
            denom = sample_count + batch_size

            p = self.get_posteriors(session, batch)

            for i in range(len(batch)):
                nominator = (sample_count * self.center_phoneme_means) + (
                    batch_size * np.mean(p[0][0], axis=0)
                )
                self.center_phoneme_means = np.divide(nominator, denom)
            sample_count += batch_size

        with open(tk.uncached_path(self.num_segments[task_id - 1]), "wb") as fp:
            pickle.dump(sample_count, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_means(self, task_id: int):
        with open(tk.uncached_path(self.prior_files[task_id - 1]), "wb") as fp:
            pickle.dump(self.center_phoneme_means, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def hash(cls, args: typing.Dict[str, typing.Any]):
        to_hash = [
            "graph",
            "model",
            "data_paths",
            "dataset_indices",
            "native_lstm_path",
            "num_states",
            "tensor_map",
        ]
        return super().hash({param: args.get(param, None) for param in to_hash})


class EstimateCartPriorsJob(EstimateMonophonePriorsJob):
    def __init__(
        self,
        graph: tk.Path,
        model: returnn.Checkpoint,
        data_paths: typing.List[tk.Path],
        dataset_indices: typing.List[int],
        num_states: int,
        native_lstm_path: typing.Optional[typing.Union[str, tk.Path]] = None,
        batch_size: typing.Optional[int] = None,
        gpu=1,
        mem=16,
        time=20,
    ):
        bs = (
            batch_size
            if batch_size is not None
            else 10000
            if num_states < 550
            else 5000
        )

        super().__init__(
            batch_size=bs,
            data_paths=data_paths,
            dataset_indices=dataset_indices,
            graph=graph,
            model=model,
            num_states=num_states,
            native_lstm_path=native_lstm_path,
            gpu=gpu,
            mem=mem,
            time=time,
            tensor_map={},
        )

    def get_posteriors(self, session, feature_vector):
        b, t = feature_vector.shape
        return session.run(
            ["output/output_batch_major:0"],
            feed_dict={
                "extern_data/placeholders/data/data:0": feature_vector.reshape(1, b, t),
                "extern_data/placeholders/data/data_dim0_size:0": [b],
            },
        )


class DumpXmlForMonophoneJob(Job):
    def __init__(
        self,
        prior_files: typing.Union[typing.List[str], typing.List[tk.Path]],
        num_segment_files: typing.Union[typing.List[str], typing.List[tk.Path]],
        num_states: int,
        log=True,
    ):
        self.prior_files = prior_files
        self.num_segment_files = num_segment_files
        self.num_segments = []
        self.center_phoneme_means = []
        self.center_phoneme_xml = self.output_path(
            "centerPhonemeScores.xml", cached=False
        )
        self.num_states = num_states
        self.log = log
        self.rqmt = {"cpu": 2, "mem": 4, "time": 0.1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def read_num_segments(self):
        for filename in self.num_segment_files:
            with open(tk.uncached_path(filename), "rb") as f:
                self.num_segments.append(pickle.load(f))

    def calculate_weighted_averages(self):
        coeffs = [
            self.num_segments[i] / np.sum(self.num_segments)
            for i in range(len(self.num_segment_files))
        ]
        for filename in self.prior_files:
            with open(tk.uncached_path(filename), "rb") as f:
                means = pickle.load(f)
                self.center_phoneme_means.append(
                    np.dot(coeffs[self.prior_files.index(filename)], means)
                )
        self.center_phoneme_means = np.sum(self.center_phoneme_means, axis=0)

    def dump_xml(self):
        priors = (
            f"{math.log(s) if self.log else s:.20e}"
            for s in np.nditer(self.center_phoneme_means)
        )
        with open(tk.uncached_path(self.center_phoneme_xml), "wt") as f:
            f.write(
                f'<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="{self.num_states}">\n'
            )
            f.write(f"{' '.join(priors)}\n")
            f.write("</vector-f32>")

    def run(self):
        self.read_num_segments()
        self.calculate_weighted_averages()
        self.dump_xml()
