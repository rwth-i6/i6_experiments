__all__ = ["EstimateDiphonePriorsJob", "DumpXmlForDiphoneJob", "DiphoneTensorMap"]

import logging
import math
import pickle
import time
import typing

import numpy as np

from sisyphus import tk, Job, Task

import i6_core.returnn as returnn

from ..factored import LabelInfo
from .mono import MonophoneTensorMap
from .util import EstimatePriorsJob


class DiphoneTensorMap(MonophoneTensorMap):
    in_classes: str
    in_encoder_output: str

    out_encoder_output: str
    out_left_context: str


def default_di_tensor_map() -> DiphoneTensorMap:
    return {
        "in_classes": "extern_data/placeholders/classes/classes:0",
        "in_data": "extern_data/placeholders/data/data:0",
        "in_encoder_output": "length_masked/strided_slice:0",
        "in_seq_length": "extern_data/placeholders/centerState/centerState_dim0_size:0",
        "out_encoder_output": "encoder__output/output_batch_major:0",
        "out_left_context": "left__output/output_batch_major:0",
        "out_center_state": "center__output/output_batch_major:0",
    }


class EstimateDiphonePriorsJob(EstimatePriorsJob):
    """Estimates the priors of the diphone state and its context."""

    def __init__(
        self,
        graph: tk.Path,
        model: returnn.Checkpoint,
        data_paths: typing.List[tk.Path],
        dataset_indices: typing.List[int],
        label_info: LabelInfo,
        native_lstm_path=None,
        batch_size=15000,
        gpu=1,
        mem=16,
        time=3,
        tensor_map: typing.Optional[DiphoneTensorMap] = None,
    ):
        super().__init__(
            graph=graph,
            model=model,
            data_paths=data_paths,
            dataset_indices=dataset_indices,
            native_lstm_library=native_lstm_path,
            batch_size=batch_size,
            gpu=gpu,
            mem=mem,
            time=time,
        )

        self.diphone_means: typing.Dict[int, np.ndarray] = {
            ctx: np.zeros(label_info.get_n_state_classes())
            for ctx in range(label_info.n_contexts)
        }
        self.context_means = np.zeros(label_info.n_contexts)
        self.num_segments = [
            self.output_path("segmentLength.%d" % index, cached=False)
            for index in self.dataset_indices
        ]
        self.diphone_files = [
            self.output_path("diphoneMeans.%d" % index, cached=False)
            for index in self.dataset_indices
        ]
        self.context_files = [
            self.output_path("contextMeans.%d" % index, cached=False)
            for index in self.dataset_indices
        ]
        self.label_info = label_info
        self.tensor_map = (
            {**default_di_tensor_map(), **tensor_map}
            if tensor_map is not None
            else default_di_tensor_map()
        )

    def get_encoder_output(self, session, feature_vector: np.ndarray):
        b, t = feature_vector.shape

        logging.info(f"encoder-output from data, b={b}", flush=True)

        return session.run(
            [self.tensor_map["out_encoder_output"]],
            feed_dict={
                self.tensor_map["in_data"]: feature_vector.reshape(1, b, t),
                self.tensor_map["in_seq_length"]: [b],
            },
        )

    def get_posteriors(
        self,
        session,
        feature_vector: np.ndarray,
        class_label_vector: typing.List[int],
        target: str,
    ):
        assert target in ["diphone", "context"]

        logging.info(
            f"{target} posteriors from encoder-output, b={feature_vector.shape[1]}, cls={len(class_label_vector)}"
        )

        if target == "diphone":
            tensor = self.tensor_map["out_center_state"]
        elif target == "context":
            tensor = self.tensor_map["out_left_context"]
        else:
            raise AttributeError(f"{target} is not a valid computation target")

        return session.run(
            [tensor],
            feed_dict={
                self.tensor_map["in_encoder_output"]: feature_vector.repeat(
                    len(class_label_vector), axis=0
                ),  # this is the encoder output
                self.tensor_map["in_classes"]: [
                    [label] * feature_vector.shape[1] for label in class_label_vector
                ],
            },
        )

    def get_dense_label(
        self,
        past_label: int,
        center_phoneme: int = 0,
        state_id: int = 0,
        future_label: int = 0,
    ):
        # don't need to support we/wb classes bc that's what we estimate

        result = center_phoneme

        result *= self.label_info.n_states_per_phone
        result += state_id

        result *= self.label_info.n_contexts
        result += past_label

        result *= self.label_info.n_contexts
        result += future_label

        return result

    def calculate_mean_posteriors(self, session, task_id: int):
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

            encoder_output, *_ = self.get_encoder_output(session, batch)

            context_posteriors, *_rest = self.get_posteriors(
                session,
                encoder_output,
                [self.get_dense_label(0)],
                target="context",
            )
            ctx = (sample_count * self.context_means) + (
                batch_size * np.mean(context_posteriors, axis=1)[0]
            )
            self.context_means = np.divide(ctx, denom)

            for past_context_id in range(self.label_info.n_contexts):
                diphone_posteriors, *_rest = self.get_posteriors(
                    session,
                    encoder_output,
                    [self.get_dense_label(past_context_id)],
                    target="diphone",
                )
                mean_diphone_posteriors = np.mean(diphone_posteriors, axis=1)

                prev = sample_count * self.diphone_means[past_context_id]
                cur = batch_size * mean_diphone_posteriors[0]
                di = prev + cur

                self.diphone_means[past_context_id] = np.divide(di, denom)

            sample_count += batch_size

        with open(tk.uncached_path(self.num_segments[task_id - 1]), "wb") as fp:
            pickle.dump(sample_count, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_means(self, task_id: int):
        with open(tk.uncached_path(self.diphone_files[task_id - 1]), "wb") as fp:
            pickle.dump(self.diphone_means, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tk.uncached_path(self.context_files[task_id - 1]), "wb") as fp:
            pickle.dump(self.context_means, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def hash(cls, args: typing.Dict[str, typing.Any]):
        to_hash = [
            "graph",
            "model",
            "data_paths",
            "dataset_indices",
            "label_info",
            "native_lstm_path",
            "tensor_map",
        ]
        return super().hash({param: args.get(param, None) for param in to_hash})


# you can use DumpXmlForDiphone and have an attribute called isSprint, with which you call your additional function.
# Generally think to merge all functions
class DumpXmlForDiphoneJob(Job):
    def __init__(
        self,
        diphone_files: typing.List[typing.Union[tk.Path, str]],
        context_files: typing.List[typing.Union[tk.Path, str]],
        num_segment_files: typing.List[typing.Union[tk.Path, str]],
        label_info: LabelInfo,
        adjust_silence: bool = True,
        adjust_non_word: bool = False,
        sil_boundary_indices: typing.List[int] = None,
        non_word_indices: typing.List[int] = None,
    ):
        self.diphone_files = diphone_files
        self.context_files = context_files
        self.num_segment_files = num_segment_files
        self.num_segments = []
        self.diphone_means = dict(
            zip(
                range(label_info.n_contexts), [[] for _ in range(label_info.n_contexts)]
            )
        )
        self.context_means = []
        self.adjust_silence = adjust_silence
        self.adjust_non_word = adjust_non_word
        self.sil_boundary_indices = (
            [0, 3] if sil_boundary_indices is None else sil_boundary_indices
        )
        self.non_word_indices = (
            [1, 2, 4] if non_word_indices is None else non_word_indices
        )

        self.out_center_state_xml = self.output_path("center-state.xml", cached=False)
        self.out_left_context_xml = self.output_path("left-context.xml", cached=False)

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
        for filename in self.diphone_files:
            with open(tk.uncached_path(filename), "rb") as f:
                diphoneDict = pickle.load(f)
                for i in range(self.nContexts):
                    self.diphone_means[i].append(
                        np.dot(
                            coeffs[self.diphone_files.index(filename)], diphoneDict[i]
                        )
                    )
        for filename in self.context_files:
            with open(tk.uncached_path(filename), "rb") as f:
                means = pickle.load(f)
                self.context_means.append(
                    np.dot(coeffs[self.context_files.index(filename)], means)
                )

        for i in range(self.nContexts):
            self.diphone_means[i] = np.sum(self.diphone_means[i], axis=0)

        self.diphone_means = np.array(
            [self.diphone_means[i] for i in range(self.nContexts)]
        )
        self.diphone_means /= np.sum(self.diphone_means)

        self.context_means = np.sum(self.context_means, axis=0)

    def set_sil_and_non_word_values(self):
        # context vectors
        sil = sum([self.context_means[i] for i in self.sil_boundary_indices])
        noise = sum([self.context_means[i] for i in self.non_word_indices])

        # center given context vectors
        meansListSil = [self.diphone_means[i] for i in self.sil_boundary_indices]
        meansListNonword = [self.diphone_means[i] for i in self.non_word_indices]
        dpSil = [sum(x) for x in zip(*meansListSil)]
        dpNoise = [sum(x) for x in zip(*meansListNonword)]

        for i in self.sil_boundary_indices:
            self.context_means[i] = sil
            self.diphone_means[i] = dpSil
        for i in self.non_word_indices:
            self.context_means[i] = noise
            self.diphone_means[i] = dpNoise

    def set_sil_values(self):
        sil = sum([self.context_means[i] for i in self.sil_boundary_indices])

        # center given context vectors
        meansListSil = [self.diphone_means[i] for i in self.sil_boundary_indices]
        dpSil = [np.sum(x) for x in zip(*meansListSil)]

        for i in self.sil_boundary_indices:
            self.context_means[i] = sil
            self.diphone_means[i] = dpSil

    def dump_xml(self):
        perturbation = 1e-8
        with open(tk.uncached_path(self.out_center_state_xml), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<matrix-f32 nRows="%d" nColumns="%d">\n'
                % (self.nContexts, self.nStateClasses)
            )
            for i in range(self.nContexts):
                self.diphone_means[i][self.diphone_means[i] == 0] = perturbation
                f.write(
                    " ".join("%.20e" % math.log(s) for s in self.diphone_means[i])
                    + "\n"
                )
            f.write("</matrix-f32>")
        with open(tk.uncached_path(self.out_left_context_xml), "wt") as f:
            self.context_means[self.context_means == 0] = perturbation
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n'
                % (self.nContexts)
            )
            f.write(
                " ".join("%.20e" % math.log(s) for s in np.nditer(self.context_means))
                + "\n"
            )
            f.write("</vector-f32>")

    def run(self):
        self.read_num_segments()
        self.calculate_weighted_averages()

        if self.adjust_silence:
            if self.adjust_non_word:
                self.set_sil_and_non_word_values()
            else:
                self.set_sil_values()

        self.dump_xml()
