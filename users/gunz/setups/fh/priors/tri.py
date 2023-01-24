__all__ = [
    "EstimateTriphoneForwardPriorsJob",
    "DumpXmlForTriphoneJob",
    "TriphoneTensorMap",
]

import itertools
import logging
import pickle
import time
import typing

import numpy as np

from sisyphus import tk, Job, Task

import i6_core.returnn as returnn

from ..factored import LabelInfo
from .di import DiphoneTensorMap
from .util import chunks, EstimatePriorsJob


class TriphoneTensorMap(DiphoneTensorMap):
    out_right_context: str


def default_tri_tensor_map() -> TriphoneTensorMap:
    return {
        "in_classes": "extern_data/placeholders/classes/classes:0",
        "in_data": "extern_data/placeholders/data/data:0",
        "in_encoder_output": "length_masked/strided_slice:0",
        "in_seq_length": "extern_data/placeholders/centerState/centerState_dim0_size:0",
        "out_encoder_output": "encoder__output/output_batch_major:0",
        "out_left_context": "left__output/output_batch_major:0",
        "out_center_state": "center__output/output_batch_major:0",
        "out_right_context": "right__output/output_batch_major:0",
    }


class EstimateTriphoneForwardPriorsJob(EstimatePriorsJob):
    """Estimates the priors of the triphone states and their context."""

    def __init__(
        self,
        graph: tk.Path,
        model: returnn.Checkpoint,
        data_paths: typing.List[tk.Path],
        dataset_indices: typing.List[int],
        label_info: LabelInfo,
        native_lstm_path=None,
        batch_size=15000,
        num_splits=3,
        gpu=1,
        mem=12,
        tensor_map: typing.Optional[TriphoneTensorMap] = None,
        mlp_fwd_batch_size: int = 3,
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
            time=72 // num_splits,
        )

        assert num_splits > 0

        self.monophone_means = np.zeros(label_info.n_contexts)
        self.diphone_means = np.zeros(
            (label_info.n_contexts, label_info.get_n_state_classes())
        )
        self.triphone_means = np.zeros(
            (
                label_info.n_contexts * label_info.get_n_state_classes(),
                label_info.n_contexts,
            )
        )

        self.num_segments = [
            self.output_path(f"segment-length.{index}.{sub}")
            for index in self.dataset_indices
            for sub in range(num_splits)
        ]
        self.right_context_files = [
            self.output_path(f"right-context-means.{index}.{sub}")
            for index in self.dataset_indices
            for sub in range(num_splits)
        ]
        self.center_state_files = [
            self.output_path(f"center-state-means.{index}.{sub}")
            for index in self.dataset_indices
            for sub in range(num_splits)
        ]
        self.left_context_files = [
            self.output_path(f"left-context-means.{index}.{sub}")
            for index in self.dataset_indices
            for sub in range(num_splits)
        ]
        self.num_splits = num_splits

        self.label_info = label_info
        self.mlp_fwd_batch_size = mlp_fwd_batch_size
        self.tensor_map = (
            {**default_tri_tensor_map(), **tensor_map}
            if tensor_map is not None
            else default_tri_tensor_map()
        )

    def get_run_args(self):
        return range(1, len(self.dataset_indices) * self.num_splits + 1)

    def get_encoder_output(self, session, feature_vector: np.ndarray):
        logging.info(f"Computing encoder-output from data", flush=True)

        b, t = feature_vector.shape
        return session.run(
            [self.tensor_map["out_encoder_output"]],
            feed_dict={
                self.tensor_map["in_data"]: feature_vector.reshape(1, b, t),
                self.tensor_map["in_seq_length"]: [b],
            },
        )

    def get_mean_posteriors(
        self,
        session,
        mean_tensor,
        feature_vector: np.ndarray,
        class_label_vector: typing.List[int],
    ) -> np.ndarray:
        return session.run(
            mean_tensor,
            feed_dict={
                self.tensor_map["in_encoder_output"]: np.repeat(
                    feature_vector, len(class_label_vector), axis=0
                ),
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
        we_class: typing.Optional[int] = None,
        bd_class: typing.Optional[int] = None,
    ):
        assert (
            we_class is None or bd_class is None
        ), "can't have both word-end and boundary classes"

        we_bd_factor = 2 if we_class is not None else 4 if bd_class is not None else 1
        we_bd_add = (
            we_class
            if we_class is not None
            else bd_class
            if bd_class is not None
            else 0
        )

        result = center_phoneme

        result *= self.label_info.n_states_per_phone
        result += state_id

        result *= we_bd_factor
        result += we_bd_add

        result *= self.label_info.n_contexts
        result += past_label

        result *= self.label_info.n_contexts
        result += future_label

        return result

    def calculate_mean_posteriors(self, session, task_id: int):
        import tensorflow as tf

        last_print = time.monotonic()
        sample_count = 0

        task_index = task_id - 1
        dataset_index = task_index // self.num_splits
        split_index = task_index % self.num_splits

        batches = self.get_segment_features_from_hdf(
            self.dataset_indices[dataset_index],
            split_index=split_index,
            num_splits=self.num_splits,
        )

        g = tf.compat.v1.get_default_graph()
        r_mean_tensor, c_mean_tensor, l_mean_tensor = [
            tf.math.reduce_mean(g.get_tensor_by_name(name), axis=1)
            for name in (
                self.tensor_map["out_right_context"],
                self.tensor_map["out_center_state"],
                self.tensor_map["out_left_context"],
            )
        ]

        for i, batch in enumerate(batches):
            now = time.monotonic()
            if now - last_print > 60:
                logging.info(f"{max(i - 1, 0)} batches done")
                last_print = now

            batch_size = len(batch)
            denom = sample_count + batch_size

            encoder_output, *_ = self.get_encoder_output(session, batch)

            logging.info(f"encoder output computed, computing posteriors...")

            mean_left_context_posteriors = self.get_mean_posteriors(
                session,
                l_mean_tensor,
                encoder_output,
                [self.get_dense_label(0)],
            )
            ctx = (sample_count * self.monophone_means) + (
                batch_size * mean_left_context_posteriors[0]
            )
            self.monophone_means = np.divide(ctx, denom)

            past_context_ids = list(range(self.label_info.n_contexts))
            past_context_chunks = chunks(
                past_context_ids, self.mlp_fwd_batch_size
            )  # chunked MLP forwarding
            for past_contexts in past_context_chunks:
                mean_center_state_posteriors = self.get_mean_posteriors(
                    session,
                    c_mean_tensor,
                    encoder_output,
                    [self.get_dense_label(ctx) for ctx in past_contexts],
                )

                for j, past_context_id in enumerate(past_contexts):
                    prev = sample_count * self.diphone_means[past_context_id]
                    cur = batch_size * mean_center_state_posteriors[j]
                    di = prev + cur

                    self.diphone_means[past_context_id] = np.divide(di, denom)

            center_left_labels = [
                self.get_dense_label(
                    past_label_id,
                    center_phoneme_id,
                    state_id,
                    we_class=we_cls,
                    bd_class=bd_cls,
                )
                for (
                    past_label_id,
                    center_phoneme_id,
                    state_id,
                    we_cls,
                    bd_cls,
                ) in itertools.product(
                    range(self.label_info.n_contexts),
                    range(self.label_info.n_contexts),
                    range(self.label_info.n_states_per_phone),
                    range(2) if self.label_info.use_word_end_classes else [None],
                    range(4) if self.label_info.use_boundary_classes else [None],
                )
            ]
            chunked_labels = chunks(
                center_left_labels, self.mlp_fwd_batch_size
            )  # chunked MLP forwarding
            for center_left_labels in chunked_labels:
                mean_right_context_posteriors = self.get_mean_posteriors(
                    session, r_mean_tensor, encoder_output, center_left_labels
                )

                for j, label in enumerate(center_left_labels):
                    center_left_index = label // self.label_info.n_contexts
                    prev = sample_count * self.triphone_means[center_left_index]
                    cur = batch_size * mean_right_context_posteriors[j]
                    di = prev + cur

                    self.triphone_means[center_left_index] = np.divide(di, denom)

            sample_count += batch_size

        with open(tk.uncached_path(self.num_segments[task_index]), "wb") as fp:
            pickle.dump(sample_count, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_means(self, task_id: int):
        task_index = task_id - 1
        with open(tk.uncached_path(self.right_context_files[task_index]), "wb") as fp:
            pickle.dump(self.triphone_means, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tk.uncached_path(self.center_state_files[task_index]), "wb") as fp:
            pickle.dump(self.diphone_means, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tk.uncached_path(self.left_context_files[task_index]), "wb") as fp:
            pickle.dump(self.monophone_means, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
            "num_splits",
            "mlp_fwd_batch_size",
        ]
        return super().hash({param: args.get(param, None) for param in to_hash})


def unpickle(path: typing.Union[tk.Path, str]) -> typing.Any:
    with open(tk.uncached_path(path), mode="rb") as f:
        return pickle.load(f)


class DumpXmlForTriphoneJob(Job):
    def __init__(
        self,
        right_context_files: typing.List[typing.Union[str, tk.Path]],
        diphone_files: typing.List[typing.Union[str, tk.Path]],
        left_context_files: typing.List[typing.Union[str, tk.Path]],
        num_segment_files: typing.List[typing.Union[str, tk.Path]],
        label_info: LabelInfo,
        adjust_silence=True,
        adjust_non_word=False,
        sil_boundary_indices: typing.Optional[typing.List[int]] = None,
        non_word_indices: typing.Optional[typing.List[int]] = None,
    ):
        self.left_context_files = left_context_files
        self.center_state_files = diphone_files
        self.right_context_files = right_context_files

        self.label_info = label_info

        self.num_segment_files = num_segment_files
        self.num_segments = []

        self.adjust_silence = adjust_silence
        self.adjust_non_word = adjust_non_word
        self.sil_boundary_indices = (
            [0, 3] if sil_boundary_indices is None else sil_boundary_indices
        )
        self.non_word_indices = (
            [1, 2, 4] if non_word_indices is None else non_word_indices
        )

        self.out_center_state_xml = self.output_path("center_states.xml")
        self.out_right_context_xml = self.output_path("right_context.xml")
        self.out_left_context_xml = self.output_path("left_context.xml")

        self.left_context_means = np.zeros(label_info.n_contexts)
        self.center_state_means = np.zeros(
            (label_info.n_contexts, label_info.get_n_state_classes())
        )
        self.right_context_means = np.zeros(
            (
                label_info.n_contexts * label_info.get_n_state_classes(),
                label_info.n_contexts,
            )
        )

        self.rqmt = {"cpu": 2, "mem": 4, "time": 0.1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def calc_weighed_averages(self):
        weights = np.array(self.num_segments) / np.sum(self.num_segments)

        # (a.T * b).T is row-wise multiplication

        left_context_means = np.array([unpickle(f) for f in self.left_context_files])
        self.left_context_means = (left_context_means.T * weights).T.sum(axis=0)
        self.left_context_means /= np.sum(self.left_context_means)

        center_state_means = np.array([unpickle(f) for f in self.center_state_files])
        self.center_state_means = (center_state_means.T * weights).T.sum(axis=0)
        self.center_state_means /= np.sum(self.center_state_means)

        right_context_means = np.array([unpickle(f) for f in self.right_context_files])
        self.right_context_means = (right_context_means.T * weights).T.sum(axis=0)
        self.right_context_means /= np.sum(self.right_context_means)

    def combine_non_word(self):
        self.combine_silence_and_boundary()

        # context vectors
        noise = sum([self.left_context_means[i] for i in self.non_word_indices])

        # center given context vectors
        meansListNonword = [self.center_state_means[i] for i in self.non_word_indices]
        dpNoise = [sum(x) for x in zip(*meansListNonword)]

        for i in self.non_word_indices:
            self.left_context_means[i] = noise
            self.center_state_means[i] = dpNoise

    def combine_silence_and_boundary(self):
        sil_left = sum(self.left_context_means[i] for i in self.sil_boundary_indices)
        sil_center = np.sum(
            [self.center_state_means[i] for i in self.sil_boundary_indices], axis=0
        )

        for i in self.sil_boundary_indices:
            self.left_context_means[i] = sil_left
            self.center_state_means[i] = sil_center

    def dump_xml(self):
        perturbation = 1e-8

        self.left_context_means[self.left_context_means == 0] = perturbation
        self.center_state_means[self.center_state_means == 0] = perturbation
        self.right_context_means[self.right_context_means == 0] = perturbation

        with open(tk.uncached_path(self.out_left_context_xml), "wt") as f:
            log_means = np.log(self.left_context_means)
            f.write(
                f'<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="{log_means.size}">\n'
            )
            f.write(" ".join(f"{s:.20e}" for s in np.nditer(log_means)))
            f.write("\n</vector-f32>")

        with open(tk.uncached_path(self.out_center_state_xml), "wt") as f:
            r, c = self.center_state_means.shape
            f.write(
                f'<?xml version="1.0" encoding="UTF-8"?>\n<matrix-f32 nRows="{r}" nColumns="{c}">\n'
            )
            for log_means in np.log(self.center_state_means)[:]:
                f.write(" ".join(f"{s:.20e}" for s in log_means) + "\n")
            f.write("</matrix-f32>")

        with open(tk.uncached_path(self.out_right_context_xml), "wt") as f:
            r, c = self.right_context_means.shape
            f.write(
                f'<?xml version="1.0" encoding="UTF-8"?>\n<matrix-f32 nRows="{r}" nColumns="{c}">\n'
            )
            for log_means in np.log(self.right_context_means)[:]:
                f.write(" ".join(f"{s:.20e}" for s in log_means) + "\n")
            f.write("</matrix-f32>")

    def run(self):
        self.num_segments = [unpickle(f) for f in self.num_segment_files]

        self.calc_weighed_averages()
        if self.adjust_silence:
            self.combine_silence_and_boundary()
        elif self.adjust_non_word:
            raise NotImplementedError("non-word is unimpl'd")
        self.dump_xml()
