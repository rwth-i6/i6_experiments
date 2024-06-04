__all__ = ["EstimateFactoredTriphonePriorsJob", "DumpXmlForTriphoneForwardJob"]


import h5py
import logging
import numpy as np
import math
from typing import List, Optional, Union

from IPython import embed

try:
    import cPickle as pickle
except ImportError:
    import pickle

from sisyphus import *
from sisyphus.delayed_ops import DelayedFormat

Path = setup_path(__package__)

from i6_core.lib.rasr_cache import FileArchive

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo
from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import DecodingTensorMap
from i6_experiments.users.raissi.setups.common.helpers.priors.util import (
    initialize_dicts,
    initialize_dicts_with_zeros,
    get_batch_from_segments,
)

from i6_experiments.users.raissi.setups.common.util.cache_manager import cache_file

###################################
# Triphone
###################################
class EstimateFactoredTriphonePriorsJob(Job):
    def __init__(
        self,
        graph_path: Path,
        model_path: DelayedFormat,
        tensor_map: Optional[Union[dict, DecodingTensorMap]],
        data_paths: [Path],
        data_indices: [int],
        start_ind_segment: int,
        end_ind_segment: int,
        label_info: LabelInfo,
        tf_library_path: str = None,
        n_batch=10000,
        cpu=2,
        gpu=1,
        mem=32,
        time=1,
    ):
        self.graph_path = graph_path
        self.model_path = model_path
        self.data_paths = data_paths
        self.data_indices = data_indices
        self.segment_slice = (start_ind_segment, end_ind_segment)
        self.tf_library_path = tf_library_path
        self.triphone_means, self.diphone_means = initialize_dicts_with_zeros(
            label_info.n_contexts, label_info.get_n_state_classes()
        )
        self.context_means = np.zeros(label_info.n_contexts)
        self.num_segments = [
            self.output_path("segment_length.%d.%d-%d" % (index, start_ind_segment, end_ind_segment), cached=False)
            for index in self.data_indices
        ]
        self.triphone_files = [
            self.output_path("triphone_means.%d.%d-%d" % (index, start_ind_segment, end_ind_segment), cached=False)
            for index in self.data_indices
        ]
        self.diphone_files = [
            self.output_path("diphone_means.%d.%d-%d" % (index, start_ind_segment, end_ind_segment), cached=False)
            for index in self.data_indices
        ]
        self.context_files = [
            self.output_path("context_means.%d.%d-%d" % (index, start_ind_segment, end_ind_segment), cached=False)
            for index in self.data_indices
        ]
        self.label_info = label_info
        self.n_batch = n_batch
        self.tensor_map = tensor_map
        self.rqmt = {"cpu": cpu, "gpu": gpu, "mem": mem, "time": float(time)}

    def tasks(self):
        self.rqmt["mem"] *= 2
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, (len(self.data_indices) + 1)))

    def get_dense_label(self, left_context, center_state, right_context=0):
        return (
            ((center_state * self.label_info.n_contexts) + left_context) * self.label_info.n_contexts
        ) + right_context

    def get_segment_features_from_hdf(self, dataIndex):
        logging.info(f"processing {self.data_paths[dataIndex]}")
        file_path = self.data_paths[dataIndex]
        hf = h5py.File(file_path)
        segment_names = list(hf["streams"]["features"]["data"])
        segments = []
        for name in segment_names:
            segments.append(hf["streams"]["features"]["data"][name])
        return np.vstack(segments[self.segment_slice[0] : self.segment_slice[1]])

    def get_encoder_output(self, session, feature_vector):
        return session.run(
            [f"{self.tensor_map.out_encoder_output}:0"],
            feed_dict={
                f"{self.tensor_map.in_data}:0": feature_vector.reshape(
                    1, feature_vector.shape[0], feature_vector.shape[1]
                ),
                f"{self.tensor_map.in_seq_length}:0": [feature_vector.shape[0]],
            },
        )

    def get_posteriors_given_encoder_output(self, session, feature_vector, class_label_vector):
        feature_in = (
            feature_vector.reshape(feature_vector.shape[1], 1, feature_vector.shape[2])
            if "fwd" in tensor_map.in_encoder_output
            else feature_vector
        )
        return session.run(
            [
                f"{self.tensor_map.out_left_context}:0",
                f"{self.tensor_map.out_center_state}:0",
                f"{self.tensor_map.out_right_context}:0",
            ],
            feed_dict={
                f"{self.tensor_map.in_encoder_output}:0": feature_in,
                f"{self.tensor_map.in_classes}:0": [[class_label_vector] * feature_vector.shape[1]],
            },
        )

    def calculate_mean_posteriors(self, session, task_id):
        logging.info(f"starting with {task_id}")
        sample_count = 0
        segments = self.get_segment_features_from_hdf(self.data_indices[task_id - 1])

        for batch in get_batch_from_segments(segments, self.n_batch):
            b_size = len(batch)
            denom = sample_count + b_size
            if len(batch) == 0:
                break
            encoder_output = self.get_encoder_output(session, batch)
            for left_context in range(self.label_info.n_contexts):
                for center_state in range(self.label_info.get_n_state_classes()):
                    denselabel = self.get_dense_label(left_context=left_context, center_state=center_state)
                    p = self.get_posteriors_given_encoder_output(session, encoder_output[0], denselabel)
                    # triphone is calculates for each center and left context
                    tri = (sample_count * self.triphone_means[left_context][center_state]) + (
                        b_size * np.mean(p[0][0], axis=0)
                    )
                    self.triphone_means[left_context][center_state] = np.divide(tri, denom)
                    # diphone is calculated for each context with centerstate 0
                    if not center_state:
                        di = (sample_count * self.diphone_means[left_context]) + (b_size * np.mean(p[1][0], axis=0))
                        self.diphone_means[left_context] = np.divide(di, denom)
                        # context is not label dependent
                        if not left_context:
                            ctx = (sample_count * self.context_means) + (b_size * np.mean(p[2][0], axis=0))
                            self.context_means = np.divide(ctx, denom)
            sample_count += b_size

        with open(self.num_segments[task_id - 1].get_path(), "wb") as fp:
            pickle.dump(sample_count, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_means(self, task_id):
        logging.info(f"dumping means")
        with open(self.triphone_files[task_id - 1].get_path(), "wb") as f1:
            pickle.dump(self.triphone_means, f1, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.diphone_files[task_id - 1].get_path(), "wb") as f2:
            pickle.dump(self.diphone_means, f2, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.context_means[task_id - 1].get_path(), "wb") as f3:
            pickle.dump(self.context_means, f3, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, task_id):
        import tensorflow as tf
        if self.tf_library_path is not None:
            tf.load_op_library(self.tf_library_path)
        mg = tf.compat.v1.MetaGraphDef()
        mg.ParseFromString(open(self.graph_path.get_path(), "rb").read())
        tf.import_graph_def(mg.graph_def, name="")
        # session
        s = tf.compat.v1.Session()
        returnValue = s.run(["save/restore_all"], feed_dict={"save/Const:0": self.model_path.get()})

        self.calculate_mean_posteriors(s, task_id)
        self.dump_means(task_id)


class CombineMeansForTriphoneForward(Job):
    def __init__(
        self,
        triphone_files: List[Path],
        diphone_files: List[Path],
        context_files: List[Path],
        num_segment_files: List[Path],
        label_info: LabelInfo,
    ):
        self.triphone_files = triphone_files
        self.diphone_files = diphone_files
        self.context_files = context_files
        self.num_segment_files = num_segment_files
        self.label_info = label_info
        self.num_segments = []
        self.triphone_means, self.diphoneMeans = initialize_dicts(
            n_contexts=label_info.n_contexts, n_state_classes=label_info.get_n_state_classes()
        )
        self.context_means = []
        self.num_segments_out = self.output_path("segment_length", cached=False)
        self.triphone_files_out = self.output_path("triphone_means", cached=False)
        self.diphone_files_out = self.output_path("diphoneMeans", cached=False)
        self.context_files_out = self.output_path("context_means", cached=False)
        self.rqmt = {"cpu": 1, "mem": 1, "time": 0.5}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def read_num_segments(self):
        for filename in self.num_segment_files:
            with open(tk.uncached_path(filename), "rb") as f:
                self.num_segments.append(pickle.load(f))

    def calculate_weighted_averages(self):
        coeffs = [self.num_segments[i] / np.sum(self.num_segments) for i in range(len(self.num_segment_files))]
        for filename in self.triphone_files:
            with open(tk.uncached_path(filename), "rb") as f:
                triphoneDict = pickle.load(f)
                for i in range(self.nContexts):
                    for j in range(self.nStates):
                        self.triphone_means[i][j].append(
                            np.dot(coeffs[self.triphone_files.index(filename)], triphoneDict[i][j])
                        )
        for filename in self.diphone_files:
            with open(tk.uncached_path(filename), "rb") as f:
                diphoneDict = pickle.load(f)
                for i in range(self.nContexts):
                    self.diphoneMeans[i].append(np.dot(coeffs[self.diphone_files.index(filename)], diphoneDict[i]))
        for filename in self.context_files:
            with open(tk.uncached_path(filename), "rb") as f:
                means = pickle.load(f)
                self.context_means.append(np.dot(coeffs[self.context_files.index(filename)], means))
        for i in range(self.nContexts):
            self.diphoneMeans[i] = np.sum(self.diphoneMeans[i], axis=0)
            for j in range(self.nStates):
                self.triphone_means[i][j] = np.sum(self.triphone_means[i][j], axis=0)
        self.context_means = np.sum(self.context_means, axis=0)

    def dump_means(self):
        with open(tk.uncached_path(self.triphone_files_out), "wb") as f1:
            pickle.dump(self.triphone_means, f1, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tk.uncached_path(self.diphone_files_out), "wb") as f2:
            pickle.dump(self.diphoneMeans, f2, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tk.uncached_path(self.context_files_out), "wb") as f3:
            pickle.dump(self.context_means, f3, protocol=pickle.HIGHEST_PROTOCOL)
        sumSegNums = np.sum(self.num_segments)
        with open(tk.uncached_path(self.num_segments_out), "wb") as f4:
            pickle.dump(sumSegNums, f4, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        self.read_num_segments()
        self.calculate_weighted_averages()
        self.dump_means()


class DumpXmlForTriphoneForwardJob(Job):
    def __init__(
        self,
        triphone_files: List,
        diphone_files: List,
        context_files: List,
        num_segment_files: List,
        label_info: LabelInfo,
    ):
        self.triphone_files = triphone_files
        self.diphone_files = diphone_files
        self.context_files = context_files
        self.num_segment_files = num_segment_files
        self.label_info = label_info
        self.num_segments = []
        self.triphone_means, self.diphone_means = initialize_dicts(
            n_contexts=label_info.n_contexts, n_state_classes=label_info.get_n_state_classes()
        )
        self.context_means = []
        self.triphone_xml = self.output_path("triphone_scores.xml", cached=False)
        self.diphone_xml = self.output_path("diphone_scores.xml", cached=False)
        self.context_xml = self.output_path("context_scores.xml", cached=False)
        self.rqmt = {"cpu": 1, "mem": 1, "time": 1}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def read_num_segments(self):
        for filename in self.num_segment_files:
            with open(filename.get_path(), "rb") as f:
                self.num_segments.append(pickle.load(f))

    def calculate_weighted_averages(self):
        coeffs = [self.num_segments[i] / np.sum(self.num_segments) for i in range(len(self.num_segment_files))]
        for filename in self.triphone_files:
            with open(filename.get_path(), "rb") as f:
                triphoneDict = pickle.load(f)
                for i in range(self.label_info.n_contexts):
                    for j in range(self.label_info.get_n_state_classes()):
                        self.triphone_means[i][j].append(
                            np.dot(coeffs[self.triphone_files.index(filename)], triphoneDict[i][j])
                        )
        for filename in self.diphone_files:
            with open(filename.get_path(), "rb") as f:
                diphone_dict = pickle.load(f)
                for i in range(self.label_info.n_contexts):
                    self.diphone_means[i].append(np.dot(coeffs[self.diphone_files.index(filename)], diphone_dict[i]))
        for filename in self.context_files:
            with open(filename.get_path(), "rb") as f:
                means = pickle.load(f)
                self.context_means.append(np.dot(coeffs[self.context_files.index(filename)], means))
        for i in range(self.label_info.n_contexts):
            self.diphone_means[i] = np.sum(self.diphone_means[i], axis=0)
            for j in range(self.label_info.get_n_state_classes()):
                self.triphone_means[i][j] = np.sum(self.triphone_means[i][j], axis=0)
        self.context_means = np.sum(self.context_means, axis=0)

    def dump_xml(self):
        for context_id in range(self.label_info.n_contexts):
            for center_stateId in range(self.label_info.get_n_state_classes()):
                for i, s in enumerate(self.triphone_means[context_id][center_stateId]):
                    if s == 0:
                        self.triphone_means[context_id][center_stateId][i] += 1e-5
        with open(self.triphone_xml.get_path(), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<matrix-f32 nRows="%d" nColumns="%d">\n'
                % (self.label_info.n_contexts * self.label_info.get_n_state_classes(), self.label_info.n_contexts)
            )
            for context_id in range(self.label_info.n_contexts):
                for center_stateId in range(self.label_info.get_n_state_classes()):
                    for i, s in enumerate(self.triphone_means[context_id][center_stateId]):
                        if s == 0:
                            self.triphone_means[context_id][center_stateId][i] += 1e-5
                    f.write(" ".join("%.20e" % math.log(s) for s in self.triphone_means[context_id][center_stateId]) + "\n")
            f.write("</matrix-f32>")
        with open(self.diphone_xml.get_path(), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<matrix-f32 nRows="%d" nColumns="%d">\n'
                % (self.label_info.n_contexts, self.label_info.get_n_state_classes())
            )
            for context_id in range(self.label_info.n_contexts):
                for i, c in enumerate(self.diphone_means[context_id]):
                    if c == 0:
                        self.diphone_means[context_id][i] += 1e-5
                f.write(" ".join("%.20e" % math.log(s) for s in self.diphone_means[context_id]) + "\n")
            f.write("</matrix-f32>")
        with open(self.context_xml.get_path(), "wt") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n' % (self.label_info.n_contexts))
            f.write(" ".join("%.20e" % math.log(s) for s in np.nditer(self.context_means)) + "\n")
            f.write("</vector-f32>")

    def run(self):
        self.read_num_segments()
        logging.info("number of segments read")
        self.calculate_weighted_averages()
        self.dump_xml()


# needs refactoring
class EstimateRasrDiphoneAndContextPriors(Job):
    def __init__(
        self,
        graph_path: Path,
        model_path: DelayedFormat,
        tensor_map: Optional[Union[dict, DecodingTensorMap]],
        data_paths: [Path],
        data_indices: [int],
        label_info: LabelInfo,
        tf_library_path: str = None,
        n_batch=12000,
        cpu=2,
        gpu=1,
        mem=4,
        time=1,
    ):
        self.graph_path = graph_path
        self.model_path = model_path
        self.tensor_map = tensor_map
        self.data_paths = data_paths
        self.data_indices = data_indices
        self.tf_library_path = tf_library_path
        self.diphoneMeans = dict(
            zip(range(label_info.n_contexts), [np.zeros(nStateClasses) for _ in range(label_info.n_contexts)])
        )
        self.context_means = np.zeros(label_info.n_contexts)
        self.num_segments = [self.output_path("segmentLength.%d" % index, cached=False) for index in self.data_indices]
        self.diphone_files = [self.output_path("diphoneMeans.%d" % index, cached=False) for index in self.data_indices]
        self.context_files = [self.output_path("context_means.%d" % index, cached=False) for index in self.data_indices]
        self.n_batch = n_batch

        if not gpu:
            time *= 4
        self.rqmt = {"cpu": cpu, "gpu": gpu, "mem": mem, "time": float(time)}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, (len(self.data_indices) + 1)))

    def get_segment_features_from_hdf(self, dataIndex):
        hf = h5py.File(tk.uncached_path(self.data_paths[dataIndex]))
        segmentNames = list(hf["streams"]["features"]["data"])
        segments = []
        for name in segmentNames:
            segments.append(hf["streams"]["features"]["data"][name])
        return np.vstack(segments)

    def get_encoder_output(self, session, feature_vector):
        return session.run(
            ["encoder-output/output_batch_major:0"],
            feed_dict={
                "extern_data/placeholders/data/data:0": feature_vector.reshape(
                    1, feature_vector.shape[0], feature_vector.shape[1]
                ),
                "extern_data/placeholders/data/data_dim0_size:0": [feature_vector.shape[0]],
            },
        )

    def getPosteriorsOfBothOutputsWithEncoded(self, session, feature_vector, class_label_vector):
        return session.run(
            [
                ("-").join([self.tensor_map["diphone"], "output/output_batch_major:0"]),
                ("-").join([self.tensor_map["context"], "output/output_batch_major:0"]),
            ],
            feed_dict={
                "concat_fwd_6_bwd_6/concat_sources/concat:0": feature_vector.reshape(
                    feature_vector.shape[1], 1, feature_vector.shape[2]
                ),
                "extern_data/placeholders/classes/classes:0": [[class_label_vector] * feature_vector.shape[1]],
            },
        )

    def get_dense_label(self, left_context, center_state, right_context=0):
        return (
            ((center_state * self.label_info.n_contexts) + left_context) * self.label_info.n_contexts
        ) + right_context

    def calculate_mean_posteriors(self, session, task_id):
        logging.info(f"starting with {task_id}")
        sample_count = 0
        segments = self.get_segment_features_from_hdf(self.data_indices[task_id - 1])

        for batch in get_batch_from_segments(segments, self.n_batch):
            b_size = len(batch)
            denom = sample_count + b_size
            if len(batch) == 0:
                break

            encoder_output = self.get_encoder_output(session, batch)
            for left_context in range(self.label_info.n_contexts):
                p = self.getPosteriorsOfBothOutputsWithEncoded(
                    session, encoder_output[0], self.get_dense_label(left_context)
                )

                di = (sample_count * self.diphoneMeans[left_context]) + (b_size * np.mean(p[0][0], axis=0))
                self.diphoneMeans[left_context] = np.divide(di, denom)
                # context is not label dependent
                if not left_context:
                    ctx = (sample_count * self.context_means) + (b_size * np.mean(p[1][0], axis=0))
                    self.context_means = np.divide(ctx, denom)
            sample_count += b_size

        with open(tk.uncached_path(self.num_segments[task_id - 1]), "wb") as fp:
            pickle.dump(sample_count, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_means(self, task_id):
        with open(tk.uncached_path(self.diphone_files[task_id - 1]), "wb") as fp:
            pickle.dump(self.diphoneMeans, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tk.uncached_path(self.context_files[task_id - 1]), "wb") as fp:
            pickle.dump(self.context_means, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, task_id):
        import tensorflow as tf
        if self.tf_library_path is not None:
            tf.load_op_library(self.tf_library_path)
        mg = tf.MetaGraphDef()
        mg.ParseFromString(open(self.graph_path.get_path(), "rb").read())
        tf.import_graph_def(mg.graph_def, name="")
        # session
        s = tf.Session()
        returnValue = s.run(["save/restore_all"], feed_dict={"save/Const:0": self.model_path.get()})

        self.calculate_mean_posteriors(s, task_id)
        self.dump_means(task_id)


# needs refactoring
# you can use dump_xmlForDiphone and have an attribute called isSprint, with which you call your additional function.
# Generally think to merge all functions
class dump_xmlRasrForDiphone(Job):
    def __init__(
        self,
        diphone_files,
        context_files,
        num_segment_files,
        nContexts,
        nStateClasses,
        adjustSilence=True,
        adjustNonWord=False,
        silBoundaryIndices=None,
        nonWordIndices=None,
    ):

        self.diphone_files = diphone_files
        self.context_files = context_files
        self.num_segment_files = num_segment_files
        self.num_segments = []
        self.diphoneMeans = dict(zip(range(nContexts), [[] for _ in range(nContexts)]))
        self.context_means = []
        self.diphoneXml = self.output_path("diphoneScores.xml", cached=False)
        self.contextXml = self.output_path("contextScores.xml", cached=False)
        self.nContexts = nContexts
        self.nStateClasses = nStateClasses
        self.adjustSilence = adjustSilence
        self.adjustNonWord = adjustNonWord
        self.silBoundaryIndices = [0, 3] if silBoundaryIndices is None else silBoundaryIndices
        self.nonWordIndices = [1, 2, 4] if nonWordIndices is None else nonWordIndices
        self.rqmt = {"cpu": 2, "mem": 4, "time": 0.1}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def read_num_segments(self):
        for filename in self.num_segment_files:
            with open(tk.uncached_path(filename), "rb") as f:
                self.num_segments.append(pickle.load(f))

    def calculate_weighted_averages(self):
        coeffs = [self.num_segments[i] / np.sum(self.num_segments) for i in range(len(self.num_segment_files))]
        for filename in self.diphone_files:
            with open(tk.uncached_path(filename), "rb") as f:
                diphone_dict = pickle.load(f)
                for i in range(self.label_info.n_contexts):
                    self.diphoneMeans[i].append(np.dot(coeffs[self.diphone_files.index(filename)], diphone_dict[i]))
        for filename in self.context_files:
            with open(tk.uncached_path(filename), "rb") as f:
                means = pickle.load(f)
                self.context_means.append(np.dot(coeffs[self.context_files.index(filename)], means))
        for i in range(self.label_info.n_contexts):
            self.diphoneMeans[i] = np.sum(self.diphoneMeans[i], axis=0)
        self.context_means = np.sum(self.context_means, axis=0)

    def setSilenceAndNonWordValues(self):
        # context vectors
        sil = sum([self.context_means[i] for i in self.silBoundaryIndices])
        noise = sum([self.context_means[i] for i in self.nonWordIndices])

        # center given context vectors
        meansListSil = [self.diphoneMeans[i] for i in self.silBoundaryIndices]
        meansListNonword = [self.diphoneMeans[i] for i in self.nonWordIndices]
        dpSil = [sum(x) for x in zip(*meansListSil)]
        dpNoise = [sum(x) for x in zip(*meansListNonword)]

        for i in self.silBoundaryIndices:
            self.context_means[i] = sil
            self.diphoneMeans[i] = dpSil
        for i in self.nonWordIndices:
            self.context_means[i] = noise
            self.diphoneMeans[i] = dpNoise

    def setSilenceValues(self):
        sil = sum([self.context_means[i] for i in self.silBoundaryIndices])

        # center given context vectors
        meansListSil = [self.diphoneMeans[i] for i in self.silBoundaryIndices]
        dpSil = [np.sum(x) for x in zip(*meansListSil)]

        for i in self.silBoundaryIndices:
            self.context_means[i] = sil
            self.diphoneMeans[i] = dpSil

    def dump_xml(self):
        perturbation = 1e-8
        with open(tk.uncached_path(self.diphoneXml), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<matrix-f32 nRows="%d" nColumns="%d">\n'
                % (self.label_info.n_contexts, self.nStateClasses)
            )
            for i in range(self.label_info.n_contexts):
                self.diphoneMeans[i][self.diphoneMeans[i] == 0] = perturbation
                f.write(" ".join("%.20e" % math.log(s) for s in self.diphoneMeans[i]) + "\n")
            f.write("</matrix-f32>")
        with open(tk.uncached_path(self.contextXml), "wt") as f:
            self.context_means[self.context_means == 0] = perturbation
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n' % (self.label_info.n_contexts))
            f.write(" ".join("%.20e" % math.log(s) for s in np.nditer(self.context_means)) + "\n")
            f.write("</vector-f32>")

    def dumpPickle(self):
        with open("/u/raissi/experiments/notebooks/diphones.pickle", "wb") as fp:
            pickle.dump(self.diphoneMeans, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open("/u/raissi/experiments/notebooks/context.pickle", "wb") as fp:
            pickle.dump(self.context_means, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        self.read_num_segments()
        self.calculate_weighted_averages()
        if self.adjustSilence:
            if self.adjustNonWord:
                self.setSilenceAndNonWordValues()
            else:
                self.setSilenceValues()

        self.dump_xml()
        self.dumpPickle()
