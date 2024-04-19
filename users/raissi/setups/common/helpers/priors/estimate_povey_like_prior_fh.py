__all__ = ["EstimateSprintTriphoneForwardPriorsJob", "DumpXmlForTriphoneForwardJob"]


import h5py
import numpy as np
import math
import tensorflow as tf

from IPython import embed

try:
    import cPickle as pickle
except ImportError:
    import pickle

from sisyphus import *

from i6_core.lib.rasr_cache import FileArchive

Path = setup_path(__package__)


def initialize_dicts_with_zeros(n_contexts, n_states, isForward=True):
    triDict = {}
    for i in range(n_contexts):
        triDict[i] = dict(zip(range(n_states), [np.zeros(n_contexts) for _ in range(n_states)]))
    if isForward:
        diDict = dict(zip(range(n_contexts), [np.zeros(n_states) for _ in range(n_contexts)]))
    else:
        diDict = dict(zip(range(n_states), [np.zeros(n_contexts) for _ in range(n_states)]))
    return triDict, diDict


def initialize_dicts(n_contexts, n_state_classes, isForward=True):
    triDict = {}
    for i in range(n_contexts):
        triDict[i] = dict(zip(range(n_state_classes), [[] for _ in range(n_state_classes)]))
    if isForward:
        diDict = dict(zip(range(n_contexts), [[] for _ in range(n_contexts)]))
    else:
        diDict = dict(zip(range(n_state_classes), [[] for _ in range(n_state_classes)]))
    return triDict, diDict


def get_batch_from_segments(segments, batchSize=10000):
    index = 0
    while True:
        try:
            yield segments[index * batchSize : (index + 1) * batchSize]
            index += 1
        except IndexError:
            index = 0


###################################
# Triphone
###################################
class EstimateSprintTriphoneForwardPriorsJob(Job):
    def __init__(
        self,
        graph_path,
        model_path,
        tensor_map,
        data_paths,
        data_indices,
        start_ind_segment,
        end_ind_segment,
        library_path,
        n_state_classes,
        n_contexts,
        n_batch=15000,
        cpu=2,
        gpu=1,
        mem=2,
        time=1,
    ):
        self.graph_path = graph_path
        self.model_path = model_path
        self.data_paths = data_paths
        self.data_indices = data_indices
        self.segment_slice = (start_ind_segment, end_ind_segment)
        self.tf_lib = library_path
        self.triphone_means, self.diphone_means = initialize_dicts_with_zeros(n_contexts, n_state_classes)
        self.context_means = np.zeros(n_contexts)
        self.numSegments = [
            self.output_path("segmentLength.%d.%d-%d" % (index, start_ind_segment, end_ind_segment), cached=False)
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
        self.context_means = [
            self.output_path("context_means.%d.%d-%d" % (index, start_ind_segment, end_ind_segment), cached=False)
            for index in self.data_indices
        ]
        self.n_contexts = n_contexts
        self.n_state_classes = n_state_classes
        self.n_batch = n_batch
        self.tensor_map = tensor_map
        self.rqmt = {"cpu": cpu, "gpu": gpu, "mem": mem, "time": float(time)}

    def tasks(self):
        self.rqmt["mem"] *= 2
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, (len(self.data_indices) + 1)))

    def get_dense_label(self, pastLabel, centerState, futureLabel=0):
        return (((centerState * self.n_contexts) + pastLabel) * self.n_contexts) + futureLabel

    def getSegmentFeaturesFromHdf(self, dataIndex):
        hf = h5py.File(self.data_paths[dataIndex].get_path())
        segmentNames = list(hf["streams"]["features"]["data"])
        segments = []
        for name in segmentNames:
            segments.append(hf["streams"]["features"]["data"][name])
        return np.vstack(segments[self.segment_slice[0] : self.segment_slice[1]])

    def getEncoderOutput(self, session, featureVector):
        return session.run(
            [self.tensor_map.out_encoder_output],
            feed_dict={
                self.tensor_map.in_data: featureVector.reshape(1, featureVector.shape[0], featureVector.shape[1]),
                self.tensor_map.in_seq_length: [featureVector.shape[0]],
            },
        )

    def getPosteriorsOfOutputsWithEncoderOutput(self, session, featureVector, classLabelVector):
        feature_in = (
            featureVector.reshape(featureVector.shape[1], 1, featureVector.shape[2])
            if "fwd" in self.tensor_map.in_encoder_output
            else featureVector
        )
        return session.run(
            [self.tensor_map.out_left_context, self.tensor_map.out_center_state, self.tensor_map.out_right_context],
            feed_dict={
                self.tensor_map.in_encoder_output: feature_in,
                self.tensor_map.in_seq_length: [[classLabelVector] * featureVector.shape[1]],
            },
        )

    def calculateMeanPosteriors(self, session, taskId):
        sampleCount = 0
        segments = self.getSegmentFeaturesFromHdf(self.data_indices[taskId - 1])

        for batch in get_batch_from_segments(segments, self.n_batch):
            bSize = len(batch)
            denom = sampleCount + bSize
            if len(batch) == 0:
                break
            encoderOutput = self.getEncoderOutput(session, batch)
            for pastContextId in range(self.n_contexts):
                for currentState in range(self.n_state_classes):
                    denselabel = self.get_dense_label(pastLabel=pastContextId, centerState=currentState)
                    p = self.getPosteriorsOfOutputsWithEncoderOutput(session, encoderOutput[0], denselabel)
                    # triphone is calculates for each center and left context
                    tri = (sampleCount * self.triphone_means[pastContextId][currentState]) + (
                        bSize * np.mean(p[0][0], axis=0)
                    )
                    self.triphone_means[pastContextId][currentState] = np.divide(tri, denom)
                    # diphone is calculated for each context with centerstate 0
                    if not currentState:
                        di = (sampleCount * self.diphone_means[pastContextId]) + (bSize * np.mean(p[1][0], axis=0))
                        self.diphone_means[pastContextId] = np.divide(di, denom)
                        # context is not label dependent
                        if not pastContextId:
                            ctx = (sampleCount * self.context_means) + (bSize * np.mean(p[2][0], axis=0))
                            self.context_means = np.divide(ctx, denom)
            sampleCount += bSize

        with open(self.numSegments[taskId - 1].get_path(), "wb") as fp:
            pickle.dump(sampleCount, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def dumpMeans(self, taskId):
        with open(self.triphone_files[taskId - 1].get_path(), "wb") as f1:
            pickle.dump(self.triphone_means, f1, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.diphone_files[taskId - 1].get_path(), "wb") as f2:
            pickle.dump(self.diphone_means, f2, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.context_means[taskId - 1].get_path(), "wb") as f3:
            pickle.dump(self.context_means, f3, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, taskId):
        tf.load_op_library(self.tf_lib)
        mg = tf.compat.v1.MetaGraphDef()
        mg.ParseFromString(open(self.graph_path.get_path(), "rb").read())
        tf.compat.v1.import_graph_def(mg.graph_def, name="")
        # session
        s = tf.compat.v1.Session()
        returnValue = s.run(["save/restore_all"], feed_dict={"save/Const:0": self.model.get_path()})

        self.calculateMeanPosteriors(s, taskId)
        self.dumpMeans(taskId)


class DumpXmlForTriphoneForwardJob(Job):
    def __init__(self, triphone_files, diphone_files, context_files, num_segment_files, n_contexts, n_state_classes):
        self.triphone_files = triphone_files
        self.diphone_files = diphone_files
        self.context_files = context_files
        self.num_segment_files = num_segment_files
        self.numSegments = []
        self.triphone_means, self.diphone_means = initialize_dicts(
            n_contexts=n_contexts, n_state_classes=n_state_classes
        )
        self.contextMeans = []
        self.triphone_xml = self.output_path("triphone_scores.xml", cached=False)
        self.diphone_xml = self.output_path("diphone_scores.xml", cached=False)
        self.context_xml = self.output_path("context_scores.xml", cached=False)
        self.n_contexts = n_contexts
        self.n_state_classes = n_state_classes
        self.rqmt = {"cpu": 1, "mem": 1, "time": 1}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def readNumSegments(self):
        for filename in self.num_segment_files:
            with open(filename.get_path(), "rb") as f:
                self.numSegments.append(pickle.load(f))

    def calculateWeightedAverages(self):
        coeffs = [self.numSegments[i] / np.sum(self.numSegments) for i in range(len(self.num_segment_files))]
        for filename in self.triphone_files:
            with open(filename.get_path(), "rb") as f:
                triphoneDict = pickle.load(f)
                for i in range(self.n_contexts):
                    for j in range(self.n_state_classes):
                        self.triphone_means[i][j].append(
                            np.dot(coeffs[self.triphone_files.index(filename)], triphoneDict[i][j])
                        )
        for filename in self.diphone_files:
            with open(filename.get_path(), "rb") as f:
                diphoneDict = pickle.load(f)
                for i in range(self.n_contexts):
                    self.diphone_means[i].append(np.dot(coeffs[self.diphone_files.index(filename)], diphoneDict[i]))
        for filename in self.context_files:
            with open(filename.get_path(), "rb") as f:
                means = pickle.load(f)
                self.contextMeans.append(np.dot(coeffs[self.context_files.index(filename)], means))
        for i in range(self.n_contexts):
            self.diphone_means[i] = np.sum(self.diphone_means[i], axis=0)
            for j in range(self.n_state_classes):
                self.triphone_means[i][j] = np.sum(self.triphone_means[i][j], axis=0)
        self.contextMeans = np.sum(self.contextMeans, axis=0)

    def dumpXml(self):
        for pastId in range(self.n_contexts):
            for currentstateId in range(self.n_state_classes):
                for i, s in enumerate(self.triphone_means[pastId][currentstateId]):
                    if s == 0:
                        self.triphone_means[pastId][currentstateId][i] += 1e-5
        with open(self.triphone_xml.get_path(), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<matrix-f32 nRows="%d" nColumns="%d">\n'
                % (self.n_contexts * self.n_state_classes, self.n_contexts)
            )
            for pastId in range(self.n_contexts):
                for currentstateId in range(self.n_state_classes):
                    for i, s in enumerate(self.triphone_means[pastId][currentstateId]):
                        if s == 0:
                            self.triphone_means[pastId][currentstateId][i] += 1e-5
                    f.write(" ".join("%.20e" % math.log(s) for s in self.triphone_means[pastId][currentstateId]) + "\n")
            f.write("</matrix-f32>")
        with open(self.diphone_xml.get_path(), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<matrix-f32 nRows="%d" nColumns="%d">\n'
                % (self.n_contexts, self.n_state_classes)
            )
            for pastId in range(self.n_contexts):
                for i, c in enumerate(self.diphone_means[pastId]):
                    if c == 0:
                        self.diphone_means[pastId][i] += 1e-5
                f.write(" ".join("%.20e" % math.log(s) for s in self.diphone_means[pastId]) + "\n")
            f.write("</matrix-f32>")
        with open(self.context_xml.get_path(), "wt") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n' % (self.n_contexts))
            f.write(" ".join("%.20e" % math.log(s) for s in np.nditer(self.contextMeans)) + "\n")
            f.write("</vector-f32>")

    def run(self):
        self.readNumSegments()
        print("number of segments read")
        self.calculateWeightedAverages()
        self.dumpXml()
