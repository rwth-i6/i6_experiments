__all__ = [
    "EstimateMonophonePriors_",
    "DumpXmlForMonophone",
    "EstimateRasrDiphoneAndContextPriors",
    "DumpXmlRasrForDiphone",
]


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

# from .utils import *

from i6_core.lib.rasr_cache import FileArchive

Path = setup_path(__package__)


###################################
# Monophone
###################################


def get_batch_from_segments(segments, batchSize=10000):
    index = 0
    while True:
        try:
            yield segments[index * batchSize : (index + 1) * batchSize]
            index += 1
        except IndexError:
            index = 0


class EstimateMonophonePriors_(Job):
    tm = {"diphone": "diphone"}
    __sis_hash_exclude__ = {"tensorMap": {"diphone": "center", "context": "context"}}

    def __init__(
        self,
        graph,
        model,
        dataPaths,
        datasetIndices,
        libraryPath,
        nBatch=15000,
        nStates=141,
        gpu=1,
        mem=8,
        time=20,
        tensorMap=tm,
    ):

        self.graphPath = graph
        self.model = model
        self.dataPaths = dataPaths
        self.datasetIndices = datasetIndices
        self.additionalLibrary = (
            libraryPath
            if libraryPath is not None
            else "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/binaries/recognition/NativeLstm2.so"
        )
        self.centerPhonemeMeans = np.zeros(nStates)
        self.numSegments = [self.output_path("segmentLength.%d" % index, cached=False) for index in self.datasetIndices]
        self.priorFiles = [
            self.output_path("centerPhonemeMeans.%d" % index, cached=False) for index in self.datasetIndices
        ]
        self.nBatch = nBatch
        self.nStates = nStates
        self.tensorMap = tensorMap
        if not gpu:
            time *= 4
        self.rqmt = {"cpu": 2, "gpu": gpu, "mem": mem, "time": float(time)}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, (len(self.datasetIndices) + 1)))

    def getSegmentFeaturesFromHdf(self, dataIndex):
        hf = h5py.File(tk.uncached_path(self.dataPaths[dataIndex]))
        segmentNames = list(hf["streams"]["features"]["data"])
        segments = []
        for name in segmentNames:
            segments.append(hf["streams"]["features"]["data"][name])
        return np.vstack(segments)

    def getPosteriors(self, session, featureVector):
        return session.run(
            [("-").join([self.tensorMap["diphone"], "output/output_batch_major:0"])],
            feed_dict={
                "extern_data/placeholders/data/data:0": featureVector.reshape(
                    1, featureVector.shape[0], featureVector.shape[1]
                ),
                "extern_data/placeholders/data/data_dim0_size:0": [featureVector.shape[0]],
            },
        )

    def calculateMeanPosteriors(self, session, taskId):
        sampleCount = 0
        segments = self.getSegmentFeaturesFromHdf(self.datasetIndices[taskId - 1])

        for batch in get_batch_from_segments(segments, self.nBatch):
            bSize = len(batch)
            denom = sampleCount + bSize
            if len(batch) == 0:
                break

            p = self.getPosteriors(session, batch)

            for i in range(len(batch)):
                nominator = (sampleCount * self.centerPhonemeMeans) + (bSize * np.mean(p[0][0], axis=0))
                self.centerPhonemeMeans = np.divide(nominator, denom)
            sampleCount += bSize

        with open(tk.uncached_path(self.numSegments[taskId - 1]), "wb") as fp:
            pickle.dump(sampleCount, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def dumpMeans(self, taskId):
        with open(tk.uncached_path(self.priorFiles[taskId - 1]), "wb") as fp:
            pickle.dump(self.centerPhonemeMeans, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, taskId):
        graphPath = tk.uncached_path(self.graphPath)
        modelPath = tk.uncached_path(self.model.ckpt_path)
        tf.load_op_library(self.additionalLibrary)
        mg = tf.MetaGraphDef()
        mg.ParseFromString(open(graphPath, "rb").read())
        tf.import_graph_def(mg.graph_def, name="")
        # session
        s = tf.Session()
        returnValue = s.run(["save/restore_all"], feed_dict={"save/Const:0": modelPath})

        self.calculateMeanPosteriors(s, taskId)
        self.dumpMeans(taskId)


class DumpXmlForMonophone(Job):
    def __init__(self, priorFiles, numSegmentFiles, nStates=141):

        self.priorFiles = priorFiles
        self.numSegmentFiles = numSegmentFiles
        self.numSegments = []
        self.centerPhonemeMeans = []
        self.centerPhonemeXml = self.output_path("centerPhonemeScores.xml", cached=False)
        self.nStates = nStates
        self.rqmt = {"cpu": 2, "mem": 4, "time": 0.1}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def readNumSegments(self):
        for filename in self.numSegmentFiles:
            with open(tk.uncached_path(filename), "rb") as f:
                self.numSegments.append(pickle.load(f))

    def calculateWeightedAverages(self):
        coeffs = [self.numSegments[i] / np.sum(self.numSegments) for i in range(len(self.numSegmentFiles))]
        for filename in self.priorFiles:
            with open(tk.uncached_path(filename), "rb") as f:
                means = pickle.load(f)
                self.centerPhonemeMeans.append(np.dot(coeffs[self.priorFiles.index(filename)], means))
        self.centerPhonemeMeans = np.sum(self.centerPhonemeMeans, axis=0)

    def dumpXml(self):
        with open(tk.uncached_path(self.centerPhonemeXml), "wt") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n' % (self.nStates))
            f.write(" ".join("%.20e" % math.log(s) for s in np.nditer(self.centerPhonemeMeans)) + "\n")
            f.write("</vector-f32>")

    def run(self):
        self.readNumSegments()
        self.calculateWeightedAverages()
        self.dumpXml()


class EstimateRasrDiphoneAndContextPriors(Job):
    tm = {"diphone": "diphone", "context": "context"}
    __sis_hash_exclude__ = {"tensorMap": {"diphone": "center", "context": "context"}}

    def __init__(
        self,
        graphPath,
        model,
        dataPaths,
        datasetIndices,
        libraryPath,
        nBatch=10000,
        nStateClasses=141,
        nPhones=47,
        nContexts=47,
        nStates=3,
        gpu=1,
        mem=8,
        time=20,
        tensorMap=tm,
    ):
        self.graphPath = graphPath
        self.model = model
        self.dataPaths = dataPaths
        self.datasetIndices = datasetIndices
        self.additionalLibrary = libraryPath
        self.diphoneMeans = dict(zip(range(nContexts), [np.zeros(nStateClasses) for _ in range(nContexts)]))
        self.contextMeans = np.zeros(nContexts)
        self.numSegments = [self.output_path("segmentLength.%d" % index, cached=False) for index in self.datasetIndices]
        self.diphoneFiles = [self.output_path("diphoneMeans.%d" % index, cached=False) for index in self.datasetIndices]
        self.contextFiles = [self.output_path("contextMeans.%d" % index, cached=False) for index in self.datasetIndices]
        self.nCenterPhones = nPhones
        self.nContexts = nContexts
        self.nStates = nStates
        self.nBatch = nBatch
        self.tensorMap = tensorMap
        if not gpu:
            time *= 4
        self.rqmt = {"cpu": 2, "gpu": gpu, "mem": mem, "time": float(time)}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, (len(self.datasetIndices) + 1)))

    def getSegmentFeaturesFromHdf(self, dataIndex):
        hf = h5py.File(tk.uncached_path(self.dataPaths[dataIndex]))
        print(self.dataPaths[dataIndex])
        segmentNames = list(hf["streams"]["features"]["data"])
        segments = []
        for name in segmentNames:
            segments.append(hf["streams"]["features"]["data"][name])
        return np.vstack(segments)

    def getEncoderOutput(self, session, featureVector):
        return session.run(
            ["encoder-output/output_batch_major:0"],
            feed_dict={
                "extern_data/placeholders/data/data:0": featureVector.reshape(
                    1, featureVector.shape[0], featureVector.shape[1]
                ),
                "extern_data/placeholders/data/data_dim0_size:0": [featureVector.shape[0]],
            },
        )

    def getPosteriorsOfBothOutputsWithEncoded(self, session, featureVector, classLabelVector):
        return session.run(
            [
                ("-").join([self.tensorMap["diphone"], "output/output_batch_major:0"]),
                ("-").join([self.tensorMap["context"], "output/output_batch_major:0"]),
            ],
            feed_dict={
                "concat_fwd_6_bwd_6/concat_sources/concat:0": featureVector.reshape(
                    featureVector.shape[1], 1, featureVector.shape[2]
                ),
                "extern_data/placeholders/classes/classes:0": [[classLabelVector] * featureVector.shape[1]],
            },
        )

    def get_dense_label(self, pastLabel, centerPhoneme=0, stateId=0, futureLabel=0):
        return (
            ((((centerPhoneme * self.nStates) + stateId) * self.nContexts) + pastLabel) * self.nContexts
        ) + futureLabel

    def calculateMeanPosteriors(self, session, taskId):
        sampleCount = 0
        segments = self.getSegmentFeaturesFromHdf(self.datasetIndices[taskId - 1])

        for batch in get_batch_from_segments(segments, self.nBatch):
            bSize = len(batch)
            denom = sampleCount + bSize
            if len(batch) == 0:
                break

            encoderOutput = self.getEncoderOutput(session, batch)
            for pastContextId in range(self.nContexts):
                p = self.getPosteriorsOfBothOutputsWithEncoded(
                    session, encoderOutput[0], self.get_dense_label(pastContextId)
                )

                di = (sampleCount * self.diphoneMeans[pastContextId]) + (bSize * np.mean(p[0][0], axis=0))
                self.diphoneMeans[pastContextId] = np.divide(di, denom)
                # context is not label dependent
                if not pastContextId:
                    ctx = (sampleCount * self.contextMeans) + (bSize * np.mean(p[1][0], axis=0))
                    self.contextMeans = np.divide(ctx, denom)
            sampleCount += bSize

        with open(tk.uncached_path(self.numSegments[taskId - 1]), "wb") as fp:
            pickle.dump(sampleCount, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def dumpMeans(self, taskId):
        with open(tk.uncached_path(self.diphoneFiles[taskId - 1]), "wb") as fp:
            pickle.dump(self.diphoneMeans, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tk.uncached_path(self.contextFiles[taskId - 1]), "wb") as fp:
            pickle.dump(self.contextMeans, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, taskId):
        graphPath = tk.uncached_path(self.graphPath)
        modelPath = tk.uncached_path(self.model.ckpt_path)
        tf.load_op_library(self.additionalLibrary)
        mg = tf.MetaGraphDef()
        mg.ParseFromString(open(graphPath, "rb").read())
        tf.import_graph_def(mg.graph_def, name="")
        # session
        s = tf.Session()
        returnValue = s.run(["save/restore_all"], feed_dict={"save/Const:0": modelPath})

        self.calculateMeanPosteriors(s, taskId)
        self.dumpMeans(taskId)


# you can use DumpXmlForDiphone and have an attribute called isSprint, with which you call your additional function.
# Generally think to merge all functions
class DumpXmlRasrForDiphone(Job):
    def __init__(
        self,
        diphoneFiles,
        contextFiles,
        numSegmentFiles,
        nContexts,
        nStateClasses,
        adjustSilence=True,
        adjustNonWord=False,
        silBoundaryIndices=None,
        nonWordIndices=None,
    ):

        self.diphoneFiles = diphoneFiles
        self.contextFiles = contextFiles
        self.numSegmentFiles = numSegmentFiles
        self.numSegments = []
        self.diphoneMeans = dict(zip(range(nContexts), [[] for _ in range(nContexts)]))
        self.contextMeans = []
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

    def readNumSegments(self):
        for filename in self.numSegmentFiles:
            with open(tk.uncached_path(filename), "rb") as f:
                self.numSegments.append(pickle.load(f))

    def calculateWeightedAverages(self):
        coeffs = [self.numSegments[i] / np.sum(self.numSegments) for i in range(len(self.numSegmentFiles))]
        for filename in self.diphoneFiles:
            with open(tk.uncached_path(filename), "rb") as f:
                diphoneDict = pickle.load(f)
                for i in range(self.nContexts):
                    self.diphoneMeans[i].append(np.dot(coeffs[self.diphoneFiles.index(filename)], diphoneDict[i]))
        for filename in self.contextFiles:
            with open(tk.uncached_path(filename), "rb") as f:
                means = pickle.load(f)
                self.contextMeans.append(np.dot(coeffs[self.contextFiles.index(filename)], means))
        for i in range(self.nContexts):
            self.diphoneMeans[i] = np.sum(self.diphoneMeans[i], axis=0)
        self.contextMeans = np.sum(self.contextMeans, axis=0)

    def setSilenceAndNonWordValues(self):
        # context vectors
        sil = sum([self.contextMeans[i] for i in self.silBoundaryIndices])
        noise = sum([self.contextMeans[i] for i in self.nonWordIndices])

        # center given context vectors
        meansListSil = [self.diphoneMeans[i] for i in self.silBoundaryIndices]
        meansListNonword = [self.diphoneMeans[i] for i in self.nonWordIndices]
        dpSil = [sum(x) for x in zip(*meansListSil)]
        dpNoise = [sum(x) for x in zip(*meansListNonword)]

        for i in self.silBoundaryIndices:
            self.contextMeans[i] = sil
            self.diphoneMeans[i] = dpSil
        for i in self.nonWordIndices:
            self.contextMeans[i] = noise
            self.diphoneMeans[i] = dpNoise

    def setSilenceValues(self):
        sil = sum([self.contextMeans[i] for i in self.silBoundaryIndices])

        # center given context vectors
        meansListSil = [self.diphoneMeans[i] for i in self.silBoundaryIndices]
        dpSil = [np.sum(x) for x in zip(*meansListSil)]

        for i in self.silBoundaryIndices:
            self.contextMeans[i] = sil
            self.diphoneMeans[i] = dpSil

    def dumpXml(self):
        perturbation = 1e-8
        with open(tk.uncached_path(self.diphoneXml), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<matrix-f32 nRows="%d" nColumns="%d">\n'
                % (self.nContexts, self.nStateClasses)
            )
            for i in range(self.nContexts):
                self.diphoneMeans[i][self.diphoneMeans[i] == 0] = perturbation
                f.write(" ".join("%.20e" % math.log(s) for s in self.diphoneMeans[i]) + "\n")
            f.write("</matrix-f32>")
        with open(tk.uncached_path(self.contextXml), "wt") as f:
            self.contextMeans[self.contextMeans == 0] = perturbation
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n' % (self.nContexts))
            f.write(" ".join("%.20e" % math.log(s) for s in np.nditer(self.contextMeans)) + "\n")
            f.write("</vector-f32>")

    def dumpPickle(self):
        with open("/u/raissi/experiments/notebooks/diphones.pickle", "wb") as fp:
            pickle.dump(self.diphoneMeans, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open("/u/raissi/experiments/notebooks/context.pickle", "wb") as fp:
            pickle.dump(self.contextMeans, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        self.readNumSegments()
        self.calculateWeightedAverages()
        if self.adjustSilence:
            if self.adjustNonWord:
                self.setSilenceAndNonWordValues()
            else:
                self.setSilenceValues()

        self.dumpXml()
        self.dumpPickle()
