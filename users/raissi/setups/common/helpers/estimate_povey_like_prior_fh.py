__all__ = ["EstimateMonophonePriors_", "DumpXmlForMonophone"]


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
#from .utils import *

from i6_core.lib.rasr_cache import FileArchive

Path = setup_path(__package__)


###################################
# Monophone
###################################

def get_batch_from_segments(segments, batchSize=10000):
    index = 0
    while True:
        try:
            yield segments[index * batchSize: (index + 1) * batchSize]
            index += 1
        except IndexError:
            index = 0

class EstimateMonophonePriors_(Job):
    tm = {"diphone": "diphone"}
    __sis_hash_exclude__ = {"tensorMap": {"diphone": "center", "context": "context"}}

    def __init__(self, graph, model, dataPaths, datasetIndices, libraryPath,
                 nBatch=15000, nStates=141, gpu=1, mem=8, time=20, tensorMap=tm):

        self.graphPath = graph
        self.model = model
        self.dataPaths = dataPaths
        self.datasetIndices = datasetIndices
        self.additionalLibrary = libraryPath if libraryPath is not None else \
            "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/binaries/recognition/NativeLstm2.so"
        self.centerPhonemeMeans = np.zeros(nStates)
        self.numSegments = [self.output_path('segmentLength.%d' % index, cached=False) for index in self.datasetIndices]
        self.priorFiles = [self.output_path('centerPhonemeMeans.%d' % index, cached=False) for index in
                           self.datasetIndices]
        self.nBatch = nBatch
        self.nStates = nStates
        self.tensorMap = tensorMap
        if not gpu: time *= 4
        self.rqmt = {'cpu': 2, 'gpu': gpu, 'mem': mem, 'time': float(time)}

    def tasks(self):
        yield Task('run', resume='run', rqmt=self.rqmt, args=range(1, (len(self.datasetIndices) + 1)))

    def getSegmentFeaturesFromHdf(self, dataIndex):
        hf = h5py.File(tk.uncached_path(self.dataPaths[dataIndex]))
        segmentNames = list(hf['streams']['features']['data'])
        segments = []
        for name in segmentNames:
            segments.append(hf['streams']['features']['data'][name])
        return np.vstack(segments)

    def getPosteriors(self, session, featureVector):
        return session.run([("-").join([self.tensorMap['diphone'], 'output/output_batch_major:0'])],
                           feed_dict={'extern_data/placeholders/data/data:0':
                                          featureVector.reshape(1, featureVector.shape[0], featureVector.shape[1]),
                                      'extern_data/placeholders/data/data_dim0_size:0': [featureVector.shape[0]]})

    def calculateMeanPosteriors(self, session, taskId):
        sampleCount = 0
        segments = self.getSegmentFeaturesFromHdf(self.datasetIndices[taskId - 1])

        for batch in get_batch_from_segments(segments, self.nBatch):
            bSize = len(batch)
            denom = sampleCount + bSize
            if (len(batch) == 0):
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
        mg.ParseFromString(open(graphPath, 'rb').read())
        tf.import_graph_def(mg.graph_def, name='')
        # session
        s = tf.Session()
        returnValue = s.run(['save/restore_all'], feed_dict={'save/Const:0': modelPath})

        self.calculateMeanPosteriors(s, taskId)
        self.dumpMeans(taskId)

class DumpXmlForMonophone(Job):
    def __init__(self, priorFiles, numSegmentFiles, nStates=141):

        self.priorFiles = priorFiles
        self.numSegmentFiles = numSegmentFiles
        self.numSegments = []
        self.centerPhonemeMeans = []
        self.centerPhonemeXml = self.output_path('centerPhonemeScores.xml', cached=False)
        self.nStates = nStates
        self.rqmt = {'cpu': 2, 'mem': 4, 'time': 0.1}

    def tasks(self):
        yield Task('run', resume='run', rqmt=self.rqmt)

    def readNumSegments(self):
        for filename in self.numSegmentFiles:
            with open(tk.uncached_path(filename), 'rb') as f:
                self.numSegments.append(pickle.load(f))

    def calculateWeightedAverages(self):
        coeffs = [self.numSegments[i] / np.sum(self.numSegments) for i in range(len(self.numSegmentFiles))]
        for filename in self.priorFiles:
            with open(tk.uncached_path(filename), 'rb') as f:
                means = pickle.load(f)
                self.centerPhonemeMeans.append(np.dot(coeffs[self.priorFiles.index(filename)], means))
        self.centerPhonemeMeans = np.sum(self.centerPhonemeMeans, axis=0)

    def dumpXml(self):
        with open(tk.uncached_path(self.centerPhonemeXml), 'wt') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n' % (self.nStates))
            f.write(' '.join('%.20e' % math.log(s) for s in np.nditer(self.centerPhonemeMeans)) + '\n')
            f.write('</vector-f32>')

    def run(self):
        self.readNumSegments()
        self.calculateWeightedAverages()
        self.dumpXml()