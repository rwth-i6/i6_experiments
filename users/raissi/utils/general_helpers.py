import numpy as np
import pickle
import xml.etree.ElementTree as ET


def dump_pickle(path, file):
    with open(path, "wb") as f:
        pickle.dump(file, f)

def load_pickle(path):
    with open(path, "rb") as f:
        l = pickle.load(f)
    return l

def read_text(path):
    with open(path, "rt") as f:
        file = f.read().splitlines()
    return file

def write_text(path, values):
    with open(path, 'w') as f:
        for item in values:
            f.write("%s\n" % item)


def get_prior_from_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    values = [np.exp(float(v)) for v in root.text.split(" ")]
    values_log = [float(v) for v in root.text.split(" ")]
    priors = {'prior_probability': values, 'prior_log_probability': values_log}

    return priors


def createTriAndDiDictsWithZeros(nContexts, nStateClasses):
  triDict = {}
  for i in range(nContexts):
    triDict[i] = dict(zip(range(nStateClasses), [np.zeros(nContexts) for _ in range(nStateClasses)]))
  diDict = dict(zip(range(nContexts), [np.zeros(nStateClasses) for _ in range(nContexts)]))

  return triDict, diDict

def initializeTriAndDiDicts(nContexts=47, nStateClasses=282):
  triDict = {}
  for i in range(nContexts):
    triDict[i] = dict(zip(range(nStateClasses), [[] for _ in range(nStateClasses)]))
  diDict = dict(zip(range(nContexts), [[] for _ in range(nContexts)]))

  return triDict, diDict

