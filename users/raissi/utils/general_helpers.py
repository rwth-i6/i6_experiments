__all__ = ["get_prior_from_pickle"]
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
    with open(path, "w") as f:
        for item in values:
            f.write("%s\n" % item)


def get_prior_from_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    values = [np.exp(float(v)) for v in root.text.split(" ")]
    values_log = [float(v) for v in root.text.split(" ")]
    priors = {"prior_probability": values, "prior_log_probability": values_log}

    return priors


def get_prior_from_pickle(path):
    import pickle
    with open(path, "rb") as f:
        priors = pickle.load(f)
    return priors