__all__ = ["read_prior_xml", "write_prior_xml"]

from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Union
import xml.etree.ElementTree as ET

from sisyphus import Path


@dataclass(frozen=True, eq=True)
class PartitionDataSetup:
    n_segment_indices: int = 20
    n_data_indices: int = 3
    segment_offset: int = 10
    data_offset: int = 10
    split_step: int = 200


@dataclass(frozen=True, eq=True)
class ParsedPriors:
    priors_log: List[float]
    shape: Union[Tuple[int], Tuple[int, int]]


def read_prior_xml(path: Union[Path, str]) -> ParsedPriors:
    tree = ET.parse(path.get_path())

    root = tree.getroot()

    n_rows = root.attrib.get("nRows")
    n_cols = root.attrib.get("nColumns")
    vec_size = root.attrib.get("size")

    priors = [float(num.strip()) for num in root.text.strip().split()]

    if vec_size is not None:
        shape = (int(vec_size),)
    else:
        assert n_cols is not None and n_rows is not None
        shape = (int(n_rows), int(n_cols))

    return ParsedPriors(priors_log=priors, shape=shape)


def write_prior_xml(log_priors: np.ndarray, path: Union[Path, str]):
    if log_priors.ndim == 1:
        attrs = {"size": str(len(log_priors))}
        element = "vector-f32"
        table = " ".join(f"{v:.20e}" for v in log_priors)
    elif log_priors.ndim == 2:
        attrs = {"nRows": str(log_priors.shape[0]), "nColumns": str(log_priors.shape[1])}
        element = "matrix-f32"
        table = "\n".join(" ".join(f"{v:.20e}" for v in row) for row in log_priors[:])
    else:
        raise f"unsupported prior array dim: {log_priors.ndim}"

    node = ET.Element(element, attrib=attrs)
    node.text = f"\n{table}\n"

    ET.ElementTree(node).write(path, encoding="utf-8")


def initialize_dicts_with_zeros(n_contexts: int, n_states: int, isForward=True):
    triDict = {}
    for i in range(n_contexts):
        triDict[i] = dict(zip(range(n_states), [np.zeros(n_contexts) for _ in range(n_states)]))
    if isForward:
        diDict = dict(zip(range(n_contexts), [np.zeros(n_states) for _ in range(n_contexts)]))
    else:
        diDict = dict(zip(range(n_states), [np.zeros(n_contexts) for _ in range(n_states)]))
    return triDict, diDict


def initialize_dicts(n_contexts: int, n_state_classes: int, isForward=True):
    triDict = {}
    for i in range(n_contexts):
        triDict[i] = dict(zip(range(n_state_classes), [[] for _ in range(n_state_classes)]))
    if isForward:
        diDict = dict(zip(range(n_contexts), [[] for _ in range(n_contexts)]))
    else:
        diDict = dict(zip(range(n_state_classes), [[] for _ in range(n_state_classes)]))
    return triDict, diDict


def get_batch_from_segments(segments: List, batchSize=10000):
    index = 0
    while True:
        try:
            yield segments[index * batchSize : (index + 1) * batchSize]
            index += 1
        except IndexError:
            index = 0
