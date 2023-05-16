__all__ = ["read_prior_xml", "write_prior_xml"]

from dataclasses import dataclass
import numpy as np
import typing
import xml.etree.ElementTree as ET

from sisyphus import Path


@dataclass(frozen=True, eq=True)
class ParsedPriors:
    priors_log: typing.List[float]
    shape: typing.Union[typing.Tuple[int], typing.Tuple[int, int]]


def read_prior_xml(path: typing.Union[Path, str]) -> ParsedPriors:
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


def write_prior_xml(log_priors: np.ndarray, path: typing.Union[Path, str]):
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
