"""
Collection of utils for RETURNN.
"""
from typing import Any, Dict, Iterable


def iterate_returnn_datasets(datasets: Dict[str, Dict]) -> Iterable[Dict[str, Any]]:
    """
    RETURNN datasets are often stored in a nested dict, e.g., like

    datasets = {
        "train": {...},
        "dev": {...},
        "eval_datasets": {"devtrain": {...}},
    }

    This helper iterates over all datasets, alleviating the need to care about the nested structure of eval datasets.
    """
    for name, dataset in datasets.items():
        if name == "eval_datasets":
            yield from iterate_returnn_datasets(dataset)
        else:
            yield dataset

