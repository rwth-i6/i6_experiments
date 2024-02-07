"""
Collection of utils for RETURNN.
"""
from typing import Any, Dict, Iterable
from i6_experiments.common.setups.rasr.util.nn.data import AllowedReturnnTrainingDataInput


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


def instanciate_returnn_data_inputs(datasets: Dict[str, AllowedReturnnTrainingDataInput]):
    """
    RETURNN datasets are often stored in a nested dict, e.g., like

    datasets = {
        "train": {...},
        "dev": {...},
        "eval_datasets": {"devtrain": {...}},
    }

    This helper iterates over all datasets and instanciates them by calling their `get_data_dict()` method, assuming 
    they are data input classes e.g. from i6_experiments.common.setups.rasr.util.nn.data.
    """
    for name, dataset in datasets.items():
        if name == "eval_datasets":
            instanciate_returnn_data_inputs(dataset)
        else:
            datasets[name] = dataset.get_data_dict()
