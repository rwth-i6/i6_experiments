"""
Convert any dataset to HDF
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Any, Callable, Dict, Tuple

from sisyphus import tk
from i6_core.util import instanciate_delayed

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.common.setups import serialization
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, ForwardRFDef, serialize_model_def
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim, TensorDict


def dataset_post_proc_seqs_to_hdf(
    dataset: DatasetConfig,
    *,
    map_seq: Optional[Callable] = None,
    map_seq_stream: Optional[Callable] = None,
    map_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tk.Path:
    """
    Given some dataset,
    apply some mapping function to each sequence or a stream of sequences
    (via :class:`PostprocessingDataset`),
    and save the result to HDF.
    """
    from .forward_to_hdf import forward_to_hdf

    assert map_seq or map_seq_stream

    dataset = dataset.copy_as_static()
    if map_outputs:
        dataset.extern_data = map_outputs
    assert dataset.main_dataset
    dataset.main_dataset = {
        "class": "PostprocessingDataset",
        "dataset": dataset.main_dataset,
        **({"map_seq": map_seq} if map_seq else {}),
        **({"map_seq_stream": map_seq_stream} if map_seq_stream else {}),
        "map_outputs": dataset.extern_data,
    }

    return forward_to_hdf(dataset=dataset)


def _make_specific_default_input(dataset: DatasetConfig, new_default_input: str = "data") -> DatasetConfig:
    default_input = dataset.get_default_input()
    if default_input == new_default_input:
        return dataset
    dataset = dataset.copy_as_static()
    dataset.main_dataset = {
        "class": "MetaDataset",
        "datasets": {"sub": dataset.main_dataset},
        "data_map": {new_default_input if k == default_input else k: ("sub", k) for k in dataset.extern_data.keys()},
        "seq_order_control_dataset": "sub",
    }
    assert new_default_input not in dataset.extern_data
    dataset.extern_data[new_default_input] = dataset.extern_data.pop(default_input)
    return dataset


def spm_to_chars_hdf(dataset: DatasetConfig) -> tk.Path:
    """
    Convert a dataset with SentencePiece labels to chars.
    """
    return dataset_post_proc_seqs_to_hdf(dataset, map_seq=_spm_to_chars_map_seq)


def _spm_to_chars_map_seq(seq: TensorDict) -> TensorDict:
    from returnn.tensor import TensorDict

    data = seq["data"]
    assert data.sparse_dim and data.vocab  # TODO or how else would we get the vocab?

    label_seq = [data.vocab.labels[label_idx] for label_idx in data.raw_tensor]
    label_seq_s = "".join(label_seq)
    label_seq_s = label_seq_s.replace("▁", " ")
    # TODO...

    # TODO where to get the output spatial dim from? rf.get_run_ctx().expected_outputs?
    #   but this here is on a single seq, not batched

    return TensorDict()
