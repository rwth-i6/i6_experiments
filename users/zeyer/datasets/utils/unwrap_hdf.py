"""
See :func:`i6_experiments.users.zeyer.forward_to_hdf.forward_to_hdf` for an explanation of the issue.
When using RETURNN :class:`SimpleHDFWriter`, e.g. via :func:`forward_to_hdf`,
the data in the HDF file was potentially modified, e.g. flattened.

Here we provide :func:`unwrap_hdf_dataset` to recover the original data format.
"""


from typing import Any, Dict
import functools
from returnn.tensor import TensorDict, Tensor, batch_dim


def unwrap_hdf_dataset(hdf_dataset: Dict[str, Any], *, extern_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recover the original data format, before it was dumped into HDF.
    See also the module docstring.

    :param hdf_dataset: dict for the HDF dataset, e.g. like {class: HDFDataset, files: [...]}.
    :param extern_data: the original data, which was dumped into the HDF via :class:`SimpleHDFWriter`,
        which we intend to recover like that
    :return: dataset which provides data matching to ``extern_data``. we use :class:`PostprocessingDataset` for that.
    """
    return {
        "class": "PostprocessingDataset",
        "dataset": hdf_dataset,
        "map_seq": functools.partial(_unwrap_hdf_extern_data, extern_data=extern_data),
        "map_outputs": {k: _wrap_extern_data_value_to_pp_dataset_map_outputs(v) for k, v in extern_data.items()},
    }


def _wrap_extern_data_value_to_pp_dataset_map_outputs(data: Dict[str, Any]) -> Dict[str, Any]:
    out = data.copy()
    if "dims" in data:
        assert data["dims"][0] == batch_dim
        out["dims"] = data["dims"][1:]
    elif "dim_tags" in data:
        assert data["dim_tags"][0].batch
        out["dim_tags"] = data["dim_tags"][1:]
    elif "shape" in data:
        pass  # nothing to do
    else:
        raise ValueError(f"don't know how to handle data {data}")
    return out


def _unwrap_hdf_extern_data(data: TensorDict, *, extern_data: Dict[str, Dict[str, Any]], **_other) -> TensorDict:
    # See returnn.datasets.hdf.SimpleHDFWriter._insert_h5_other for reference.
    # Also note that the handling for "data" is slightly different from for other keys.
    res = TensorDict()
    for key, out_data_template in extern_data.items():
        assert key in data.data
        in_data: Tensor = data.data[key]
        in_raw = in_data.raw_tensor
        out_data = Tensor(key, **out_data_template)
        assert in_data.sparse == out_data.sparse
        if len(in_data.dims) != len(out_data.dims):
            assert len(in_data.dims) < len(out_data.dims)
            # TODO...
        out_data.raw_tensor = in_raw
        res.data[key] = out_data
    # There might be other meta data in `data`, like "seq_tag", "complete_frac",
    # which we should just copy over.
    for key, in_data in data.data.items():
        if key in res.data:
            continue
        res.data[key] = in_data
    return res
