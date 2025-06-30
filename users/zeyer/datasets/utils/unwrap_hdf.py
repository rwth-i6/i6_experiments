"""
See :func:`i6_experiments.users.zeyer.forward_to_hdf.forward_to_hdf` for an explanation of the issue.
When using RETURNN :class:`SimpleHDFWriter`, e.g. via :func:`forward_to_hdf`,
the data in the HDF file was potentially modified, e.g. flattened.

Here we provide :func:`unwrap_hdf_dataset` to recover the original data format.
"""


from typing import Any, Dict
import functools
from returnn.tensor import TensorDict, Tensor, Dim, batch_dim


def unwrap_hdf_dataset(hdf_dataset: Dict[str, Any], *, extern_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recover the original data format, before it was dumped into HDF.
    See also the module docstring.

    :param hdf_dataset: dict for the HDF dataset, e.g. like {class: HDFDataset, files: [...]}.
    :param extern_data: the original data, which was dumped into the HDF via :class:`SimpleHDFWriter`,
        which we intend to recover like that
    :return: dataset which provides data matching to ``extern_data``. we use :class:`PostprocessingDataset` for that.
    """
    map_outputs = {k: _wrap_extern_data_value_to_pp_dataset_map_outputs(v) for k, v in extern_data.items()}
    return {
        "class": "PostprocessingDataset",
        "dataset": hdf_dataset,
        "map_seq": functools.partial(_unwrap_hdf_map_seq, map_outputs=map_outputs),
        "map_outputs": map_outputs,
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


def _unwrap_hdf_map_seq(data: TensorDict, *, map_outputs: Dict[str, Dict[str, Any]], **_other) -> TensorDict:
    # See returnn.datasets.hdf.SimpleHDFWriter._insert_h5_other for reference.
    # Also note that the handling for "data" is slightly different from for other keys.
    res = TensorDict()

    # Handle "data" first, as this is also serialized slightly differently,
    # and it will store dynamic sizes in "sizes" when there are multiple.
    if "data" in map_outputs:
        key = "data"
        out_data_template = map_outputs[key]
        in_data: Tensor = data.data[key]
        in_raw = in_data.raw_tensor
        out_data = Tensor(key, **out_data_template)
        assert batch_dim not in out_data.dims  # sanity check
        if not out_data.dims:
            # in_data should be of shape [1]
            assert in_raw.shape == (1,)
            in_raw = in_raw.squeeze(0)
        else:
            if in_raw.ndim == 2:  # assume [flat_seq_len,dim]
                assert out_data.dims[-1].dimension == in_raw.shape[-1]
                flattened_dims = out_data.dims[:-1]  # could also be empty
            elif in_raw.ndim == 1:  # assume [flat_seq_len]
                flattened_dims = out_data.dims
            else:
                raise ValueError(
                    f"unexpected 'data' {in_data} in_raw shape {in_raw.shape}, expect to convert to {out_data_template}"
                )
            if len(flattened_dims) > 1:
                # Expect "sizes" in data.
                assert "sizes" in data.data, f"got {data.data}, 'sizes' is missing"
                sizes = data.data["sizes"].raw_tensor
                assert sizes.ndim == 1 and sizes.shape[0] == len(flattened_dims)
                in_raw = in_raw.reshape(*sizes, *in_raw.shape[1:])
            elif len(flattened_dims) == 1:
                pass
                # No need to reshape, should already be good.
            else:  # no dims
                assert in_raw.shape[0] == 1  # there was a dummy time dim
                in_raw = in_raw.squeeze(0)
        out_data.raw_tensor = in_raw
        res.data[key] = out_data

    for key, out_data_template in map_outputs.items():
        if key == "data":
            continue
        assert key in data.data
        in_data: Tensor = data.data[key]
        in_raw = in_data.raw_tensor
        out_data = Tensor(key, **out_data_template)
        assert batch_dim not in out_data.dims  # sanity check
        if len(out_data.dims) == 0:
            assert in_raw.shape == (1,)
            in_raw = in_raw.squeeze(0)
        else:
            assert in_raw.ndim == len(out_data.dims)
        out_data.raw_tensor = in_raw
        res.data[key] = out_data

    # There might be other meta data in `data`, like "seq_tag", "complete_frac",
    # which we should just copy over.
    for key, in_data in data.data.items():
        if key in res.data:
            continue
        res.data[key] = in_data
    return res
