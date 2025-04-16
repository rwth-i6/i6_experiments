"""
Concatenate multiple HDF datasets into a single HDF dataset.
"""


from typing import Any, Sequence, Dict
from sisyphus import tk
from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf
from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic


def concat_hdfs(
    hdf_files: Sequence[tk.Path], *, extern_data: Dict[str, Dict[str, Any]], default_input: str = "data"
) -> tk.Path:
    """
    Concatenate multiple HDF files into a single one.
    This uses :func:`i6_experiments.users.zeyer.forward_to_hdf.forward_to_hdf`.

    :return: new HDF
    """
    from i6_experiments.users.zeyer.datasets.utils.unwrap_hdf import unwrap_hdf_dataset

    hdf_dataset_dict = {"class": "HDFDataset", "files": list(hdf_files), "use_cache_manager": True}
    unwrapped_ds_dict = unwrap_hdf_dataset(hdf_dataset_dict, extern_data=extern_data)

    ds = DatasetConfigStatic(
        main_name="hdfs",
        main_dataset=unwrapped_ds_dict,
        extern_data=extern_data,
        default_input=default_input,
        use_deep_copy=True,
    )
    return forward_to_hdf(dataset=ds, config={"backend": "torch", "behavior_version": 24}, forward_mem_rqmt=20)
