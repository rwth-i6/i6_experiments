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
    ds = DatasetConfigStatic(
        main_name="hdfs",
        main_dataset={"class": "HDFDataset", "files": list(hdf_files), "use_cache_manager": True},
        extern_data=extern_data,
        default_input=default_input,
        use_deep_copy=True,
    )
    return forward_to_hdf(dataset=ds, forward_mem_rqmt=20)
