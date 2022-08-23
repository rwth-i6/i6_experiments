from sisyphus import *
from typing import Any, Dict, List, Optional

from i6_core.returnn import ReturnnConfig, ReturnnForwardJob


def get_returnn_length_hdfs(
    dataset_dict: Dict[str, Any],
    extern_data: Dict[str, Any],
    dataset_keys: List[str],
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    job_alias: Optional[str] = None,
    mem_rqmt: float = 4,
    time_rqmt: float = 4,
) -> Dict[str, tk.Path]:
    """
    Uses returnn to extract the length of sequences in the datasets after feature extraction, returns a separate hdf
    file per dataset key
    :param dataset_dict: RETURNN config dict for dataset
    :param extern_data: extern data dict matching the dataset
    :param dataset_keys: keys of the datastreams
    :param returnn_exe:
    :param returnn_root:
    :param job_alias: full alias of the forward job
    :param mem_rqmt: job memory requirement in GB
    :param time_rqmt: job time requirement in hours
    :return: dict of hdfs paths containing the lengths of a datastream
    """
    post_config = {"use_tensorflow": True}
    config = {
        "eval": dataset_dict,
        "network": {
            "output": {
                "class": "length",
                "axis": "T",
                "from": f"data:{dataset_keys[0]}",
            },
        },
        "extern_data": extern_data,
        "max_seqs": 50,
    }
    for idx, key in enumerate(dataset_keys[1:]):
        config["network"][f"length_{idx}"] = {
            "class": "length",
            "axis": "T",
            "from": f"data:{key}",
        }
        config["network"][f"dump_{idx}"] = {
            "class": "hdf_dump",
            "filename": f"{key}.hdf",
            "from": f"length_data_{idx}",
        }

    hdf_outputs = [k + ".hdf" for k in dataset_keys[1:]]
    config = ReturnnConfig(config=config, post_config=post_config)
    forward_job = ReturnnForwardJob(
        model_checkpoint=None,
        returnn_config=config,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        hdf_outputs=hdf_outputs,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
    )
    if job_alias is not None:
        forward_job.add_alias(job_alias)
    # remove .hdf extension and map default output to first dataset key
    hdf_dict = {k[:-4]: v for k, v in forward_job.out_hdf_files.items()}
    hdf_dict[dataset_keys[0]] = hdf_dict["output"]
    del hdf_dict["output"]
    return hdf_dict
