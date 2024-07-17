from dataclasses import dataclass
from i6_core.tools import CloneGitRepositoryJob
from typing import List, Union, Callable, Optional
import functools
from sisyphus import tk


def skip_layer(network: dict, layer_name: str) -> None:
    """Removes a layer from the network and connects its outputs to its inputs directly instead."""
    if not isinstance(network, dict):
        return

    layer_dict = network.pop(layer_name, None)
    if not layer_dict:
        return

    layer_from = layer_dict.get("from", "data")

    change_source_name(network, orig_name=layer_name, new_name=layer_from)


def change_source_name(network: dict, orig_name: str, new_name: Union[str, List[str]]):
    """Goes through the network and changes all appearances of orig_name in fromLists to new_name."""
    if not isinstance(network, dict):
        return

    if isinstance(new_name, str):
        new_name = [new_name]

    for x, attributes in network.items():
        if not isinstance(attributes, dict):
            continue
        if "from" in attributes:
            from_list = attributes["from"]
            if isinstance(from_list, str) and from_list == orig_name:
                attributes["from"] = new_name
            elif isinstance(from_list, list) and orig_name in from_list:
                index = from_list.index(orig_name)
                attributes["from"] = from_list[:index] + new_name + from_list[index + 1 :]

        if "subnetwork" in attributes:
            change_source_name(
                attributes["subnetwork"],
                "base:" + orig_name,
                [f"base:{name}" for name in new_name],
            )

        if "unit" in attributes:
            change_source_name(
                attributes["unit"],
                "base:" + orig_name,
                [f"base:{name}" for name in new_name],
            )


def recursive_update(orig_dict: dict, update: dict):
    """Recursively updates dict and sub-dicts."""

    for k, v in update.items():
        if isinstance(v, dict):
            orig_dict[k] = recursive_update(orig_dict.get(k, {}), v)
        else:
            orig_dict[k] = v

    return orig_dict


def lru_cache_with_signature(_func=None, *, maxsize=None, typed=False):
    def decorator(func: Callable):
        cached_function = functools.lru_cache(maxsize=maxsize, typed=typed)(func)
        functools.update_wrapper(cached_function, func)
        return cached_function

    if _func is None:
        return decorator
    else:
        return decorator(_func)


@dataclass
class ToolPaths:
    returnn_root: Optional[tk.Path] = None
    returnn_python_exe: Optional[tk.Path] = None
    rasr_binary_path: Optional[tk.Path] = None
    returnn_common_root: Optional[tk.Path] = None
    blas_lib: Optional[tk.Path] = None
    rasr_python_exe: Optional[tk.Path] = None

    def __post_init__(self) -> None:
        if self.rasr_python_exe is None:
            self.rasr_python_exe = self.returnn_python_exe


default_tools = ToolPaths(
    returnn_root=tk.Path("/u/berger/software/returnn"),
    returnn_python_exe=tk.Path("/usr/bin/python3"),
    # returnn_python_exe=tk.Path("/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3.8"),
    rasr_binary_path=tk.Path("/u/berger/software/rasr_apptainer/arch/linux-x86_64-standard"),
    returnn_common_root=tk.Path("/u/berger/software/returnn_common"),
    # blas_lib=tk.Path(
    #     "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so"
    # ),
)

default_tools_v2 = ToolPaths(
    # returnn_root=tk.Path("/u/rossenbach/src/NoReturnn", hash_overwrite="/u/berger/repositories/returnn"),
    returnn_root=tk.Path("/u/berger/repositories/returnn"),
    returnn_python_exe=tk.Path("/usr/bin/python3"),
    rasr_binary_path=tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard"),
    returnn_common_root=tk.Path("/u/berger/repositories/returnn_common"),
)

default_tools_apptek = ToolPaths(
    returnn_root=CloneGitRepositoryJob("https://github.com/rwth-i6/returnn.git").out_repository,
    returnn_python_exe=tk.Path("/usr/bin/python3"),
    rasr_binary_path=tk.Path("/home/sberger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard"),
    returnn_common_root=tk.Path(""),
)
