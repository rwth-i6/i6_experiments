"""
Serialization helpers for RETURNN, such as ReturnnConfig
"""


from __future__ import annotations
from copy import deepcopy
from i6_core.returnn.config import ReturnnConfig

# The code here does not need the user to use returnn_common.
# However, we internally make use of some helper code from returnn_common.
from returnn_common.nn.naming import ReturnnDimTagsProxy, ReturnnConfigSerializer


def get_serializable_config(config: ReturnnConfig) -> ReturnnConfig:
    """
    Takes the config, goes through the config (e.g. network dict)
    and replaces some non-serializable objects (e.g. dim tags) with serializable ones.
    (Currently, it is all about dim tags.)
    """
    config = deepcopy(config)
    dim_tag_proxy = ReturnnDimTagsProxy()
    config.config = dim_tag_proxy.collect_dim_tags_and_transform_config(config.config)
    config.post_config = dim_tag_proxy.collect_dim_tags_and_transform_config(config.post_config)
    config.staged_network_dict = dim_tag_proxy.collect_dim_tags_and_transform_config(config.staged_network_dict)

    if not dim_tag_proxy.dim_refs_by_name:
        # No dim tags found, just return as-is.
        return config

    # Prepare object to use config.update(),
    # because config.update() does reasonable logic for python_epilog code merging,
    # including handling of python_epilog_hash.
    python_epilog_ext = []
    dim_tag_def_code = dim_tag_proxy.py_code_str()
    for code in [ReturnnConfigSerializer.ImportPyCodeStr, dim_tag_def_code]:
        if config.python_epilog and code not in config.python_epilog:
            python_epilog_ext.append(code)
    config_update = ReturnnConfig(
        {}, python_epilog=python_epilog_ext, hash_full_python_code=config.hash_full_python_code
    )
    config.update(config_update)

    return config
