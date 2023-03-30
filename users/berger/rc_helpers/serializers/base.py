from typing import List
from sisyphus import tk
from i6_experiments.common.setups.returnn_common.serialization import (
    Collection,
    Network,
    ExternData,
    Import,
    PythonEnlargeStackWorkaroundNonhashedCode,
    PythonModelineNonhashedCode,
)


def get_base_rc_network_serializer(
    model_name: str,
    net_func_map: dict,
    extern_data: ExternData,
    network_kwargs: dict,
    returnn_common_root: tk.Path,
    module_import_path: str,
    additional_serializer_objects: List = [],
    additional_packages: set = set(),
) -> Collection:
    rc_construction_code = Import(f"{module_import_path}.models.{model_name}.construct_net_with_data")
    network = Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map=net_func_map,
        net_kwargs=network_kwargs,
    )

    serializer = Collection(
        serializer_objects=[
            PythonModelineNonhashedCode,
            PythonEnlargeStackWorkaroundNonhashedCode,
            extern_data,
            rc_construction_code,
            *additional_serializer_objects,
            network,
        ],
        returnn_common_root=returnn_common_root,
        make_local_package_copy=True,
        packages={module_import_path}.union(additional_packages),
    )

    return serializer
