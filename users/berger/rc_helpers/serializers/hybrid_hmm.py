import os
from sisyphus import tk

from i6_experiments.common.setups.returnn_common.serialization import DataInitArgs, DimInitArgs, ExternData
from .base import get_base_rc_network_serializer


def get_hybrid_hmm_rc_network_serializer(num_outputs: int, network_kwargs: dict, returnn_common_root: tk.Path):
    time_dim = DimInitArgs("data_time", dim=None)
    data_feature = DimInitArgs("data_feature", dim=1, is_feature=True)
    classes_feature = DimInitArgs("classes_feature", dim=num_outputs, is_feature=True)

    data_init = [
        DataInitArgs(
            name="data", available_for_inference=True, dim_tags=[time_dim, data_feature], sparse_dim=None, dtype="int16"
        ),
        DataInitArgs(
            name="classes",
            available_for_inference=False,
            dim_tags=[time_dim, classes_feature],
            sparse_dim=classes_feature,
            dtype="int32",
        ),
    ]
    extern_data = ExternData(data_init)
    module_path = "i6_experiments.users.berger.rc_modules"

    return get_base_rc_network_serializer(
        model_name=os.path.basename(__file__)[:-3],
        net_func_map={"audio_data": "data", "label_data": "classes"},
        extern_data=extern_data,
        network_kwargs=network_kwargs,
        returnn_common_root=returnn_common_root,
        module_import_path=module_path,
    )
