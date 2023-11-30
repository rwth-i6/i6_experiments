__all__ = ["get_extern_data_config"]

import typing

import i6_core.returnn as returnn

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo


def get_extern_data_config(
    label_info: LabelInfo,
    time_tag_name: typing.Optional[str],
    add_single_state_label: bool = False,
) -> typing.Dict[str, typing.Any]:

    conf = [
        ("classes", label_info.get_n_of_dense_classes(), True),
        ("centerState", label_info.get_n_state_classes(), True),
        ("pastLabel", label_info.n_contexts, True),
        ("futureLabel", label_info.n_contexts, True),
    ]
    if add_single_state_label:
        conf.append(("singleStateCenter", label_info.get_n_single_state_classes(), True))
    return {
        k: {
            "dim": dim,
            "dtype": "int32",
            "sparse": True,
            "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)} if time_tag_name is not None else None,
            "available_for_inference": inference,
        }
        for k, dim, inference in conf
    }
