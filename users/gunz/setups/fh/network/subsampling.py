__all__ = ["PoolingMode", "PoolingReduction", "TemporalReduction", "SelectOneReduction", "reduce_output_step_rate"]

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Union

from i6_core import returnn
from ....setups.fh.factored import LabelInfo


class PoolingMode(Enum):
    avg = "avg"
    max = "max"

    def __str__(self):
        return self.value


@dataclass(frozen=True, eq=True)
class PoolingReduction:
    mode: PoolingMode

    def __str__(self):
        return f"pool{self.mode}"

    @classmethod
    def avg(cls):
        return cls(mode=PoolingMode.avg)

    @classmethod
    def max(cls):
        return cls(mode=PoolingMode.max)


@dataclass(frozen=True, eq=True)
class SelectOneReduction:
    take_i: int

    def __str__(self):
        return f"sel{self.take_i}"


TemporalReduction = Union[PoolingReduction, SelectOneReduction]


def reduce_output_step_rate(
    returnn_config: returnn.ReturnnConfig,
    input_label_info: LabelInfo,
    output_label_info: LabelInfo,
    *,
    input_center_softmax_layer_name: str = "center-output",
    output_center_softmax_layer_name: str = "center-output-ss",
    input_left_softmax_layer_name: str = "left-output",
    output_left_softmax_layer_name: str = "left-output-ss",
    input_right_softmax_layer_name: str = "right-output",
    output_right_softmax_layer_name: str = "right-output-ss",
    temporal_reduction: TemporalReduction = PoolingReduction(mode=PoolingMode.avg),
) -> returnn.ReturnnConfig:
    """
    Takes a model and subsamples the output by reducing the HMM states.
    """

    assert input_label_info.n_states_per_phone
    assert output_label_info.n_states_per_phone == 1
    assert output_label_info.phoneme_state_classes == input_label_info.phoneme_state_classes
    assert isinstance(temporal_reduction, (PoolingReduction, SelectOneReduction))
    assert (
        not isinstance(temporal_reduction, SelectOneReduction)
        or input_label_info.n_states_per_phone > temporal_reduction.take_i >= 0
    )

    network = {
        "center-output-window": {
            "class": "split_dims",
            "from": input_center_softmax_layer_name,
            "axis": "F",
            "dims": (-1, input_label_info.n_states_per_phone, input_label_info.phoneme_state_classes.factor()),
        },
        "center-output-reduce": {
            "class": "reduce",
            "from": "center-output-window",
            "mode": "sum",
            "axis": f"dim:{input_label_info.n_states_per_phone}",
        },
        "center-output-flatten": {
            "class": "merge_dims",
            "from": "center-output-reduce",
            "axes": "except_time",
        },
    }

    if isinstance(temporal_reduction, PoolingReduction):

        def add_pool(network: dict, in_layer: str, out_layer: str):
            return {
                **network,
                f"{out_layer}_pool": {
                    "class": "pool",
                    "from": in_layer,
                    "mode": str(temporal_reduction.mode),
                    "padding": "same",
                    "pool_size": (input_label_info.n_states_per_phone,),
                },
                f"{out_layer}_sum": {
                    "class": "reduce",
                    "from": f"{out_layer}_pool",
                    "axes": "F",
                    "mode": "sum",
                },
                f"{out_layer}_renorm": {
                    "class": "combine",
                    "from": [f"{out_layer}_pool", f"{out_layer}_sum"],
                    "kind": "truediv",
                },
                out_layer: {
                    "class": "copy",
                    "from": f"{out_layer}_renorm",
                    "register_as_extern_data": out_layer,
                },
            }

        for input_layer, output_layer in [
            ("center-output-flatten", output_center_softmax_layer_name),
            (input_left_softmax_layer_name, output_left_softmax_layer_name),
            (input_right_softmax_layer_name, output_right_softmax_layer_name),
        ]:
            network = add_pool(network, input_layer, output_layer)
    elif isinstance(temporal_reduction, SelectOneReduction):

        def add_throwaway(network: dict, in_layer: str, out_layer: str, take_n: int):
            return {
                **network,
                f"{out_layer}_split": {
                    "class": "split_dims",
                    "axis": "T",
                    "dims": (-1, input_label_info.n_states_per_phone),
                    "from": in_layer,
                },
                out_layer: {
                    "axis": f"dim:{input_label_info.n_states_per_phone}",
                    "class": "gather",
                    "from": f"{out_layer}_split",
                    "position": take_n,
                    "register_as_extern_data": out_layer,
                },
            }

        for input_layer, out_layer in [
            ("center-output-flatten", output_center_softmax_layer_name),
            (input_left_softmax_layer_name, output_left_softmax_layer_name),
            (input_right_softmax_layer_name, output_right_softmax_layer_name),
        ]:
            network = add_throwaway(network, input_layer, out_layer, temporal_reduction.take_i)
    else:
        raise ValueError(f"unknown temporal reduction mode {temporal_reduction}")

    returnn_config = copy.deepcopy(returnn_config)
    update_cfg = returnn.ReturnnConfig({"network": {**returnn_config.config["network"], **network}})
    returnn_config.update(update_cfg)
    return returnn_config
