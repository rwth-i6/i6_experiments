import copy
from enum import auto, Enum

from i6_core import returnn
from i6_experiments.users.gunz.setups.fh.factored import LabelInfo


class TemporalReductionMode(Enum):
    pooling = auto()
    throwaway = auto()


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
    temporal_reduction_mode: TemporalReductionMode = TemporalReductionMode.pooling,
    pool_mode: str = "avg",
    take_temporal_index: int = 0,
) -> returnn.ReturnnConfig:
    """
    Takes a model and subsamples the output by reducing the HMM states.
    """

    assert input_label_info.n_states_per_phone
    assert output_label_info.n_states_per_phone == 1
    assert output_label_info.phoneme_state_classes == input_label_info.phoneme_state_classes
    assert pool_mode in ["avg", "max"]
    assert input_label_info.n_states_per_phone > take_temporal_index >= 0

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

    if temporal_reduction_mode == TemporalReductionMode.pooling:
        network = {
            **network,
            output_center_softmax_layer_name: {
                "class": "pool",
                "from": "center-output-flatten",
                "mode": pool_mode,
                "padding": "same",
                "pool_size": (input_label_info.n_states_per_phone,),
                "register_as_extern_data": output_center_softmax_layer_name,
            },
            output_left_softmax_layer_name: {
                "class": "pool",
                "from": input_left_softmax_layer_name,
                "mode": pool_mode,
                "padding": "same",
                "pool_size": (input_label_info.n_states_per_phone,),
                "register_as_extern_data": output_left_softmax_layer_name,
            },
            output_right_softmax_layer_name: {
                "class": "pool",
                "from": input_right_softmax_layer_name,
                "mode": pool_mode,
                "padding": "same",
                "pool_size": (input_label_info.n_states_per_phone,),
                "register_as_extern_data": output_right_softmax_layer_name,
            },
        }
    elif temporal_reduction_mode.throwaway:

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
                },
            }

        for input_layer, out_layer in [
            (input_center_softmax_layer_name, output_center_softmax_layer_name),
            (input_left_softmax_layer_name, output_left_softmax_layer_name),
            (input_right_softmax_layer_name, output_right_softmax_layer_name),
        ]:
            network = add_throwaway(network, input_layer, out_layer, take_temporal_index)
    else:
        raise ValueError(f"unknown temporal reduction mode {temporal_reduction_mode}")

    returnn_config = copy.deepcopy(returnn_config)
    update_cfg = returnn.ReturnnConfig({"network": {**returnn_config.config["network"], **network}})
    returnn_config.update(update_cfg)
    return returnn_config
