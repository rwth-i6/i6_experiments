__all__ = ["get_tf_flow", "add_tf_flow_to_base_flow"]

from typing import List, Optional, Union

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.returnn as returnn


def get_tf_flow(
    checkpoint_path: Union[tk.Path, returnn.Checkpoint],
    tf_graph_path: tk.Path,
    returnn_op_path: Optional[Union[tk.Path, List[tk.Path]]] = None,
    forward_output_layer: str = "log_output",
    tf_fwd_input_name: str = "tf-fwd-input",
) -> rasr.FlowNetwork:
    """
    Create flow network and config for the tf-fwd node
    """
    tf_flow = rasr.FlowNetwork()

    tf_flow.add_input(tf_fwd_input_name)
    tf_flow.add_output("features")
    tf_flow.add_param("id")
    tf_fwd = tf_flow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
    tf_flow.link(f"network:{tf_fwd_input_name}", tf_fwd + ":input")
    tf_flow.link(tf_fwd + ":log-posteriors", "network:features")

    tf_flow.config = rasr.RasrConfig()

    tf_flow.config[tf_fwd].input_map.info_0.param_name = "input"
    tf_flow.config[
        tf_fwd
    ].input_map.info_0.tensor_name = "extern_data/placeholders/data/data"
    tf_flow.config[
        tf_fwd
    ].input_map.info_0.seq_length_tensor_name = (
        "extern_data/placeholders/data/data_dim0_size"
    )

    tf_flow.config[tf_fwd].output_map.info_0.param_name = "log-posteriors"
    tf_flow.config[
        tf_fwd
    ].output_map.info_0.tensor_name = f"{forward_output_layer}/output_batch_major"

    tf_flow.config[tf_fwd].loader.type = "meta"
    tf_flow.config[tf_fwd].loader.meta_graph_file = tf_graph_path
    tf_flow.config[tf_fwd].loader.saved_model_file = checkpoint_path

    if returnn_op_path is not None:
        tf_flow.config[tf_fwd].loader.required_libraries = returnn_op_path

    return tf_flow


def add_tf_flow_to_base_flow(
    base_flow: rasr.FlowNetwork,
    tf_flow: rasr.FlowNetwork,
    tf_fwd_input_name: str = "tf-fwd-input",
) -> rasr.FlowNetwork:
    """
    Integrate tf-fwd node into the regular flow network

    :param FlowNetwork base_flow:
    :param FlowNetwork tf_flow:
    :param str tf_fwd_input_name: see: get_tf_flow()
    :rtype: FlowNetwork
    """
    assert (
        len(base_flow.outputs) == 1
    ), "Not implemented otherwise"  # see hard coded tf-fwd input
    base_output = list(base_flow.outputs)[0]

    input_name = tf_fwd_input_name

    feature_flow = rasr.FlowNetwork()
    base_mapping = feature_flow.add_net(base_flow)
    tf_mapping = feature_flow.add_net(tf_flow)
    feature_flow.interconnect_inputs(base_flow, base_mapping)
    feature_flow.interconnect(
        base_flow, base_mapping, tf_flow, tf_mapping, {base_output: input_name}
    )
    feature_flow.interconnect_outputs(tf_flow, tf_mapping)

    return feature_flow
