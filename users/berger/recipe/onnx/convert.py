__all__ = ["ConvertTFCheckpointToOnnxJob"]


from typing import List, Optional, Dict

from sisyphus import Job, Task, tk

from i6_core.returnn.training import Checkpoint


class ConvertTFCheckpointToOnnxJob(Job):
    """
    Converts a given TensorFlow checkpoint to ONNX format.
    """

    __sis_hash_exclude__ = {"state_mapping": None, "meta_graph": None}

    def __init__(
        self,
        checkpoint: Checkpoint,
        input_names: List[str],
        output_names: List[str],
        meta_graph: tk.Path = None,
        state_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        :param i6_core.returnn.training.Checkpoint checkpoint: TensorFlow checkpoint
        :param list[str] input_names: input tensor names
        :param list[str] output_names: output tensor names.
        :param tk.Path meta_graph: TensorFlow meta graph path
        :param dict[str, str] state_mapping: mapping of input state names to the corresponding output state names.
        These names should not appear in output_names or input_names.
        """
        self.checkpoint = checkpoint
        self.meta_graph = meta_graph
        self.input_names = input_names
        self.output_names = output_names
        self.state_mapping = state_mapping

        if state_mapping is not None:
            for input_state, output_state in state_mapping.items():
                assert (
                    input_state not in self.input_names
                ), f"Input state {input_state} should not appear in the list of input names {self.input_names}."
                assert (
                    output_state not in self.output_names
                ), f"Output state {output_state} should not appear in the list of output names {self.output_names}."

                self.input_names.append(input_state)
                self.output_names.append(output_state)

        self.out_onnx_path = self.output_path("model.onnx")

        self.rqmt = {"cpu": 1, "mem": 8.0, "time": 1.0}

    def tasks(self):
        yield Task(
            "run",
            resume="run",
            rqmt=self.rqmt,
            mini_task=True if self.rqmt is None else False,
        )

    def run(self):
        import tensorflow as tf

        from tf2onnx.tfonnx import process_tf_graph
        from tf2onnx import utils, optimizer
        from tf2onnx.tf_loader import (
            tf_session,
            tf_reset_default_graph,
            tf_import_meta_graph,
            tf_optimize,
            freeze_session,
            remove_redundant_inputs,
            inputs_without_resource,
        )

        if not self.meta_graph:
            meta_graph = f"{self.checkpoint.ckpt_path}.meta"
        else:
            meta_graph = self.meta_graph

        tf_reset_default_graph()
        with tf.device("/cpu:0"):
            with tf_session() as sess:
                saver = tf_import_meta_graph(meta_graph, clear_devices=True)
                saver.restore(sess, self.checkpoint.ckpt_path)

                input_names = inputs_without_resource(sess, self.input_names)
                frozen_graph = freeze_session(sess, input_names=input_names, output_names=self.output_names)
                input_names = remove_redundant_inputs(frozen_graph, input_names)

            tf_reset_default_graph()
            with tf_session() as sess:
                frozen_graph = tf_optimize(input_names, self.output_names, frozen_graph)

        tf_reset_default_graph()
        with tf.device("/cpu:0"):
            with tf.Graph().as_default() as tf_graph:
                tf.import_graph_def(frozen_graph, name="")
                g = process_tf_graph(
                    tf_graph,
                    continue_on_error=False,
                    opset=None,
                    input_names=input_names,
                    output_names=self.output_names,
                )

                onnx_graph = optimizer.optimize_graph(g, catch_errors=True)
                model_proto = onnx_graph.make_model(f"converted from {self.meta_graph}")

        if self.state_mapping is not None:
            for input_state, output_state in self.state_mapping.items():
                input_state_key = "STATE_" + input_state
                metadata = model_proto.metadata_props.add()
                metadata.key = input_state_key
                metadata.value = output_state

        utils.save_protobuf(self.out_onnx_path.get_path(), model_proto)

