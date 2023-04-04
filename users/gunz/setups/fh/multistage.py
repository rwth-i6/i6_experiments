__all__ = [
    "transform_checkpoint",
    "Shape",
    "Transformation",
    "TransformCheckpointJob",
    "Init",
    "InitNewLayersTransformation",
    "ResizeLayersTransformation",
]

import copy
from enum import Enum
import logging
import os
import typing

import numpy as np

from sisyphus import tk, Task

import i6_core.returnn as returnn

from ..common.compile_graph import compile_tf_graph_from_returnn_config
from .factored import LabelInfo

if typing.TYPE_CHECKING:
    from tensorflow.core.framework.variable_pb2 import VariableDef
    import tensorflow as tf


Shape2 = typing.Tuple[int, int]
Shape3 = typing.Tuple[int, int, int]
Shape = typing.Union[Shape2, Shape3]


class Transformation:
    def transform(
        self,
        var_data: typing.Dict[str, np.ndarray],
        input_mg: "tf.compat.v1.MetaGraphDef",
        output_mg: "tf.compat.v1.MetaGraphDef",
        input_vars: typing.Dict[str, "VariableDef"],
        output_vars: typing.Dict[str, "VariableDef"],
    ):
        return var_data

    def hash(self, values):
        return None


class TransformCheckpointJob(tk.Job):
    """
    This jobs serves as a framework to transform the contents of a tensorflow checkpoint.

    Given an input MetaGraphDef and a checkpoint this job will load the contents of all
    trainable variables into a python-dict and use the provided transformations to modify
    the variables. A new checkpoint is created based on the output MetaGraphDef.

    A typical use-case for this job would be to facilitate graph-transformations (e.g. quantization,
    compression, optimization).
    """

    def __init__(
        self,
        input_mg_path: typing.Union[str, tk.Path],
        output_mg_path: typing.Union[str, tk.Path],
        input_checkpoint: typing.Union[str, tk.Path, returnn.Checkpoint],
        transformations: typing.List[Transformation],
        tf_op_libraries: typing.Optional[typing.List[tk.Path]] = None,
    ):
        assert all(isinstance(t, Transformation) for t in transformations)

        self.input_mg_path = input_mg_path
        self.output_mg_path = output_mg_path
        self.input_checkpoint = input_checkpoint
        self.transformations = transformations
        self.tf_op_libraries = [] if tf_op_libraries is None else tf_op_libraries

        self.index_path = self.output_path("checkpoint.index")
        self._checkpoint_path = os.path.join(os.path.dirname(self.index_path.get_path()), "checkpoint")
        self.output_ckpt = returnn.Checkpoint(self.index_path)

        self.rqmt = {"cpu": 1, "mem": 8.0, "time": 1.0}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import tensorflow as tf
        from tensorflow.core.framework.variable_pb2 import VariableDef

        if len(self.tf_op_libraries):
            tf.load_op_library(self.tf_op_libraries)

        def load_graph(meta_path):
            mg = tf.compat.v1.MetaGraphDef()
            with open(tk.uncached_path(meta_path), "rb") as f:
                mg.ParseFromString(f.read())
            tf.import_graph_def(mg.graph_def, name="")

            return mg

        def load_checkpoint(session, mg, checkpoint_path):
            session.run(tf.compat.v1.global_variables_initializer())
            session.run(
                mg.saver_def.restore_op_name,
                feed_dict={mg.saver_def.filename_tensor_name: checkpoint_path},
            )

        def parse_variables(mg, collection="trainable_variables"):
            res = {}
            for s in mg.collection_def[collection].bytes_list.value:
                v = VariableDef()
                v.ParseFromString(s)
                res[v.variable_name] = v
            return res

        input_mg = load_graph(self.input_mg_path)
        output_mg = load_graph(self.output_mg_path)

        tf_input_vars = parse_variables(input_mg)
        tf_output_vars = parse_variables(output_mg)
        all_output_vars = parse_variables(output_mg, "variables")

        with tf.device("/CPU:0"):
            s = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={"GPU": 0}))
            tf.import_graph_def(input_mg.graph_def, name="")
        load_checkpoint(s, input_mg, tk.uncached_path(self.input_checkpoint))
        var_data = s.run({v.variable_name: v.snapshot_name for v in tf_input_vars.values()})
        s.close()
        tf.compat.v1.reset_default_graph()

        for k, v in var_data.items():
            logging.info("Input: %s shape: %s", k, str(v.shape))

        for t in self.transformations:
            var_data = t.transform(var_data, input_mg, output_mg, dict(tf_input_vars), dict(tf_output_vars))

        for k, v in var_data.items():
            logging.info("Output: %s shape: %s", k, str(v.shape))

        with tf.device("/CPU:0"):
            s = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={"GPU": 0}))
            tf.import_graph_def(output_mg.graph_def, name="")

        s.run(tf.compat.v1.global_variables_initializer())

        for v in var_data:
            if v in tf_output_vars:
                s.run(
                    tf_output_vars[v].initializer_name,
                    feed_dict={tf_output_vars[v].initial_value_name: var_data[v]},
                )
            else:
                logging.warning("Transformed variable %s not in output graph", v)
        for v in tf_output_vars:
            if v not in var_data:
                logging.warning("Variable %s not set", v)

        # this is a bit ugly, but the global step does not appear in the variable definitions of the meta_graph
        s.run("global_step/Assign", feed_dict={"global_step/Initializer/zeros:0": 0})

        s.run(
            output_mg.saver_def.save_tensor_name,
            feed_dict={output_mg.saver_def.filename_tensor_name: tk.uncached_path(self._checkpoint_path)},
        )

    @classmethod
    def hash(cls, kwargs):
        d = {k: kwargs[k] for k in ["input_mg_path", "output_mg_path", "input_checkpoint"]}
        d["transformations"] = [
            (type(t).__name__, t.hash(kwargs["transformations"][0])) for t in kwargs["transformations"]
        ]

        return super().hash(d)


class Init(Enum):
    zero = "zero"
    random_normal = "random_normal"

    def get_value(self, shape: Shape) -> np.ndarray:
        if self == Init.zero:
            return np.zeros(shape)
        elif self == Init.random_normal:
            return np.random.normal(0, 0.01, np.prod(shape)).reshape(shape)
        else:
            raise NotImplementedError(f"unimplemented init algorithm")


class InitNewLayersTransformation(Transformation):
    """
    Initializes the weights of layers that do not exist in the original config.
    """

    def __init__(self, init: Init) -> None:
        super().__init__()

        self.init = init

    def transform(
        self,
        var_data: typing.Dict[str, np.ndarray],
        input_mg: "tf.compat.v1.MetaGraphDef",
        output_mg: "tf.compat.v1.MetaGraphDef",
        input_vars: typing.Dict[str, "VariableDef"],
        output_vars: typing.Dict[str, "VariableDef"],
    ) -> typing.Dict[str, np.ndarray]:
        import tensorflow as tf

        g_out = tf.Graph()
        with g_out.as_default():
            tf.import_graph_def(output_mg.graph_def, name="")

        to_init = [layer for layer in output_vars.keys() if layer not in input_vars]
        for var_name in to_init:
            shape = tuple(g_out.get_tensor_by_name(var_name).shape.as_list())
            logging.info(f"initializing {var_name}:{shape} with {self.init}")

            var_data[var_name] = self.init.get_value(shape)

        return var_data

    @classmethod
    def hash(cls, kwargs):
        args = copy.deepcopy(vars(kwargs))
        hash_keys = args.keys()
        d = {k: vars(kwargs)[k] for k in hash_keys}
        return d


class ResizeLayersTransformation(Transformation):
    """
    This just pads the existing layers w/ zeros, dynamically computing the required
    amount from the graph.
    """

    def needs_extension(
        self,
        var_name: str,
        input_g: "tf.Graph",
        output_g: "tf.Graph",
    ) -> bool:
        try:
            input_var = input_g.get_tensor_by_name(var_name)
            output_var = output_g.get_tensor_by_name(var_name)
        except:
            return False

        assert len(output_var.shape) == len(input_var.shape)

        shape_diff = [a - b for a, b in zip(input_var.shape.as_list(), output_var.shape.as_list())]

        return any(d != 0 for d in shape_diff)

    def transform(
        self,
        var_data: typing.Dict[str, np.ndarray],
        input_mg: "tf.compat.v1.MetaGraphDef",
        output_mg: "tf.compat.v1.MetaGraphDef",
        input_vars: typing.Dict[str, "VariableDef"],
        output_vars: typing.Dict[str, "VariableDef"],
    ) -> typing.Dict[str, np.ndarray]:
        import tensorflow as tf

        g_in = tf.Graph()
        with g_in.as_default():
            tf.import_graph_def(input_mg.graph_def, name="")
        g_out = tf.Graph()
        with g_out.as_default():
            tf.import_graph_def(output_mg.graph_def, name="")

        to_extend = [layer for layer in output_vars.keys() if self.needs_extension(layer, g_in, g_out)]
        for layer in to_extend:
            in_sh = tuple(g_in.get_tensor_by_name(layer).shape.as_list())
            out_sh = tuple(g_out.get_tensor_by_name(layer).shape.as_list())

            logging.info(f"padding {layer} {in_sh} -> {out_sh}")

            copied = np.array(var_data[layer])
            copied.resize(out_sh)

            var_data[layer] = copied

        return var_data

    @classmethod
    def hash(cls, kwargs):
        args = copy.deepcopy(vars(kwargs))
        hash_keys = args.keys()
        d = {k: vars(kwargs)[k] for k in hash_keys}
        return d


def transform_checkpoint(
    name: typing.Optional[str],
    input_returnn_config: returnn.ReturnnConfig,
    input_label_info: LabelInfo,
    input_model_path: typing.Union[str, tk.Path, returnn.Checkpoint],
    output_returnn_config: returnn.ReturnnConfig,
    output_label_info: LabelInfo,
    *,
    init_new: Init = Init.zero,
    returnn_root: typing.Union[None, str, tk.Path] = None,
    returnn_python_exe: typing.Union[None, str, tk.Path] = None,
    tf_library: typing.Optional[typing.Union[str, tk.Path]] = None,
):
    """
    Transforms the weights of one checkpoint to be compatible with the other.

    Initializes new weights to zero and widens existing weights by zero-extending
    them to the required dimension.

    Used for phonetic multistage training.
    """

    n_state_diff = output_label_info.get_n_state_classes() - input_label_info.get_n_state_classes()
    assert n_state_diff == 0, "do not initialize models w/ different number of center states"

    input_graph = compile_tf_graph_from_returnn_config(
        input_returnn_config.config,
        python_epilog=input_returnn_config.python_epilog,
        python_prolog=input_returnn_config.python_prolog,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
    )
    output_graph = compile_tf_graph_from_returnn_config(
        output_returnn_config.config,
        python_epilog=output_returnn_config.python_epilog,
        python_prolog=output_returnn_config.python_prolog,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
    )

    logging.debug(f"IN: {input_returnn_config.config['extern_data']}")
    logging.debug(f"OUT: {output_returnn_config.config['extern_data']}")

    j = TransformCheckpointJob(
        input_mg_path=input_graph,
        input_checkpoint=input_model_path,
        output_mg_path=output_graph,
        transformations=[
            InitNewLayersTransformation(init_new),
            ResizeLayersTransformation(),
        ],
        tf_op_libraries=tf_library,
    )

    if name is not None:
        j.add_alias(f"transform/{name}")

    returnn_config = copy.deepcopy(output_returnn_config)
    returnn_config.config["preload_from_files"] = {
        "existing-model": {
            "init_for_train": True,
            "ignore_missing": True,
            "filename": j.output_ckpt,
        }
    }

    return returnn_config
