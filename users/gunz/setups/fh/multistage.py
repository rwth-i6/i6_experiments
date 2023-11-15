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

from ..common.nn.compile_graph import compile_tf_graph_from_returnn_config
from .factored import LabelInfo

if typing.TYPE_CHECKING:
    from tensorflow.core.framework.variable_pb2 import VariableDef
    import tensorflow as tf


Shape2 = typing.Tuple[int, int]
Shape3 = typing.Tuple[int, int, int]
Shape = typing.Union[Shape2, Shape3]


class Transformation:
    def collect_shapes(self, vars: typing.Dict[str, "VariableDef"], gd: "tf.compat.v1.GraphDef"):
        import tensorflow as tf

        with tf.compat.v1.Session() as s:
            tf.import_graph_def(gd, name="")
            s.run(tf.compat.v1.global_variables_initializer())

            data = s.run({v.variable_name: v.initial_value_name for v in vars.values()})

        reverse_mapping = {v.variable_name: k for k, v in vars.items()}
        shapes = {reverse_mapping[k]: v.shape for k, v in data.items()}

        return shapes

    def transform(
        self,
        var_data: typing.Dict[str, np.ndarray],
        input_mg: "tf.compat.v1.MetaGraphDef",
        input_gd: "tf.compat.v1.GraphDef",
        output_mg: "tf.compat.v1.MetaGraphDef",
        output_gd: "tf.compat.v1.GraphDef",
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
        input_gd_path: typing.Union[str, tk.Path],
        output_mg_path: typing.Union[str, tk.Path],
        output_gd_path: typing.Union[str, tk.Path],
        input_checkpoint: typing.Union[str, tk.Path, returnn.Checkpoint],
        transformations: typing.List[Transformation],
        tf_op_libraries: typing.Optional[typing.List[tk.Path]] = None,
    ):
        assert all(isinstance(t, Transformation) for t in transformations)

        self.input_mg_path = input_mg_path
        self.input_gd_path = input_gd_path
        self.output_mg_path = output_mg_path
        self.output_gd_path = output_gd_path
        self.input_checkpoint = input_checkpoint
        self.transformations = transformations
        self.tf_op_libraries = [] if tf_op_libraries is None else tf_op_libraries

        self.index_path = self.output_path("checkpoint.index")
        self._checkpoint_path = os.path.join(os.path.dirname(self.index_path.get_path()), "checkpoint")
        self.output_ckpt = returnn.Checkpoint(self.index_path)

        self.rqmt = {"cpu": 1, "mem": 8.0, "time": 1.0}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import tensorflow as tf
        from tensorflow.core.framework.variable_pb2 import VariableDef

        tf.compat.v1.disable_eager_execution()

        if len(self.tf_op_libraries):
            tf.load_op_library(self.tf_op_libraries)

        def load_graph(meta_path):
            mg = tf.compat.v1.MetaGraphDef()
            with open(tk.uncached_path(meta_path), "rb") as f:
                mg.ParseFromString(f.read())
            tf.import_graph_def(mg.graph_def, name="")

            return mg

        def load_graph_def(gd_path):
            gd = tf.compat.v1.GraphDef()
            with open(gd_path, "rb") as f:
                gd.ParseFromString(f.read())
            return gd

        def load_checkpoint(session: tf.compat.v1.Session, mg, checkpoint_path):
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
        input_gd = load_graph_def(self.input_gd_path)
        output_mg = load_graph(self.output_mg_path)
        output_gd = load_graph_def(self.output_gd_path)

        tf_input_vars = parse_variables(input_mg)
        tf_output_vars = parse_variables(output_mg)

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={"GPU": 0})) as s:
            tf.import_graph_def(input_mg.graph_def, name="")
            load_checkpoint(s, input_mg, tk.uncached_path(self.input_checkpoint))
            var_data = s.run({v.variable_name: v.snapshot_name for v in tf_input_vars.values()})

        tf.compat.v1.reset_default_graph()

        for k, v in var_data.items():
            logging.info("Input: %s shape: %s", k, str(v.shape))

        for t in self.transformations:
            var_data = t.transform(
                var_data=var_data,
                input_mg=input_mg,
                input_gd=input_gd,
                output_mg=output_mg,
                output_gd=output_gd,
                input_vars=dict(tf_input_vars),
                output_vars=dict(tf_output_vars),
            )

        for k, v in var_data.items():
            logging.info("Output: %s shape: %s", k, str(v.shape))

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={"GPU": 0})) as s:
            tf.import_graph_def(output_mg.graph_def, name="")
            load_checkpoint(s, input_mg, tk.uncached_path(self.input_checkpoint))

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
            # s.run("global_step/Assign", feed_dict={"global_step/Initializer/zeros:0": 0})

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
    glorot_uniform = "glorot_uniform"

    def get_value(self, shape: Shape) -> np.ndarray:
        if self == Init.zero:
            return np.zeros(shape)
        elif self == Init.random_normal:
            return np.random.normal(0, 0.01, np.prod(shape)).reshape(shape)
        elif self == Init.glorot_uniform:
            lim = 1 / np.sqrt(shape[-1])
            return np.random.uniform(lim, lim, np.prod(shape)).reshape(shape)
        else:
            raise NotImplementedError(f"unimplemented init algorithm")


class InitNewLayersTransformation(Transformation):
    """
    Initializes the weights of layers that do not exist in the original config.
    """

    def __init__(self, init: Init, force_init: typing.Dict[str, tuple]) -> None:
        super().__init__()

        self.force_init = force_init
        self.init = init

    def transform(
        self,
        var_data: typing.Dict[str, np.ndarray],
        input_mg: "tf.compat.v1.MetaGraphDef",
        input_gd: "tf.compat.v1.GraphDef",
        output_mg: "tf.compat.v1.MetaGraphDef",
        output_gd: "tf.compat.v1.GraphDef",
        input_vars: typing.Dict[str, "VariableDef"],
        output_vars: typing.Dict[str, "VariableDef"],
    ) -> typing.Dict[str, np.ndarray]:
        to_init = [
            layer
            for layer in output_vars.keys()
            if layer not in input_vars
            or layer not in var_data
            or (self.force_init is not None and any(layer.startswith(l) for l in self.force_init))
        ]
        vars_to_init = {k: output_vars[k] for k in to_init}
        shapes = self.collect_shapes(vars_to_init, output_gd)

        for var_name in to_init:
            if self.force_init is not None and var_name in self.force_init:
                data = self.force_init[var_name]
                if isinstance(data, np.ndarray) or isinstance(data, list):
                    array = data if isinstance(data, np.ndarray) else np.array(data)
                    var_data[var_name] = array
                    logging.info(f"initializing {var_name} with data from dict {array.shape}")
                    continue
                else:
                    shape = self.force_init[var_name]
            else:
                shape = shapes[var_name]

            if (shape is None or len(shape) == 0) and var_name in var_data:
                # try taking shape from input
                shape = var_data[var_name].shape

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

    def transform(
        self,
        var_data: typing.Dict[str, np.ndarray],
        input_mg: "tf.compat.v1.MetaGraphDef",
        input_gd: "tf.compat.v1.GraphDef",
        output_mg: "tf.compat.v1.MetaGraphDef",
        output_gd: "tf.compat.v1.GraphDef",
        input_vars: typing.Dict[str, "VariableDef"],
        output_vars: typing.Dict[str, "VariableDef"],
    ) -> typing.Dict[str, np.ndarray]:
        shapes_in = {k: v.shape for k, v in var_data.items()}
        shapes_out = self.collect_shapes(output_vars, output_gd)

        needs_extension = [
            k
            for k in output_vars.keys()
            if k in shapes_in and any(a - b != 0 for a, b in zip(shapes_in[k], shapes_out[k]))
        ]

        no_extension_needed = set(output_vars.keys()) - set(needs_extension)
        for layer in sorted(no_extension_needed):
            logging.info(f"keeping {layer} {shapes_in.get(layer, None)} == {shapes_out.get(layer, None)}")

        for layer in needs_extension:
            in_sh = tuple(shapes_in[layer])
            out_sh = tuple(shapes_out[layer])

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
    force_init: typing.Optional[typing.Dict[str, typing.Union[tuple, np.ndarray, list]]] = None,
    init_new: Init = Init.zero,
    returnn_root: typing.Union[None, str, tk.Path] = None,
    returnn_python_exe: typing.Union[None, str, tk.Path] = None,
    tf_library: typing.Optional[typing.Union[str, tk.Path]] = None,
) -> returnn.ReturnnConfig:
    """
    Transforms the weights of one checkpoint to be compatible with the other.

    Initializes new weights to zero and widens existing weights by zero-extending
    them to the required dimension.

    Used for phonetic multistage training.
    """

    n_state_diff = output_label_info.get_n_state_classes() - input_label_info.get_n_state_classes()
    assert (
        force_init is None or any(k.startswith("center__output") for k in force_init) or n_state_diff == 0
    ), "do not initialize models w/ different number of center states"

    # Need both meta graph def and "plain" graph def format.
    #
    # The meta graph def contains the saver, while the plain one contains the shapes.
    input_graph_meta = compile_tf_graph_from_returnn_config(
        input_returnn_config, output_format="meta", returnn_root=returnn_root, returnn_python_exe=returnn_python_exe
    )
    input_graph_pb = compile_tf_graph_from_returnn_config(
        input_returnn_config, output_format="pb", returnn_root=returnn_root, returnn_python_exe=returnn_python_exe
    )
    output_graph_meta = compile_tf_graph_from_returnn_config(
        output_returnn_config, output_format="meta", returnn_root=returnn_root, returnn_python_exe=returnn_python_exe
    )
    output_graph_pb = compile_tf_graph_from_returnn_config(
        output_returnn_config, output_format="pb", returnn_root=returnn_root, returnn_python_exe=returnn_python_exe
    )

    logging.debug(f"IN: {input_returnn_config.config['extern_data']}")
    logging.debug(f"OUT: {output_returnn_config.config['extern_data']}")

    j = TransformCheckpointJob(
        input_mg_path=input_graph_meta,
        input_gd_path=input_graph_pb,
        input_checkpoint=input_model_path,
        output_mg_path=output_graph_meta,
        output_gd_path=output_graph_pb,
        transformations=[
            InitNewLayersTransformation(init_new, force_init or {}),
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
