"""
Convert a given checkpoint.
"""

from __future__ import annotations

import os.path
from typing import TYPE_CHECKING, Callable
from sisyphus import Job, Task
from i6_core.returnn.training import Checkpoint

if TYPE_CHECKING:
    import numpy
    import tensorflow as tf
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
    from returnn_common import nn


class ConvertCheckpointJob(Job):
    """
    Convert the given checkpoint.
    """

    def __init__(
        self,
        *,
        checkpoint: Checkpoint,
        make_model_func: Callable[[], nn.Module],
        map_func: Callable[[CheckpointReader, str, tf.compat.v1.Variable], numpy.ndarray],
    ):
        """
        :param checkpoint:
        :param make_model_func:
        :param map_func: (reader, name, var) -> var_value
        """
        self.in_checkpoint = checkpoint
        self.make_model_func = make_model_func
        self.map_func = map_func
        self._out_model_dir = self.output_path("model", directory=True)
        self.out_checkpoint = Checkpoint(self.output_path("model/checkpoint.index"))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import numpy
        from returnn_common import nn
        from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
        import tensorflow as tf
        from glob import glob

        tf1 = tf
        if tf.__version__.startswith("2."):
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.disable_v2_tensorshape()
            tf1 = tf.compat.v1

        reader = CheckpointReader(self.in_checkpoint.ckpt_path)
        print("Input checkpoint:")
        print(reader.debug_string().decode("utf-8"))
        print()

        print("Creating model...")
        model = self.make_model_func()
        print("Created model:", model)
        print("Model parameters:")
        for name, param in model.named_parameters():
            assert isinstance(name, str)
            assert isinstance(param, nn.Parameter)
            print(f"{name}: {param}")
        print()

        tf_vars = []
        with tf1.Graph().as_default() as graph, tf1.Session(
            graph=graph, config=tf1.ConfigProto(device_count=dict(GPU=0))
        ).as_default() as session:

            for name, param in model.named_parameters():
                assert isinstance(name, str)
                assert isinstance(param, nn.Parameter)

                tf_var_name = name.replace(".", "/") + "/param"
                dtype = param.dtype
                shape = tuple(d.dimension for d in param.dims)
                tf_var = tf1.Variable(
                    name=tf_var_name,
                    initial_value=tf.zeros(shape, dtype=dtype),
                    dtype=dtype,
                    shape=shape,
                )
                value = self.map_func(reader, name, tf_var)
                tf_var.load(value, session=session)
                tf_vars.append(tf_var)

            if reader.has_tensor("global_step"):
                value = reader.get_tensor("global_step")
                tf_var = tf1.train.get_or_create_global_step()
                tf_var.load(value, session=session)
                tf_vars.append(tf_var)

        saver = tf1.train.Saver(var_list=tf_vars, max_to_keep=2**31 - 1)
        ckpt_name = os.path.basename(self.in_checkpoint.ckpt_path)
        saver.save(session, self._out_model_dir.get_path() + "/" + ckpt_name)

        if ckpt_name != "checkpoint":
            prefix = self._out_model_dir.get_path() + "/" + ckpt_name + "."
            for filename in glob(prefix + "*"):
                postfix = filename[len(prefix) :]
                os.symlink(
                    os.path.basename(filename),
                    self._out_model_dir.get_path() + "/checkpoint." + postfix,
                )
            assert os.path.exists(self.out_checkpoint.index_path.get_path())
