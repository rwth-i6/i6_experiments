"""
Convert a given checkpoint.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional
import os.path
from sisyphus import Job, Task
from i6_core.returnn.training import Checkpoint

if TYPE_CHECKING:
    import numpy
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
    import returnn.frontend as rf


class ConvertTfCheckpointToRfPtJob(Job):
    """
    Convert the given TF checkpoint
    to a RETURNN frontend model with PT backend.
    """

    def __init__(
        self,
        *,
        checkpoint: Checkpoint,
        make_model_func: Callable[[], rf.Module],
        map_func: Callable[[CheckpointReader, str, rf.Parameter], numpy.ndarray],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        :param checkpoint: TF checkpoint
        :param make_model_func:
        :param map_func: (reader, name, var) -> var_value
        :param epoch: to store in the output checkpoint.
            If not given, try to infer from given checkpoint.
            It will error if this fails.
        :param step: to store in the output checkpoint
            If not given, try to infer from given checkpoint.
            It will error if this fails.
        """
        self.in_checkpoint = checkpoint
        self.make_model_func = make_model_func
        self.map_func = map_func
        self.epoch = epoch
        self.step = step
        self._out_model_dir = self.output_path("model", directory=True)
        self.out_checkpoint = self.output_path("model/checkpoint.pt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import returnn.frontend as rf
        from returnn.torch.frontend.bridge import rf_module_to_pt_module
        from returnn.util.basic import model_epoch_from_filename
        from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
        import torch
        import numpy

        reader = CheckpointReader(self.in_checkpoint.ckpt_path)
        print("Input checkpoint:")
        print(reader.debug_string().decode("utf-8"))
        print()

        print("Creating model...")
        rf.select_backend_torch()
        model = self.make_model_func()
        print("Created model:", model)
        print("Model parameters:")
        for name, param in model.named_parameters():
            assert isinstance(name, str)
            assert isinstance(param, rf.Parameter)
            print(f"{name}: {param}")
        print()

        for name, param in model.named_parameters():
            assert isinstance(name, str)
            assert isinstance(param, rf.Parameter)

            value = self.map_func(reader, name, param)
            assert isinstance(value, numpy.ndarray)
            # noinspection PyProtectedMember
            param._raw_backend.set_parameter_initial_value(param, value)

        epoch = self.epoch
        if epoch is None:
            epoch = model_epoch_from_filename(self.in_checkpoint.ckpt_path)

        step = self.step
        if step is None:
            assert reader.has_tensor("global_step")
            step = int(reader.get_tensor("global_step"))

        ckpt_name = os.path.basename(self.in_checkpoint.ckpt_path)

        pt_model = rf_module_to_pt_module(model)

        os.makedirs(self._out_model_dir.get_path(), exist_ok=True)
        filename = self._out_model_dir.get_path() + "/" + ckpt_name + ".pt"
        print(f"*** saving PyTorch model checkpoint: {filename}")
        torch.save({"model": pt_model.state_dict(), "epoch": epoch, "step": step}, filename)

        if ckpt_name != "checkpoint":
            symlink_filename = self._out_model_dir.get_path() + "/checkpoint.pt"
            print(f"*** creating symlink {symlink_filename} -> {os.path.basename(filename)}")
            os.symlink(os.path.basename(filename), symlink_filename)
        assert os.path.exists(self.out_checkpoint.get_path())
