from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional
import os.path
from sisyphus import Job, Task
from i6_core.returnn.training import Checkpoint
import torch

if TYPE_CHECKING:
    import numpy
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader


class ConvertTfCheckpointToPtJob(Job):
    """
    Convert the given TF checkpoint to PT checkpoint.
    """

    def __init__(
        self,
        *,
        checkpoint: Checkpoint,
        make_model_func: Callable[[], torch.nn.Module],
        param_map_func: Callable[[CheckpointReader, str, torch.nn.Parameter], numpy.ndarray],
        buffer_map_func: Callable[[CheckpointReader, str, torch.Tensor], numpy.ndarray],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        :param checkpoint: TF checkpoint
        :param make_model_func:
        :param param_map_func: (reader, name, var) -> var_value
        :param buffer_map_func: (reader, name, var) -> var_value
        :param epoch: to store in the output checkpoint.
            If not given, try to infer from given checkpoint.
            It will error if this fails.
        :param step: to store in the output checkpoint
            If not given, try to infer from given checkpoint.
            It will error if this fails.
        """
        self.in_checkpoint = checkpoint
        self.make_model_func = make_model_func
        self.param_map_func = param_map_func
        self.buffer_map_func = buffer_map_func
        self.epoch = epoch
        self.step = step
        self._out_model_dir = self.output_path("model", directory=True)
        self.out_checkpoint = self.output_path("model/checkpoint.pt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        from returnn.util.basic import model_epoch_from_filename
        from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
        import numpy

        reader = CheckpointReader(self.in_checkpoint.ckpt_path)
        print("Input checkpoint:")
        print(reader.debug_string().decode("utf-8"))
        print()

        print("Creating model...")
        pt_model = self.make_model_func()

        print("Created model:", pt_model)
        print("Model parameters:")
        for name, param in pt_model.named_parameters():
            assert isinstance(name, str)
            assert isinstance(param, torch.nn.Parameter)
            print(f"{name}: {param}")
        print()

        print("Model buffers:")
        for name, buf in pt_model.named_buffers():
            assert isinstance(name, str)
            assert isinstance(buf, torch.Tensor)
            print(f"{name}: {buf}")
        print()

        for name, param in pt_model.named_parameters():
            assert isinstance(name, str)
            assert isinstance(param, torch.nn.Parameter)

            value = self.param_map_func(reader, name, param)
            assert isinstance(value, numpy.ndarray)
            # noinspection PyProtectedMember
            param.data = torch.from_numpy(value)

        for name, buffer in pt_model.named_buffers():
            assert isinstance(name, str)
            assert isinstance(buffer, torch.Tensor)

            value = self.buffer_map_func(reader, name, buffer)
            assert isinstance(value, numpy.ndarray)
            # noinspection PyProtectedMember
            buffer.data = torch.from_numpy(value)

        epoch = self.epoch
        if epoch is None:
            epoch = model_epoch_from_filename(self.in_checkpoint.ckpt_path)

        step = self.step
        if step is None:
            assert reader.has_tensor("global_step")
            step = int(reader.get_tensor("global_step"))

        ckpt_name = os.path.basename(self.in_checkpoint.ckpt_path)

        os.makedirs(self._out_model_dir.get_path(), exist_ok=True)
        filename = self._out_model_dir.get_path() + "/" + ckpt_name + ".pt"
        print(f"*** saving PyTorch model checkpoint: {filename}")
        torch.save({"model": pt_model.state_dict(), "epoch": epoch, "step": step}, filename)

        if ckpt_name != "checkpoint":
            symlink_filename = self._out_model_dir.get_path() + "/checkpoint.pt"
            print(f"*** creating symlink {symlink_filename} -> {os.path.basename(filename)}")
            os.symlink(os.path.basename(filename), symlink_filename)
        assert os.path.exists(self.out_checkpoint.get_path())
