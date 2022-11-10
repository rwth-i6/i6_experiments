import os
import shutil
import subprocess as sp

from sisyphus import Job, Task, gs, tk

from i6_core.returnn.training import Checkpoint, GetBestEpochJob as _GetBestEpochJob, GetBestTFCheckpointJob as _GetBestTFCheckpointJob


class GetBestEpochJob(_GetBestEpochJob):
    pass


class GetBestCheckpointJob(_GetBestTFCheckpointJob):
    pass


class AverageCheckpointsJob(Job):

    def __init__(self, model_dir, epochs, returnn_python_exe, returnn_root):
        """

        :param tk.Path model_dir:
        :param list[int|tk.Path] epochs:
        :param tk.Path returnn_python_exe:
        :param tk.Path returnn_root:
        """
        self.model_dir = model_dir
        self.epochs = epochs
        self.returnn_python_exe   = returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE
        self.returnn_root         = returnn_root       if returnn_root is not None else gs.RETURNN_ROOT

        self.avg_model_dir = self.output_path("avg_model", directory=True)
        self.avg_epoch = self.output_var("epoch")

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        import numpy
        epochs = [epoch.get() if isinstance(epoch, tk.Variable) else epoch for epoch in self.epochs]
        avg_epoch = int(numpy.round(numpy.average(epochs), 0))
        args = [
            self.returnn_python_exe.get_path(),
            os.path.join(
                self.returnn_root.get_path(), "tools/tf_avg_checkpoints.py"),
            "--checkpoints", ','.join([str(epoch) for epoch in self.epochs]),
            "--prefix", self.model_dir.get_path() + "/epoch.",
            "--output_path", os.path.join(self.avg_model_dir.get_path(), "epoch.%.3d" % avg_epoch)
        ]
        sp.check_call(args)
        self.avg_epoch.set(avg_epoch)


class AverageCheckpointsJobV2(Job):

    def __init__(self, model_dir, epochs, returnn_python_exe, returnn_root):
        """

        :param tk.Path model_dir:
        :param list[int|tk.Path] epochs:
        :param tk.Path returnn_python_exe:
        :param tk.Path returnn_root:
        """
        self.model_dir = model_dir
        self.epochs = epochs
        self.returnn_python_exe   = returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE
        self.returnn_root         = returnn_root       if returnn_root is not None else gs.RETURNN_ROOT

        self.out_checkpoint = Checkpoint(self.output_path("epoch.001.index"))

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        args = [
            self.returnn_python_exe.get_path(),
            os.path.join(
                self.returnn_root.get_path(), "tools/tf_avg_checkpoints.py"),
            "--checkpoints", ','.join([str(epoch) for epoch in self.epochs]),
            "--prefix", self.model_dir.get_path() + "/epoch.",
            "--output_path", self.out_checkpoint.index_path.get_path()[:-len(".index")]
        ]
        sp.check_call(args, env={"CUDA_VISIBLE_DEVICES": ""})
