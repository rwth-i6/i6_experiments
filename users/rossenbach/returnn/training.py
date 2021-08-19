import os
import shutil
import subprocess as sp

from sisyphus import Job, Task, gs, tk

from i6_core.returnn.training import Checkpoint


class GetBestEpochJob(Job):
    """
    Provided a RETURNN model directory and an optional score key, finds the best epoch.
    The sorting is lower=better, so to acces the model with the highest values use negative index values (e.g. -1 for
    the model with the highest score)
    """

    def __init__(self, model_dir, learning_rates, index=0, key=None):
        """

        :param Path model_dir: model_dir output from a RETURNNTrainingJob
        :param Path learning_rates: learning_rates output from a RETURNNTrainingJob
        :param int index: index of the sorted list to access, 0 for the lowest, -1 for the highest score
        :param str key: a key from the learning rate file that is used to sort the models
        """
        self.model_dir = model_dir
        self.learning_rates = learning_rates
        self.index = index
        self.out_epoch = self.output_var("epoch")
        self.key = key

        assert isinstance(index, int)

    def run(self):
        # this has to be defined in order for "eval" to work
        def EpochData(learningRate, error):
            return {'learning_rate': learningRate, 'error': error}

        with open(self.learning_rates.get_path(), 'rt') as f:
            text = f.read()

        data = eval(text, {'inf': 1e99, 'EpochData': EpochData})

        epochs = list(sorted(data.keys()))

        if self.key == None:
            dev_score_keys = [k for k in data[epochs[-1]]['error'] if k.startswith('dev_score')]
            dsk = dev_score_keys[0]
        else:
            dsk = self.key

        scores = [(epoch, data[epoch]['error'][dsk]) for epoch in epochs if dsk in data[epoch]['error']]
        sorted_scores = list(sorted(scores, key=lambda x: x[1]))

        self.out_epoch.set(sorted_scores[self.index][0])

    def tasks(self):
        yield Task('run', mini_task=True)


class GetBestCheckpointJob(GetBestEpochJob):
    """

    """

    def __init__(self, model_dir, learning_rates, index=0, key=None):
        """

        :param Path model_dir: model_dir output from a RETURNNTrainingJob
        :param Path learning_rates: learning_rates output from a RETURNNTrainingJob
        :param int index: index of the sorted list to access, 0 for the lowest, -1 for the highest score
        :param str key: a key from the learning rate file that is used to sort the models
        """
        super().__init__(model_dir, learning_rates, index, key)
        self._out_model_dir = self.output_path("model", directory=True)
        self.out_checkpoint = Checkpoint(self.output_path("model/checkpoint.index"))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        super().run()

        shutil.copy(
            os.path.join(self.model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get()),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.index")
        )
        shutil.copy(
            os.path.join(self.model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.meta")
        )
        shutil.copy(
            os.path.join(self.model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get()),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.data-00000-of-00001")
        )


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