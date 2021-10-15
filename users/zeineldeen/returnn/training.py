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

    If no key is provided, will search for a key prefixed with "dev_score_output", and default to the first key
    starting with "dev_score" otherwise.
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

        error_key = None
        if self.key == None:
            dev_score_keys = [k for k in data[epochs[-1]]['error'] if k.startswith('dev_score')]
            for key in dev_score_keys:
                if key.startswith("dev_score_output"):
                    error_key = key
            if not error_key:
                error_key = dev_score_keys[0]
        else:
            error_key = self.key

        scores = [(epoch, data[epoch]['error'][error_key]) for epoch in epochs if error_key in data[epoch]['error']]
        sorted_scores = list(sorted(scores, key=lambda x: x[1]))

        self.out_epoch.set(sorted_scores[self.index][0])

    def tasks(self):
        yield Task('run', mini_task=True)


class GetBestCheckpointJob(GetBestEpochJob):
    """
    Returns the best checkpoint given a training model dir and a learning-rates file
    The best checkpoint will be HARD-linked, so that no space is wasted but also the model not
    deleted in case that the training folder is removed.



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

        try:
            os.link(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get())
            )
            os.link(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get())
            )
            os.link(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get())
            )
        except OSError:
            # the hardlink will fail when there was an imported job on a different filesystem,
            # thus do a copy instead then
            shutil.copy(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get())
            )
            shutil.copy(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get())
            )
            shutil.copy(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get())
            )

        os.symlink(
            os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get()),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.index")
        )
        os.symlink(
            os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.meta")
        )
        os.symlink(
            os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get()),
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