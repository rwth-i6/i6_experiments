from i6_core import returnn
from sisyphus import tk

from ... import dataclasses
from ..base import TrainFunctor


class ReturnnTrainFunctor(TrainFunctor[returnn.ReturnnTrainingJob, returnn.ReturnnConfig]):
    def __init__(
        self,
        returnn_root: tk.Path,
        returnn_python_exe: tk.Path,
    ):
        self.returnn_root = returnn_root
        self.returnn_python_exe = returnn_python_exe

    def __call__(
        self, train_config: dataclasses.NamedConfig[returnn.ReturnnConfig], **kwargs
    ) -> returnn.ReturnnTrainingJob:
        train_job = returnn.ReturnnTrainingJob(
            returnn_config=train_config.config,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
            **kwargs,
        )

        train_job.add_alias(f"train_nn/{train_config.name}")
        tk.register_output(f"train_nn/{train_config.name}/learning_rate.png", train_job.out_plot_lr)

        return train_job
