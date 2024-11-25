from i6_experiments.users.berger.recipe import returnn
from sisyphus import tk

from ... import dataclasses
from ..base import TrainFunctor


class OptunaReturnnTrainFunctor(TrainFunctor[returnn.OptunaReturnnTrainingJob, returnn.OptunaReturnnConfig]):
    def __init__(
        self,
        returnn_root: tk.Path,
        returnn_python_exe: tk.Path,
    ):
        self.returnn_root = returnn_root
        self.returnn_python_exe = returnn_python_exe

    def __call__(
        self,
        train_config: dataclasses.NamedConfig[returnn.OptunaReturnnConfig],
        **kwargs,
    ) -> returnn.OptunaReturnnTrainingJob:
        train_job = returnn.OptunaReturnnTrainingJob(
            optuna_returnn_config=train_config.config,
            study_name=train_config.name,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
            **kwargs,
        )

        train_job.add_alias(f"train_nn/{train_config.name}")
        for trial_num, learning_rate_file in train_job.out_trial_learning_rates.items():
            tk.register_output(f"train_nn/{train_config.name}/trial-{trial_num:03d}/learning_rates", learning_rate_file)

        return train_job
