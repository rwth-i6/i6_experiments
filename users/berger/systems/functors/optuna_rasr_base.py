from abc import ABC
from typing import Union

from i6_core.returnn import Checkpoint
from i6_experiments.users.berger.recipe import returnn
from i6_experiments.users.berger.recipe.returnn.training import Backend
from i6_experiments.users.berger.util import lru_cache_with_signature
from sisyphus import tk

from .. import types
from .rasr_base import RasrFunctor


class OptunaRasrFunctor(RasrFunctor, ABC):
    @lru_cache_with_signature
    def _get_epoch_value(
        self,
        train_job: returnn.OptunaReturnnTrainingJob,
        epoch: types.EpochType,
        trial_num: types.TrialType,
    ) -> Union[int, tk.Variable]:
        if epoch != "best":
            return epoch

        if trial_num == "best":
            lr = train_job.out_learning_rates
        else:
            lr = train_job.out_trial_learning_rates[trial_num]
        return returnn.GetBestEpochJob(lr).out_epoch

    def _make_tf_graph(
        self,
        train_job: returnn.OptunaReturnnTrainingJob,
        returnn_config: returnn.OptunaReturnnConfig,
        epoch: types.EpochType,
        label_scorer_type: str = "precomputed-log-posterior",
        trial_num: types.TrialType = "best",
    ) -> tk.Path:
        if trial_num == "best":
            trial = train_job.out_best_trial
        else:
            trial = train_job.out_trials[trial_num]
        rec_step_by_step = "output" if self._is_autoregressive_decoding(label_scorer_type) else None
        graph_compile_job = returnn.OptunaCompileTFGraphJob(
            returnn_config,
            trial=trial,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            epoch=self._get_epoch_value(train_job, epoch, trial_num),
            rec_step_by_step=rec_step_by_step,
            rec_json_info=bool(rec_step_by_step),
        )
        return graph_compile_job.out_graph

    def _make_onnx_model(
        self,
        train_job: returnn.OptunaReturnnTrainingJob,
        returnn_config: returnn.OptunaReturnnConfig,
        checkpoint: returnn.PtCheckpoint,
        trial_num: types.TrialType = "best",
    ) -> tk.Path:
        if trial_num == "best":
            trial = train_job.out_best_trial
        else:
            trial = train_job.out_trials[trial_num]

        onnx_export_job = returnn.OptunaExportPyTorchModelToOnnxJob(
            pytorch_checkpoint=checkpoint,
            returnn_config=returnn_config,
            trial=trial,
            returnn_root=self.returnn_root,
        )
        return onnx_export_job.out_onnx_model

    @lru_cache_with_signature
    def _get_checkpoint(
        self,
        train_job: returnn.OptunaReturnnTrainingJob,
        epoch: types.EpochType,
        trial_num: types.TrialType = "best",
        backend: Backend = Backend.TENSORFLOW,
    ) -> Checkpoint:
        if trial_num == "best":
            if epoch == "best":
                return returnn.GetBestCheckpointJob(
                    model_dir=train_job.out_model_dir,
                    learning_rates=train_job.out_learning_rates,
                    backend=backend,
                ).out_checkpoint
            return train_job.out_checkpoints[epoch]

        if epoch == "best":
            return returnn.GetBestCheckpointJob(
                model_dir=train_job.out_trial_model_dir[trial_num],
                learning_rates=train_job.out_trial_learning_rates[trial_num],
                backend=backend,
            ).out_checkpoint
        return train_job.out_trial_checkpoints[trial_num][epoch]

    def _get_prior_file(
        self,
        train_job: returnn.OptunaReturnnTrainingJob,
        prior_config: returnn.OptunaReturnnConfig,
        checkpoint: Checkpoint,
        trial_num: types.TrialType = "best",
        backend: Backend = Backend.TENSORFLOW,
        **kwargs,
    ) -> tk.Path:
        if trial_num == "best":
            trial = train_job.out_best_trial
        else:
            trial = train_job.out_trials[trial_num]
        if backend == backend.TENSORFLOW:
            prior_job = returnn.OptunaReturnnComputePriorJob(
                model_checkpoint=checkpoint,
                trial=trial,
                optuna_returnn_config=prior_config,
                returnn_python_exe=self.returnn_python_exe,
                returnn_root=self.returnn_root,
                **kwargs,
            )

            prior_job.update_rqmt("run", {"file_size": 150})

            return prior_job.out_prior_xml_file
        elif backend == backend.PYTORCH:
            forward_job = returnn.OptunaReturnnForwardComputePriorJob(
                model_checkpoint=checkpoint,
                trial=trial,
                optuna_returnn_config=prior_config,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
            )
            return forward_job.out_prior_xml_file
        else:
            raise NotImplementedError

    @lru_cache_with_signature
    def _get_trial_value(
        self,
        train_job: returnn.OptunaReturnnTrainingJob,
        trial_num: types.TrialType = "best",
    ) -> Union[int, tk.Variable]:
        if trial_num != "best":
            return trial_num

        return train_job.out_best_trial_num
