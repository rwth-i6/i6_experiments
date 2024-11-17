__all__ = ["OptunaReturnnTrainingJob", "OptunaReportIntermediateScoreJob", "OptunaReportFinalScoreJob"]

import inspect
import logging
import os
import subprocess as sp
import sys
import time
from typing import Generator, List, Optional, TYPE_CHECKING

from i6_experiments.users.berger.recipe.returnn.training import Backend
from sisyphus import Task, tk, Job

import i6_core.util as util
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint, PtCheckpoint, ReturnnTrainingJob
from .optuna_config import OptunaReturnnConfig

if TYPE_CHECKING:
    import optuna


class OptunaReturnnTrainingJob(Job):
    def __init__(
        self,
        optuna_returnn_config: OptunaReturnnConfig,
        study_name: Optional[str] = None,
        study_storage: Optional[str] = None,
        sampler_seed: int = 42,
        num_trials: int = 15,
        num_parallel: int = 3,
        *,
        backend: Backend = Backend.TENSORFLOW,
        log_verbosity: int = 3,
        device: str = "gpu",
        num_epochs: int = 1,
        save_interval: int = 1,
        keep_epochs: Optional[List[int]] = None,
        time_rqmt: int = 4,
        mem_rqmt: int = 4,
        cpu_rqmt: int = 2,
        gpu_mem_rqmt: int = 11,
        horovod_num_processes: Optional[int] = None,
        multi_node_slots: Optional[int] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ) -> None:
        self.kwargs = locals()
        del self.kwargs["self"]

        self.optuna_returnn_config = optuna_returnn_config

        self.study_name = study_name or "optuna_study"
        self.study_storage = study_storage or "sqlite:///study_storage.db"
        self.sampler_seed = sampler_seed
        self.num_trials = num_trials
        self.num_parallel = num_parallel

        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root
        self.horovod_num_processes = horovod_num_processes
        self.multi_node_slots = multi_node_slots

        self.num_epochs = num_epochs

        stored_epochs = list(range(save_interval, num_epochs, save_interval)) + [num_epochs]
        if keep_epochs is None:
            self.keep_epochs = set(stored_epochs)
        else:
            self.keep_epochs = set(keep_epochs)

        self.out_trial_returnn_config_files = {
            i: self.output_path(f"trial-{i:03d}/returnn.config") for i in range(self.num_trials)
        }
        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_trial_learning_rates = {
            i: self.output_path(f"trial-{i:03d}/learning_rates") for i in range(self.num_trials)
        }
        self.out_learning_rates = self.output_path("learning_rates")
        self.out_trial_model_dir = {
            i: self.output_path(f"trial-{i:03d}/models", directory=True) for i in range(self.num_trials)
        }
        self.out_model_dir = self.output_path("models", directory=True)

        if backend == Backend.TENSORFLOW:
            self.out_trial_checkpoints = {
                i: {
                    k: Checkpoint(self.output_path(f"trial-{i:03d}/models/epoch.{k:03d}.index"))
                    for k in stored_epochs
                    if k in self.keep_epochs
                }
                for i in range(self.num_trials)
            }
        elif backend == Backend.PYTORCH:
            self.out_trial_checkpoints = {
                i: {
                    k: PtCheckpoint(self.output_path(f"trial-{i:03d}/models/epoch.{k:03d}.pt"))
                    for k in stored_epochs
                    if k in self.keep_epochs
                }
                for i in range(self.num_trials)
            }
        else:
            raise NotImplementedError

        self.out_task_id_to_trial_num = {i: self.output_var(f"task-{i:03d}-trial") for i in range(self.num_trials)}
        self.out_trials = {i: self.output_var(f"trial-{i:03d}/trial", pickle=True) for i in range(self.num_trials)}
        self.out_trial_params = {i: self.output_var(f"trial-{i:03d}/params") for i in range(self.num_trials)}

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
            "gpu_mem": gpu_mem_rqmt,
        }

        if self.multi_node_slots:
            assert self.horovod_num_processes, "multi_node_slots only supported together with Horovod currently"
            assert self.horovod_num_processes >= self.multi_node_slots
            assert self.horovod_num_processes % self.multi_node_slots == 0
            self.rqmt["multi_node_slots"] = self.multi_node_slots

        slots = self.multi_node_slots or 1
        if self.horovod_num_processes and self.horovod_num_processes > slots:
            assert self.horovod_num_processes % slots == 0
            self.rqmt["cpu"] *= self.horovod_num_processes // slots
            self.rqmt["gpu"] *= self.horovod_num_processes // slots
            self.rqmt["mem"] *= self.horovod_num_processes // slots

    # ------------------ Helpers ------------------

    def _get_run_cmd(self, config_file: tk.Path) -> List[str]:
        run_cmd = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            config_file.get_path(),
        ]

        if self.horovod_num_processes:
            # Normally, if the engine (e.g. SGE or Slurm) is configured correctly,
            # it automatically provides the information on multiple nodes to mpirun,
            # so it is not needed to explicitly pass on any hostnames here.
            run_cmd = [
                "mpirun",
                "-np",
                str(self.horovod_num_processes),
                "-bind-to",
                "none",
                "-map-by",
                "slot",
                "-mca",
                "pml",
                "ob1",
                "-mca",
                "btl",
                "^openib",
                "--report-bindings",
            ] + run_cmd

        return run_cmd

    def path_available(self, path: tk.Path) -> bool:
        # if job is finished the path is available
        if super().path_available(path):
            return True

        # learning rate files are only available at the end
        if path == self.out_learning_rates:
            return super().path_available(path)

        # maybe the file already exists
        if os.path.exists(path.get_path()):
            return True

        # maybe the model is just a pretrain model
        file = os.path.basename(path.get_path())
        directory = os.path.dirname(path.get_path())
        if file.startswith("epoch."):
            segments = file.split(".")
            pretrain_file = ".".join([segments[0], "pretrain", segments[1]])
            pretrain_path = os.path.join(directory, pretrain_file)
            return os.path.exists(pretrain_path)

        return False

    def get_returnn_config(self, trial: "optuna.Trial") -> ReturnnConfig:
        returnn_config = self.optuna_returnn_config.generate_config(trial)
        trial_num = trial.number
        returnn_config.post_config["model"] = os.path.join(self.out_trial_model_dir[trial_num].get_path(), "epoch")
        returnn_config.post_config.pop("learning_rate_file", None)
        returnn_config.config["learning_rate_file"] = f"trial-{trial_num:03d}/learning_rates"

        returnn_config.post_config["log"] = f"./trial-{trial_num:03d}/returnn.log"

        ReturnnTrainingJob.check_blacklisted_parameters(returnn_config)
        returnn_config = ReturnnTrainingJob.create_returnn_config(returnn_config, **self.kwargs)

        return returnn_config

    def prepare_trial_files(self, returnn_config: ReturnnConfig, trial_num: int) -> None:
        config_file = self.out_trial_returnn_config_files[trial_num]
        returnn_config.write(config_file.get_path())
        os.mkdir(f"trial-{trial_num:03d}")
        util.create_executable(f"trial-{trial_num:03d}/rnn.sh", self._get_run_cmd(config_file))

    def prepare_env(self) -> None:
        if not self.multi_node_slots:
            return
        # Some useful debugging, specifically for SGE parallel environment (PE).
        if "PE_HOSTFILE" in os.environ:
            print("PE_HOSTFILE =", os.environ["PE_HOSTFILE"])
            if os.environ["PE_HOSTFILE"]:
                try:
                    print("Content:")
                    with open(os.environ["PE_HOSTFILE"]) as f:
                        print(f.read())
                except Exception as exc:
                    print("Cannot read:", exc)
        sys.stdout.flush()

    def parse_lr_file(self, trial_num: int) -> dict:
        def EpochData(learningRate, error):
            return {"learning_rate": learningRate, "error": error}

        filename = f"trial-{trial_num:03d}/learning_rates"
        with open(filename, "rt") as f:
            lr_text = f.read()
        return eval(lr_text)

    # ------------------ Tasks ------------------

    def tasks(self) -> Generator[Task, None, None]:
        yield Task("create_study", mini_task=True)
        yield Task(
            "run",
            resume="run",
            rqmt=self.rqmt,
            parallel=self.num_parallel,
            args=range(self.num_trials),
        )

    def create_study(self) -> None:
        import optuna

        optuna.create_study(
            study_name=self.study_name,
            storage=self.study_storage,
            sampler=optuna.samplers.TPESampler(n_startup_trials=max(5, self.num_parallel), seed=self.sampler_seed),
            direction="minimize",
            load_if_exists=True,
        )

    @staticmethod
    def _check_trial_finished(study: "optuna.Study", trial_num: int) -> bool:
        import optuna

        for frozen_trial in study.get_trials(
            states=[
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.PRUNED,
            ]
        ):
            if frozen_trial.number != trial_num:
                continue
            logging.info(f"Trial has already finished with state {frozen_trial.state}")
            return True
        return False

    def run(self, task_id: int) -> None:
        import optuna

        storage = optuna.storages.get_storage(self.study_storage)
        study = optuna.load_study(
            study_name=self.study_name,
            storage=storage,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=max(5, self.num_parallel),
                n_warmup_steps=self.num_epochs // 2,
            ),
        )

        if self.out_task_id_to_trial_num[task_id].is_set():
            trial_num = int(self.out_task_id_to_trial_num[task_id].get())
            logging.info(f"Found existing trial with number {trial_num}")

            if self._check_trial_finished(study, trial_num):
                return

            # Retreive the trial id for reporting purposes
            # Returnn config should already exist and does not need to be created again
            study_id = storage.get_study_id_from_name(self.study_name)
            trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, trial_num)
            trial = optuna.Trial(study, trial_id)

        else:
            trial = study.ask()
            trial_num = trial.number
            self.out_task_id_to_trial_num[task_id].set(trial_num)
            logging.info(f"Start new trial with number {trial_num}")
            returnn_config = self.get_returnn_config(trial)
            self.prepare_trial_files(returnn_config, trial_num)
            self.out_trial_params[trial_num].set(trial.params)
            self.out_trials[trial_num].set(optuna.trial.FixedTrial(trial.params, trial_num))

        config_file = self.out_trial_returnn_config_files[trial_num]

        run_cmd = self._get_run_cmd(config_file)
        training_process = sp.Popen(run_cmd)

        trial_pruned = False
        while training_process.poll() is None:
            time.sleep(30)

            if trial.should_prune():
                trial_pruned = True
                training_process.terminate()
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                break

        if trial_pruned:
            logging.info("Pruned trial run")
            os.link(
                f"trial-{trial_num:03d}/learning_rates",
                self.out_trial_learning_rates[trial_num].get_path(),
            )

        lr_data = self.parse_lr_file(trial_num)
        max_epoch = max([ep for ep, ep_data in lr_data.items() if ep_data["error"] != {}])

        if not trial_pruned and max_epoch == self.num_epochs:
            logging.info("Finished trial run normally")
            os.link(
                f"trial-{trial_num:03d}/learning_rates",
                self.out_trial_learning_rates[trial_num].get_path(),
            )

        if not trial_pruned and max_epoch != self.num_epochs:
            logging.info("Training had an error")
            raise sp.CalledProcessError(-1, cmd=run_cmd)

    @classmethod
    def hash(cls, kwargs):
        d = {
            "optuna_returnn_config": kwargs["optuna_returnn_config"],
            "sampler_seed": kwargs["sampler_seed"],
            "num_trials": kwargs["num_trials"],
            "num_parallel": kwargs["num_parallel"],
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        if kwargs["study_name"] is not None:
            d["study_name"] = kwargs["study_name"]
        if kwargs["study_storage"] is not None:
            d["study_storage"] = kwargs["study_storage"]
        if kwargs["horovod_num_processes"] is not None:
            d["horovod_num_processes"] = kwargs["horovod_num_processes"]
        if kwargs["multi_node_slots"] is not None:
            d["multi_node_slots"] = kwargs["multi_node_slots"]

        return super().hash(d)


class OptunaReportIntermediateScoreJob(Job):
    def __init__(
        self,
        trial_num: int,
        step: int,
        score: tk.Variable,
        study_name: Optional[str] = None,
        study_storage: Optional[str] = None,
    ) -> None:
        self.study_name = study_name or "optuna_study"
        self.study_storage = study_storage or "sqlite:///study_storage.db"
        self.trial_num = trial_num
        self.step = step
        self.score = score

        self.out_reported_score = self.output_var("reported_score")

    def tasks(self) -> Generator[Task, None, None]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        import optuna

        storage = optuna.storages.get_storage(self.study_storage)
        study = optuna.load_study(
            study_name=self.study_name,
            storage=storage,
        )

        study_id = storage.get_study_id_from_name(self.study_name)
        trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, self.trial_num)
        trial = optuna.Trial(study, trial_id)

        self.out_reported_score.set(self.score.get())

        for frozen_trial in study.get_trials(
            states=[
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.PRUNED,
            ]
        ):
            if frozen_trial.number == self.trial_num:
                logging.info(f"Trial has already finished with state {frozen_trial.state}")
                return

        trial.report(value=self.score.get(), step=self.step)

    @classmethod
    def hash(cls, kwargs):
        d = {
            "trial_num": kwargs["trial_num"],
            "step": kwargs["step"],
            "score": kwargs["score"],
        }

        if kwargs["study_name"] is not None:
            d["study_name"] = kwargs["study_name"]
        if kwargs["study_storage"] is not None:
            d["study_storage"] = kwargs["study_storage"]

        return super().hash(d)


class OptunaReportFinalScoreJob(Job):
    def __init__(
        self,
        trial_num: int,
        scores: List[tk.Variable],
        study_name: Optional[str] = None,
        study_storage: Optional[str] = None,
    ) -> None:
        self.study_name = study_name or "optuna_study"
        self.study_storage = study_storage or "sqlite:///study_storage.db"
        self.trial_num = trial_num
        self.scores = scores

        self.out_reported_score = self.output_var("reported_score")

    def tasks(self) -> Generator[Task, None, None]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        import optuna

        storage = optuna.storages.get_storage(self.study_storage)
        study = optuna.load_study(
            study_name=self.study_name,
            storage=storage,
        )

        best_score = min([score.get() for score in self.scores])
        self.out_reported_score.set(best_score)

        for frozen_trial in study.get_trials(
            states=[
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.PRUNED,
            ]
        ):
            if frozen_trial.number == self.trial_num:
                logging.info(f"Trial has already finished with state {frozen_trial.state}")
                return

        study.tell(trial=self.trial_num, values=best_score, state=optuna.trial.TrialState.COMPLETE)

    @classmethod
    def hash(cls, kwargs):
        d = {
            "trial_num": kwargs["trial_num"],
            "scores": kwargs["scores"],
        }

        if kwargs["study_name"] is not None:
            d["study_name"] = kwargs["study_name"]
        if kwargs["study_storage"] is not None:
            d["study_storage"] = kwargs["study_storage"]

        return super().hash(d)
