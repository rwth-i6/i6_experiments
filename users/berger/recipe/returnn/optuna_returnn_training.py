import glob
import os
import time
import subprocess as sp
import sys
from typing import Generator, List, Optional
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint, ReturnnTrainingJob
import i6_core.util as util
from sisyphus import Task, tk, Job
from typing import Callable
import optuna


class OptunaReturnnTrainingJob(Job):
    def __init__(
        self,
        returnn_config_generator: Callable[..., ReturnnConfig],
        returnn_config_generator_kwargs: dict = {},
        study_name: Optional[str] = None,
        study_storage: Optional[str] = None,
        sampler_seed: int = 42,
        score_key: str = "dev_score",
        num_trials: int = 15,
        num_parallel: int = 3,
        *,
        log_verbosity: int = 3,
        device: str = "gpu",
        num_epochs: int = 1,
        save_interval: int = 1,
        keep_epochs: Optional[List[int]] = None,
        time_rqmt: int = 4,
        mem_rqmt: int = 4,
        cpu_rqmt: int = 2,
        horovod_num_processes: Optional[int] = None,
        multi_node_slots: Optional[int] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ) -> None:
        self.kwargs = locals()
        del self.kwargs["self"]

        self.returnn_config_generator = returnn_config_generator
        self.returnn_config_generator_kwargs = returnn_config_generator_kwargs

        self.study_name = study_name or "optuna_study"
        self.study_storage = study_storage or f"sqlite:///study_storage.db"
        self.sampler_seed = sampler_seed
        self.score_key = score_key
        self.num_trials = num_trials
        self.num_parallel = num_parallel

        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root
        self.horovod_num_processes = horovod_num_processes
        self.multi_node_slots = multi_node_slots

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
        self.out_trial_checkpoints = {
            i: {
                k: Checkpoint(self.output_path(f"trial-{i:03d}/models/epoch.{k:03d}.index"))
                for k in stored_epochs
                if k in self.keep_epochs
            }
            for i in range(self.num_trials)
        }
        self.out_checkpoints = {
            k: Checkpoint(self.output_path(f"models/epoch.{k:03d}.index"))
            for k in stored_epochs
            if k in self.keep_epochs
        }

        self.out_trial_nums = {i: self.output_var(f"trial-{i:03d}/trial_num") for i in range(self.num_trials)}
        self.out_trials = {i: self.output_var(f"trial-{i:03d}/trial", pickle=True) for i in range(self.num_trials)}
        self.out_trial_params = {i: self.output_var(f"trial-{i:03d}/params") for i in range(self.num_trials)}
        self.out_trial_scores = {i: self.output_var(f"trial-{i:03d}/score") for i in range(self.num_trials)}
        self.out_best_trial = self.output_var("best_trial", pickle=True)
        self.out_best_params = self.output_var("best_params")
        self.out_best_score = self.output_var("best_score")

        self.out_plot_se = self.output_path(f"score_and_error.png")
        self.out_plot_lr = self.output_path(f"learning_rate.png")

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
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

    def get_returnn_config(self, trial: optuna.trial.BaseTrial, task_id: int) -> ReturnnConfig:
        returnn_config = self.returnn_config_generator(trial, **self.returnn_config_generator_kwargs)
        returnn_config.post_config["model"] = os.path.join(self.out_trial_model_dir[task_id].get_path(), "epoch")
        returnn_config.post_config.pop("learning_rate_file", None)
        returnn_config.config["learning_rate_file"] = f"trial-{task_id:03d}/learning_rates"

        returnn_config.post_config["log"] = f"./trial-{task_id:03d}/returnn.log"

        ReturnnTrainingJob.check_blacklisted_parameters(returnn_config)
        returnn_config = ReturnnTrainingJob.create_returnn_config(returnn_config, **self.kwargs)

        return returnn_config

    def prepare_trial_files(self, returnn_config: ReturnnConfig, task_id: int) -> None:
        config_file = self.out_trial_returnn_config_files[task_id]
        returnn_config.write(config_file.get_path())
        os.mkdir(f"trial-{task_id:03d}")
        util.create_executable(f"trial-{task_id:03d}/rnn.sh", self._get_run_cmd(config_file))

        # Additional import packages that are created by returnn common
        for f in glob.glob("../output/*"):
            f_name = os.path.basename(f)
            if f_name.startswith("trial-"):
                continue
            if f_name == "models":
                continue
            os.symlink(f"../{f_name}", f"../output/trial-{task_id:03d}/{f_name}")

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

    def parse_lr_file(self, task_id: Optional[int] = None) -> dict:
        def EpochData(learningRate, error):
            return {"learning_rate": learningRate, "error": error}

        if task_id is None:
            filename = self.out_learning_rates
        else:
            filename = f"trial-{task_id:03d}/learning_rates"
        with open(filename, "rt") as f:
            lr_text = f.read()
        return eval(lr_text)

    def link_to_final_output(self, task_id: int) -> None:
        os.link(
            self.out_trial_returnn_config_files[task_id],
            self.out_returnn_config_file,
        )
        os.link(self.out_trial_learning_rates[task_id], self.out_learning_rates)
        for k in self.out_checkpoints:
            for suffix in ["index", "meta", "data-00000-of-00001"]:
                orig_file = f"{self.out_trial_checkpoints[task_id][k]}.{suffix}"
                if not os.path.exists(orig_file):
                    continue
                os.link(orig_file, f"{self.out_checkpoints[k]}.{suffix}")

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
        yield Task("select_best_trial", mini_task=True)
        yield Task("plot", resume="plot", mini_task=True)

    def create_study(self) -> None:
        optuna.create_study(
            study_name=self.study_name,
            storage=self.study_storage,
            sampler=optuna.samplers.TPESampler(n_startup_trials=max(5, self.num_parallel), seed=self.sampler_seed),
            direction="minimize",
            load_if_exists=True,
        )

    def run(self, task_id: int) -> None:
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.study_storage,
        )

        study = optuna.load_study(study_name=self.study_name, storage=self.study_storage)
        if self.out_trials[task_id].is_set():
            trial = self.out_trials[task_id].get()
            trial_num = self.out_trial_nums[task_id].get()
            print(f"Found existing trial with number {trial_num}")
            assert isinstance(trial, optuna.trial.FixedTrial)
            assert isinstance(trial_num, int)
            returnn_config = self.get_returnn_config(trial, trial_num)
        else:
            trial = study.ask()
            trial_num = trial.number
            print(f"Start new trial with number {trial_num}")
            returnn_config = self.get_returnn_config(trial, trial_num)
            self.prepare_trial_files(returnn_config, trial_num)
            self.out_trial_nums[task_id].set(trial_num)
            self.out_trial_params[trial_num].set(trial.params)
            self.out_trials[trial_num].set(optuna.trial.FixedTrial(trial.params, trial_num))

        config_file = self.out_trial_returnn_config_files[trial_num]

        run_cmd = self._get_run_cmd(config_file)
        training_process = sp.Popen(run_cmd)

        max_epoch = 0
        best_score = float("inf")
        trial_pruned = False
        while return_code := training_process.poll() is None:
            time.sleep(30)
            try:
                lr_data = self.parse_lr_file(trial_num)
            except (FileNotFoundError, SyntaxError):
                continue
            epochs = list(sorted(lr_data.keys()))
            new_epochs = [e for e in epochs if e > max_epoch]
            for e in new_epochs:
                if self.score_key not in lr_data[e]["error"]:
                    continue
                max_epoch = e
                score = lr_data[e]["error"][self.score_key]
                if score < best_score:
                    best_score = score

                trial.report(score, e)

            if trial.should_prune():
                trial_pruned = True
                training_process.terminate()
                study.tell(trial_num, state=optuna.trial.TrialState.PRUNED)
                break

        if trial_pruned or return_code == 0:
            self.out_trial_scores[trial_num].set(best_score)
            os.link(
                f"trial-{trial_num:03d}/learning_rates",
                self.out_trial_learning_rates[trial_num].get_path(),
            )

        if not trial_pruned and return_code == 0:
            assert max_epoch == returnn_config.config["num_epochs"]
            study.tell(trial_num, best_score, state=optuna.trial.TrialState.COMPLETE)

        if not trial_pruned and return_code != 0:
            raise sp.CalledProcessError(return_code, cmd=run_cmd)

    def select_best_trial(self) -> None:
        study = optuna.load_study(study_name=self.study_name, storage=self.study_storage)
        self.out_best_params.set(study.best_params)
        self.out_best_trial.set(study.best_trial)
        self.out_best_score.set(study.best_value)
        for task_id, trial_num in self.out_trial_nums.items():
            if trial_num.get() == study.best_trial.number:
                self.link_to_final_output(task_id=task_id)
                break

    def plot(self):
        data = self.parse_lr_file()

        epochs = list(sorted(data.keys()))
        train_score_keys = [k for k in data[epochs[0]]["error"] if k.startswith("train_score")]
        dev_score_keys = [k for k in data[epochs[0]]["error"] if k.startswith("dev_score")]
        dev_error_keys = [k for k in data[epochs[0]]["error"] if k.startswith("dev_error")]

        train_scores = [
            [(epoch, data[epoch]["error"][tsk]) for epoch in epochs if tsk in data[epoch]["error"]]
            for tsk in train_score_keys
        ]
        dev_scores = [
            [(epoch, data[epoch]["error"][dsk]) for epoch in epochs if dsk in data[epoch]["error"]]
            for dsk in dev_score_keys
        ]
        dev_errors = [
            [(epoch, data[epoch]["error"][dek]) for epoch in epochs if dek in data[epoch]["error"]]
            for dek in dev_error_keys
        ]
        learing_rates = [data[epoch]["learning_rate"] for epoch in epochs]

        colors = ["#2A4D6E", "#AA3C39", "#93A537"]  # blue red yellowgreen

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        for ts in train_scores:
            ax1.plot([d[0] for d in ts], [d[1] for d in ts], "o-", color=colors[0])
        for ds in dev_scores:
            ax1.plot([d[0] for d in ds], [d[1] for d in ds], "o-", color=colors[1])
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("scores", color=colors[0])
        for tl in ax1.get_yticklabels():
            tl.set_color(colors[0])

        if len(dev_errors) > 0 and any(len(de) > 0 for de in dev_errors):
            ax2 = ax1.twinx()
            ax2.set_ylabel("dev error", color=colors[2])
            for de in dev_errors:
                ax2.plot([d[0] for d in de], [d[1] for d in de], "o-", color=colors[2])
            for tl in ax2.get_yticklabels():
                tl.set_color(colors[2])

        fig.savefig(fname=self.out_plot_se.get_path())

        fig, ax1 = plt.subplots()
        ax1.semilogy(epochs, learing_rates, "ro-")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("learning_rate")

        fig.savefig(fname=self.out_plot_lr.get_path())

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config_generator": kwargs["returnn_config_generator"],
            "returnn_config_generator_kwargs": kwargs["returnn_config_generator_kwargs"],
            "sampler_seed": kwargs["sampler_seed"],
            "score_key": kwargs["score_key"],
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
