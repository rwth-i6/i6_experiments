import copy
import inspect
import os
import shutil
import subprocess as sp
import tempfile
from typing import List, Optional, Union

from i6_core import util
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint, PtCheckpoint
from i6_experiments.users.berger.recipe.returnn.optuna_config import OptunaReturnnConfig
from sisyphus import Job, Task, tk, gs


class OptunaReturnnForwardComputePriorJob(Job):
    def __init__(
        self,
        model_checkpoint: Optional[Union[Checkpoint, PtCheckpoint]],
        trial: tk.Variable,
        optuna_returnn_config: OptunaReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        *,  # args below are keyword only
        log_verbosity: int = 5,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
    ):
        self.optuna_returnn_config = optuna_returnn_config
        self.trial = trial
        self.model_checkpoint = model_checkpoint
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root
        self.log_verbosity = log_verbosity
        self.device = device

        self.out_returnn_config_file = self.output_path("returnn.config")

        self.out_prior_txt_file = self.output_path("prior.txt")
        self.out_prior_xml_file = self.output_path("prior.xml")
        self.out_prior_png_file = self.output_path("prior.png")

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        config = self.create_returnn_config(
            model_checkpoint=self.model_checkpoint,
            returnn_config=self.optuna_returnn_config.generate_config(self.trial.get()),  # type: ignore
            log_verbosity=self.log_verbosity,
            device=self.device,
        )
        config.write(self.out_returnn_config_file.get_path())

        cmd = [
            self.returnn_python_exe.get_path(),
            os.path.join(self.returnn_root.get_path(), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        if self.model_checkpoint is not None:
            assert self.model_checkpoint.exists(), "Provided model does not exists: %s" % str(self.model_checkpoint)

    def run(self):
        sp.check_call(
            [
                self.returnn_python_exe.get_path(),
                self.returnn_root.join_right("rnn.py").get_path(),
                self.out_returnn_config_file.get_path(),
            ]
        )

    @classmethod
    def create_returnn_config(
        cls,
        model_checkpoint: Optional[Union[Checkpoint, PtCheckpoint]],
        returnn_config: ReturnnConfig,
        log_verbosity: int,
        device: str,
        **_,
    ):
        assert device in ["gpu", "cpu"]
        assert "task" not in returnn_config.config
        assert "load" not in returnn_config.config
        assert "model" not in returnn_config.config

        res = copy.deepcopy(returnn_config)

        config = {"load": model_checkpoint, "task": "forward", "forward_data": "train"}

        post_config = {
            "device": device,
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
        }

        config.update(returnn_config.config)
        post_config.update(returnn_config.post_config)

        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config_generator": inspect.getsource(kwargs["optuna_returnn_config"].config_generator),
            "returnn_config_generator_kwargs": list(sorted(kwargs["optuna_returnn_config"].config_kwargs)),
            "model_checkpoint": kwargs["model_checkpoint"],
            "trial": kwargs["trial"],
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return super().hash(d)


class OptunaReturnnForwardJob(Job):
    def __init__(
        self,
        *,
        model_checkpoint: Optional[Union[Checkpoint, PtCheckpoint]],
        trial: tk.Variable,
        optuna_returnn_config: OptunaReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        output_files: List[str],
        log_verbosity: int = 5,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
    ):
        self.optuna_returnn_config = optuna_returnn_config
        self.trial = trial
        self.model_checkpoint = model_checkpoint
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root
        self.log_verbosity = log_verbosity
        self.device = device

        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_files = {output: self.output_path(output) for output in output_files}

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        """create files"""
        config = self.create_returnn_config(
            model_checkpoint=self.model_checkpoint,
            returnn_config=self.optuna_returnn_config.generate_config(self.trial.get()),  # type: ignore
            log_verbosity=self.log_verbosity,
            device=self.device,
        )
        config.write(self.out_returnn_config_file.get_path())

        cmd = [
            self.returnn_python_exe.get_path(),
            os.path.join(self.returnn_root.get_path(), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        if self.model_checkpoint is not None:
            assert self.model_checkpoint.exists(), f"Provided model checkpoint does not exists: {self.model_checkpoint}"

    def run(self):
        """run"""
        # run everything in a TempDir as writing files can cause heavy load
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
            print("using temp-dir: %s" % tmp_dir)
            call = [
                self.returnn_python_exe.get_path(),
                os.path.join(self.returnn_root.get_path(), "rnn.py"),
                self.out_returnn_config_file.get_path(),
            ]

            try:
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(self.rqmt["cpu"])
                env["MKL_NUM_THREADS"] = str(self.rqmt["cpu"])
                sp.check_call(call, cwd=tmp_dir, env=env)
            except Exception:
                print("Run crashed - copy temporary work folder as 'crash_dir'")
                if os.path.exists("crash_dir"):
                    shutil.rmtree("crash_dir")
                shutil.copytree(tmp_dir, "crash_dir", dirs_exist_ok=True)
                raise

            # move outputs to output folder
            for k, v in self.out_files.items():
                assert os.path.exists(f"{tmp_dir}/{k}"), f"Output file {k} does not exist"
                shutil.move(f"{tmp_dir}/{k}", v.get_path())

            # copy logs and anything else. don't make assumptions on filenames
            shutil.copytree(tmp_dir, ".", dirs_exist_ok=True)

    @classmethod
    def create_returnn_config(
        cls,
        *,
        model_checkpoint: Optional[Union[Checkpoint, PtCheckpoint]],
        returnn_config: ReturnnConfig,
        log_verbosity: int,
        device: str,
        **_,
    ):
        """
        Update the config locally to make it ready for the forward/eval task.
        The resulting config will be used for hashing.

        :param model_checkpoint:
        :param returnn_config:
        :param log_verbosity:
        :param device:
        :return:
        """
        assert "load" not in returnn_config.config
        assert "model" not in returnn_config.config

        res = copy.deepcopy(returnn_config)

        res.config.setdefault("task", "forward")
        if model_checkpoint is not None:
            res.config["load"] = model_checkpoint
        else:
            res.config.setdefault("allow_random_model_init", True)

        res.post_config.setdefault("device", device)
        res.post_config.setdefault("log", ["./returnn.log"])
        res.post_config.setdefault("tf_log_dir", "returnn-tf-log")
        res.post_config.setdefault("log_verbosity", log_verbosity)

        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = {
            "optuna_returnn_config": kwargs["optuna_returnn_config"],
            "model_checkpoint": kwargs["model_checkpoint"],
            "trial": kwargs["trial"],
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return super().hash(d)
