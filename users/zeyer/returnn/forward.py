"""
More generic forward job
"""

from __future__ import annotations
from typing import Optional, Union, List
import os
import copy
import tempfile
import shutil
import subprocess
from sisyphus import Job, Task
from sisyphus import tk, gs
from i6_core import util
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint


__all__ = ["ReturnnForwardJobV2"]


Checkpoint = Union[TfCheckpoint, PtCheckpoint, tk.Path]


class ReturnnForwardJobV2(Job):
    """
    Generic forward job.

    The user specifies the outputs in the RETURNN config
    via `forward_callback`.
    That is expected to be an instance of `returnn.forward_iface.ForwardCallbackIface`
    or a callable/function which returns such an instance.

    The callback is supposed to generate the output files in the current directory.
    The current directory will be a local temporary directory
    and the files are moved to the output directory at the end.

    Nothing is enforced here by intention, to keep it generic.
    The task by default is set to "forward",
    but other tasks of RETURNN might be used as well.
    """

    def __init__(
        self,
        *,
        model_checkpoint: Optional[Checkpoint],
        returnn_config: ReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        output_files: List[str],
        log_verbosity: int = 5,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
    ):
        """
        :param model_checkpoint: Checkpoint object pointing to a stored RETURNN Tensorflow/PyTorch model
            or None if network has no parameters or should be randomly initialized
        :param returnn_config: RETURNN config object
        :param returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param returnn_root: path to the RETURNN src folder
        :param output_files: list of output file names that will be generated. These are just the basenames,
            and they are supposed to be created in the current directory.
        :param log_verbosity: RETURNN log verbosity
        :param device: RETURNN device, cpu or gpu
        :param time_rqmt: job time requirement in hours
        :param mem_rqmt: job memory requirement in GB
        :param cpu_rqmt: job cpu requirement
        """
        self.returnn_config = returnn_config
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
        config = self.create_returnn_config(
            model_checkpoint=self.model_checkpoint,
            returnn_config=self.returnn_config,
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
            assert os.path.exists(
                _get_model_path(self.model_checkpoint).get_path()
            ), f"Provided model checkpoint does not exists: {self.model_checkpoint}"

    def run(self):
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
                subprocess.check_call(call, cwd=tmp_dir, env=env)
            except Exception:
                print("Run crashed - copy temporary work folder as 'crash_dir'")
                shutil.copytree(tmp_dir, "crash_dir")
                raise

            # move logs
            if os.path.exists(f"{tmp_dir}/returnn.log"):
                shutil.move(f"{tmp_dir}/returnn.log", "returnn.log")
            if os.path.exists(f"{tmp_dir}/returnn-tf-log"):
                shutil.move(f"{tmp_dir}/returnn-tf-log", ".")

            # move outputs to output folder
            for k, v in self.out_files.items():
                assert os.path.exists(f"{tmp_dir}/{k}"), f"Output file {k} does not exist"
                shutil.move(f"{tmp_dir}/{k}", v.get_path())

    @classmethod
    def create_returnn_config(
        cls,
        *,
        model_checkpoint: Optional[Checkpoint],
        returnn_config: ReturnnConfig,
        log_verbosity: int,
        device: str,
        **_kwargs,
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

        returnn_config.post_config.setdefault("device", device)
        returnn_config.post_config.setdefault("log", ["./returnn.log"])
        returnn_config.post_config.setdefault("tf_log_dir", "returnn-tf-log")
        returnn_config.post_config.setdefault("log_verbosity", log_verbosity)

        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config": ReturnnForwardJobV2.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return super().hash(d)


def _get_model_path(model: Checkpoint) -> tk.Path:
    if isinstance(model, tk.Path):
        return model
    if isinstance(model, TfCheckpoint):
        return model.index_path
    if isinstance(model, PtCheckpoint):
        return model.path
    raise TypeError(f"Unknown model checkpoint type: {type(model)}")
