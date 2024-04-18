import copy
import os
import subprocess as sp
from typing import Optional, Union

from i6_core import util
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint, PtCheckpoint
from sisyphus import Job, Task, tk


class ReturnnForwardComputePriorJob(Job):
    def __init__(
        self,
        model_checkpoint: Optional[Union[Checkpoint, PtCheckpoint]],
        returnn_config: ReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        *,  # args below are keyword only
        log_verbosity: int = 5,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
    ):
        self.returnn_config = returnn_config
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
        **kwargs,
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
            "returnn_config": ReturnnForwardComputePriorJob.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return super().hash(d)
