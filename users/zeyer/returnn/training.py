"""
RETURNN training utils
"""

import os
import subprocess
import copy
from sisyphus import gs, tk, Job, Task
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig
import i6_core.util as util


class ReturnnInitModelJob(Job):
    """
    Initialize the model. task="initialize_model" in RETURNN.

    Only returnn_config, returnn_python_exe and returnn_root influence the hash.

    The outputs provided are:

     - out_returnn_config_file: the finalized Returnn config which is used for the rnn.py call
     - out_checkpoint:
    """

    def __init__(
        self,
        returnn_config,
        *,
        log_verbosity=3,
        time_rqmt=1,
        mem_rqmt=4,
        cpu_rqmt=1,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param ReturnnConfig returnn_config:
        :param int log_verbosity: RETURNN log verbosity from 1 (least verbose) to 5 (most verbose)
        :param int|float time_rqmt:
        :param int|float mem_rqmt:
        :param int cpu_rqmt:
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        """
        assert isinstance(returnn_config, ReturnnConfig)
        kwargs = locals()
        del kwargs["self"]

        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else getattr(gs, "RETURNN_PYTHON_EXE")
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else getattr(gs, "RETURNN_ROOT")
        )
        self.returnn_config = self.create_returnn_config(**kwargs)

        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_model_dir = self.output_path("models", directory=True)
        self.out_checkpoint = Checkpoint(index_path=self.output_path("models/init.index"))

        self.returnn_config.post_config["model"] = os.path.join(
            self.out_model_dir.get_path(), "init"
        )

        self.rqmt = {
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def _get_run_cmd(self):
        run_cmd = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        return run_cmd

    def tasks(self):
        """sis job tasks"""
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        """create config"""
        self.returnn_config.write(self.out_returnn_config_file.get_path())
        util.create_executable("rnn.sh", self._get_run_cmd())

    def run(self):
        """run task"""
        subprocess.check_call(self._get_run_cmd())

    @classmethod
    def create_returnn_config(
        cls,
        returnn_config: ReturnnConfig,
        log_verbosity,
        **_kwargs,
    ) -> ReturnnConfig:
        """create derived and adapted config"""
        res = copy.deepcopy(returnn_config)

        config = {
            "task": "initialize_model",
            "target": "classes",
        }

        post_config = {
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
        }

        config.update(copy.deepcopy(returnn_config.config))
        if returnn_config.post_config is not None:
            post_config.update(copy.deepcopy(returnn_config.post_config))

        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        """hash"""
        d = {
            "returnn_config": cls.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }
        return super().hash(d)
