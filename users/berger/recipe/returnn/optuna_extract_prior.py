import copy
import os
import i6_core.util as util
import numpy as np
import subprocess as sp
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig
from typing import Callable, Optional, Dict, Any
from sisyphus import Task, tk, Job


class OptunaReturnnComputePriorJob(Job):
    """
    Given a model checkpoint, run compute_prior task with RETURNN
    """

    def __init__(
        self,
        model_checkpoint: Checkpoint,
        trial: tk.Variable,
        returnn_config_generator: Callable,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        returnn_config_generator_kwargs: Dict = {},
        prior_data: Optional[Dict[str, Any]] = None,
        *,
        log_verbosity: int = 3,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
    ):
        """
        :param model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param returnn_config: object representing RETURNN config
        :param prior_data: dataset used to compute prior (None = use one train epoch)
        :param log_verbosity: RETURNN log verbosity
        :param device: RETURNN device, cpu or gpu
        :param time_rqmt: job time requirement in hours
        :param mem_rqmt: job memory requirement in GB
        :param cpu_rqmt: job cpu requirement in GB
        :param returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param returnn_root: path to the RETURNN src folder
        """
        kwargs = locals()
        del kwargs["self"]
        self.returnn_config_kwargs = kwargs

        self.model_checkpoint = model_checkpoint

        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root

        self.trial = trial

        self.returnn_config_generator = returnn_config_generator
        self.returnn_config_generator_kwargs = returnn_config_generator_kwargs

        self.prior_data = prior_data

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
        yield Task("plot", resume="plot", mini_task=True)

    def create_files(self):
        self.returnn_config = self.returnn_config_generator(
            self.trial.get(), **self.returnn_config_generator_kwargs
        )
        self.returnn_config = self.create_returnn_config(
            returnn_config=self.returnn_config, **self.returnn_config_kwargs
        )
        self.returnn_config.post_config["output_file"] = self.out_prior_txt_file
        self.returnn_config.write(self.out_returnn_config_file.get_path())

        cmd = self._get_run_cmd()
        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        assert os.path.exists(self.model_checkpoint.index_path.get_path()), (
            "Provided model does not exists: %s" % self.model_checkpoint
        )

    def run(self):
        cmd = self._get_run_cmd()
        sp.check_call(cmd)

        merged_scores = np.loadtxt(self.out_prior_txt_file.get_path(), delimiter=" ")

        with open(self.out_prior_xml_file.get_path(), "wt") as f_out:
            f_out.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n'
                % len(merged_scores)
            )
            f_out.write(" ".join("%.20e" % s for s in merged_scores) + "\n")
            f_out.write("</vector-f32>")

    def plot(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        merged_scores = np.loadtxt(self.out_prior_txt_file.get_path(), delimiter=" ")

        xdata = range(len(merged_scores))
        plt.semilogy(xdata, np.exp(merged_scores))
        plt.xlabel("emission idx")
        plt.ylabel("prior")
        plt.grid(True)
        plt.savefig(self.out_prior_png_file.get_path())

    def _get_run_cmd(self):
        return [
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("rnn.py").get_path(),
            self.out_returnn_config_file.get_path(),
        ]

    @classmethod
    def create_returnn_config(
        cls,
        model_checkpoint: Checkpoint,
        returnn_config: ReturnnConfig,
        prior_data: Optional[Dict[str, Any]],
        log_verbosity: int,
        device: str,
        **kwargs,
    ):
        """
        Creates compute_prior RETURNN config
        :param model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param returnn_config: object representing RETURNN config
        :param prior_data: dataset used to compute prior (None = use one train epoch)
        :param log_verbosity: RETURNN log verbosity
        :param device: RETURNN device, cpu or gpu
        :rtype: ReturnnConfig
        """
        assert device in ["gpu", "cpu"]
        original_config = returnn_config.config

        config = copy.deepcopy(original_config)
        config["load"] = model_checkpoint
        config["task"] = "compute_priors"

        if prior_data is not None:
            config["train"] = prior_data

        post_config = {
            "device": device,
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
        }

        post_config.update(copy.deepcopy(returnn_config.post_config))

        res = copy.deepcopy(returnn_config)
        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config_generator": kwargs["returnn_config_generator"],
            "returnn_config_generator_kwargs": kwargs[
                "returnn_config_generator_kwargs"
            ],
            "model_checkpoint": kwargs["model_checkpoint"],
            "trial": kwargs["trial"],
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }
        return super().hash(d)
