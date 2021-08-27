import os
import subprocess as sp
import yaml

from sisyphus import *

import recipe.i6_core.util as util


class FairseqHydraConfig:
    """
    An object that manages a Fairseq hydra config (inspired by the ReturnnConfig).
    """

    def __init__(self, fairseq_hydra_config_dict, *, yaml_prefix=""):
        """
        :param dict fairseq_hydra_config_dict: Contains the information which is needed for fairseq-hydra-train. Will be converted and dumped into a .yaml
        :param str yaml_prefix: Prefix which should be written to the beginning of the config, for example "# @package _group_"
        """
        assert isinstance(fairseq_hydra_config_dict, dict)
        self.fairseq_hydra_config_dict = fairseq_hydra_config_dict
        self.yaml_prefix = yaml_prefix

    def write(self, path):
        yaml_dict = yaml.dump(self.fairseq_hydra_config_dict)
        # "# @package _group_" was written at the beginning in the example .yaml from fairseq:
        if self.yaml_prefix != "":
            yaml_dict = self.yaml_prefix + "\n" + yaml_dict
        with open(path, "w") as file:
            file.write(yaml_dict)


class FairseqHydraTrainingJob(Job):
    """
    Train a Fairseq model using fairseq-hydra-train
    """

    def __init__(
        self,
        fairseq_hydra_config,
        *,  # args below are keyword only
        command_line_args=None,
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        gpu_rqmt=1,
        fairseq_python_exe=None,
        fairseq_hydra_exe=None,
    ):
        """
        :param FairseqHydraConfig fairseq_hydra_config:
        :param list command_line_args: The command line arguments needed to configure the Fairseq-hydra task ('--config-dir' and '--config-name' are already taken care of)
        :param int|float time_rqmt: Overall time requirements
        :param int|float mem_rqmt: Memory requirements (per GPU)
        :param int cpu_rqmt: Required number of CPUs (per GPU)
        :param int gpu_rqmt: Number of required GPUs
        :param Path|str python_exe: File path to the executable for running python
        :param Path|str fairseq_hydra_exe: File path to the Fairseq executable: fairseq-hydra-train (usually in the same folder as python exe '.../bin/')

        """
        self.fairseq_hydra_config = fairseq_hydra_config
        self.command_line_args = command_line_args or []
        self.yaml_config_name = "fairseq_hydra_config.yaml"
        self.out_fairseq_hydra_yaml = self.output_path(self.yaml_config_name)
        self.out_checkpoint_dir = self.output_path("checkpoints")
        self.gpu_rqmt = gpu_rqmt
        self.fairseq_python_exe = (
            fairseq_python_exe
            if fairseq_python_exe is not None
            else gs.FAIRSEQ_PYTHON_EXE
        )
        self.fairseq_hydra_exe = (
            fairseq_hydra_exe if fairseq_hydra_exe is not None else gs.FAIRSEQ_HYDRA_EXE
        )

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

        if self.gpu_rqmt > 1:
            self.rqmt["cpu"] *= self.gpu_rqmt
            self.rqmt["mem"] *= self.gpu_rqmt

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def _get_run_cmd(self):
        run_cmd = [
            tk.uncached_path(self.fairseq_python_exe),
            tk.uncached_path(self.fairseq_hydra_exe),
            "--config-dir",
            os.path.dirname(self.out_returnn_config_file.get_path()),
            "--config-name",
            self.yaml_config_name,
        ]
        run_cmd += self.command_line_args
        run_cmd += ["checkpoint.save_dir=" + str(self.out_checkpoint_dir)]
        return run_cmd

    def create_files(self):
        self.fairseq_hydra_config.write(self.out_fairseq_hydra_yaml.get_path())
        util.create_executable("fairseq.sh", self._get_run_cmd())

    def run(self):
        sp.check_call(self._get_run_cmd())
