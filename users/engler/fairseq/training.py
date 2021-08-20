import os
import subprocess as sp
import yaml

from sisyphus import *

import recipe.i6_core.util as util


class DictToYamlJob(Job):
    """
    Writes a dict into a .yaml file
    """

    def __init__(self, config_dict, hash_reset, *, yaml_prefix=""):
        """
        :param dict config_dict:
        :param str hash_reset: Temporary way to reset the sisyphus hash for debugging purposes
        :param str yaml_prefix: Prefix which should be written to the beginning, for example "# @package _group_" or "#!rnn.py"
        """
        assert isinstance(config_dict, dict)

        self.config_dict = config_dict
        self.yaml_prefix = yaml_prefix
        self.out_config_file = self.output_path("fairseq_config.yaml")
        self.file_out = self.output_path("out")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        yaml_dict = yaml.dump(self.config_dict)
        # "# @package _group_" was written at the beginning in the example .yaml from fairseq:
        if self.yaml_prefix != "":
            yaml_dict = self.yaml_prefix + "\n" + yaml_dict
        with open(self.out_config_file.get_path(), "w") as file:
            file.write(yaml_dict)


class FairseqHydraTrainingJob(Job):
    """
    Train a Fairseq model using fairseq-hydra-train
    """

    def __init__(
        self,
        python_exe,
        fairseq_exe,
        hash_reset,
        *,  # args below are keyword only
        additional_args=[],
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        gpu_rqmt=1,
    ):
        """
        :param Path|str python_exe: File path to the executable for running python
        :param Path|str fairseq_exe: File path to the Fairseq executable: fairseq-hydra-train (usually in the same folder as python exe '.../bin/')
        :param str hash_reset: temporary way to reset the sisyphus hash for debugging purposes
        :param list additional_args: The arguments needed to configure the Fairseq task
        :param int|float time_rqmt:
        :param int|float mem_rqmt:
        :param int cpu_rqmt:
        :param int gpu_rqmt:
        """
        self.fairseq_python_exe = python_exe
        self.fairseq_train = fairseq_exe
        self.additional_args = additional_args
        self.checkpoint_dir = self.output_path("checkpoints")
        self.file_out = self.output_path("out")

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def _get_run_cmd(self):
        run_cmd = [
            tk.uncached_path(self.fairseq_python_exe),
            tk.uncached_path(self.fairseq_train),
        ]
        run_cmd += self.additional_args
        run_cmd += ["checkpoint.save_dir=" + str(self.checkpoint_dir)]
        return run_cmd

    def create_files(self):
        util.create_executable("fairseq.sh", self._get_run_cmd())

    def run(self):
        sp.check_call(self._get_run_cmd())
