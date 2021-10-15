import os
import subprocess as sp
import yaml
import sys

from sisyphus import *

import recipe.i6_core.util as util


class FairseqHydraConfig:
    """
    An object that manages a Fairseq hydra config (inspired by the ReturnnConfig).
    """

    def __init__(self, config_dict, *, yaml_prefix=""):
        """
        :param dict config_dict: Contains the information which is needed for fairseq-hydra-train. Will be converted and dumped into a .yaml
        :param str yaml_prefix: Prefix which should be written to the beginning of the config, for example "# @package _group_"
        """
        assert isinstance(config_dict, dict)
        self.config_dict = config_dict
        self.yaml_prefix = yaml_prefix

    def write(self, path):
        config_yaml = yaml.dump(self.config_dict)
        # "# @package _group_" was written at the beginning in the example .yaml from fairseq:
        if self.yaml_prefix != "":
            config_yaml = self.yaml_prefix + "\n" + config_yaml
        with open(path, "w") as file:
            file.write(config_yaml)


class PytorchHydraModel:
    """
    Defines a Pytorch hydra model as yaml config, pytorch checkpoint file and epoch
    """

    def __init__(self, fairseq_hydra_config_file, model, epoch):
        """

        :param Path fairseq_hydra_config_file: Path to a returnn config file
        :param Path model: Path to a pytorch checkpoint
        :param int epoch:
        """
        self.returnn_config_file = fairseq_hydra_config_file
        self.model = model
        self.epoch = epoch


class FairseqHydraTrainingJob(Job):
    """
    Train a Fairseq model using fairseq-hydra-train
    """

    def __init__(
        self,
        fairseq_hydra_config,
        *,  # args below are keyword only
        command_line_args=None,
        max_epoch=1,
        save_interval=1,
        keep_epochs=None,
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        gpu_rqmt=1,
        fairseq_python_exe=None,
        fairseq_hydra_exe=None,
        fairseq_root=None,
    ):
        """
        :param FairseqHydraConfig fairseq_hydra_config:
        :param list command_line_args: Additional command line arguments (starting with "--*"),
            to configure the Fairseq-hydra task
        :param int max_epoch: maximum number of epochs to run. Note that this value IS currently HASHED.
        :param int save_interval: save a checkpoint each n-th epoch
        :param list[int]|set[int]|None keep_epochs: specify which checkpoints are kept in self.out_models.
            Use None for each save_interval-th epoch
        :param int|float time_rqmt: Overall time requirements
        :param int|float mem_rqmt: Memory requirements (per GPU)
        :param int cpu_rqmt: Required number of CPUs (per GPU)
        :param int gpu_rqmt: Number of required GPUs
        :param Path|str python_exe: File path to the executable for running python
        :param Path|str fairseq_hydra_exe: File path to the python executable for running fairseq-hydra-train.
            (usually in the same folder as python exe '.../bin/')
        :param Path|str fairseq_root: File path to the fairseq git for alternative call of fairseq-hydra-train
            (no need to install fairseq here)

        """

        # Inputs:
        self.fairseq_hydra_config = fairseq_hydra_config
        self.command_line_args = command_line_args or []
        stored_epochs = list(range(save_interval, max_epoch, save_interval)) + [
            max_epoch
        ]
        if keep_epochs is None:
            self.keep_epochs = set(stored_epochs)
        else:
            self.keep_epochs = set(keep_epochs)
        self.max_epoch = max_epoch
        self.save_interval = save_interval
        self.fairseq_python_exe = (
            fairseq_python_exe
            if fairseq_python_exe is not None
            else gs.FAIRSEQ_PYTHON_EXE
        )
        self.fairseq_hydra_exe = (
            fairseq_hydra_exe if fairseq_hydra_exe is not None else gs.FAIRSEQ_HYDRA_EXE
        )
        self.fairseq_root = fairseq_root

        # Outputs:
        self.out_fairseq_hydra_yaml = self.output_path("fairseq_hydra_config.yaml")
        self.out_checkpoint_dir = self.output_path("checkpoints", directory=True)
        self.out_models = {
            k: PytorchHydraModel(
                self.out_fairseq_hydra_yaml,
                self.output_path("checkpoints/checkpoint{}.pt".format(k)),
                k,
            )
            for k in stored_epochs
            if k in self.keep_epochs
        }

        # Requirements:
        self.gpu_rqmt = gpu_rqmt
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

    def create_files(self):
        self.fairseq_hydra_config.write(self.out_fairseq_hydra_yaml.get_path())
        util.create_executable("fairseq.sh", self._get_run_cmd())

    def run(self):
        my_env = os.environ
        if self.fairseq_root is not None:
            my_env["PYTHONPATH"] = self.fairseq_root
        sp.check_call(self._get_run_cmd(), env=my_env)

    def _get_run_cmd(self):
        run_cmd = [
            "--config-dir",
            os.path.dirname(self.out_fairseq_hydra_yaml.get_path()),
            "--config-name",
            os.path.basename(self.out_fairseq_hydra_yaml.get_path()),
        ]
        run_cmd += self.command_line_args
        run_cmd += ["checkpoint.save_dir=" + self.out_checkpoint_dir.get_path()]
        run_cmd += ["checkpoint.save_interval=" + str(self.save_interval)]
        run_cmd += ["optimization.max_epoch=" + str(self.max_epoch)]
        if self.fairseq_root is not None:
            sys.path.insert(0, self.fairseq_root)
            hydra_train_entry = self.fairseq_root + "fairseq_cli/hydra_train.py"
            run_cmd.insert(0, tk.uncached_path(self.fairseq_python_exe))
            run_cmd.insert(1, tk.uncached_path(hydra_train_entry))
        else:
            run_cmd.insert(0, tk.uncached_path(self.fairseq_python_exe))
            run_cmd.insert(1, tk.uncached_path(self.fairseq_hydra_exe))
        return run_cmd