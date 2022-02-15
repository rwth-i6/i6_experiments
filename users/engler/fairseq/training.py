import logging
import os
import subprocess as sp
import yaml
import sys
import copy

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
        # recursively go through config dictionary to get all sisyphus paths inplace
        def get_sis_paths(cnfg_d):
            for k in cnfg_d.keys():
                if type(cnfg_d[k]) == dict:
                    get_sis_paths(cnfg_d[k])
                elif type(cnfg_d[k]) == tk.Path:
                    cnfg_d[k] = cnfg_d[k].get_path()
        path_corrected_config = self.config_dict.copy()
        get_sis_paths(path_corrected_config)

        config_yaml = yaml.dump(path_corrected_config)
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
        max_epoch=None,
        save_interval=None,
        keep_epochs=None,
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        gpu_rqmt=1,
        fairseq_python_exe=None,
        fairseq_hydra_exe=None,
        fairseq_root=None,
        use_cache_manager=True,
    ):
        """
        :param FairseqHydraConfig fairseq_hydra_config:
        :param list command_line_args: Additional command line arguments (starting with "--*"),
            to configure the Fairseq-hydra task
        :param int|None max_epoch: maximum number of epochs to run. Note that this value IS currently HASHED.
        :param int|None save_interval: save a checkpoint each n-th epoch
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
        :param bool use_cache_manager: enables caching of data given in the manifest with the i6 cache manager

        """

        # Inputs:
        self.fairseq_hydra_config = fairseq_hydra_config
        self.command_line_args = command_line_args or []
        warning_text = "is specified as input arg and in fairseq_hydra_config. We take the input arg"
        save_interval_config = fairseq_hydra_config.config_dict.get(
            "checkpoint", {}
        ).get("save_interval", None)
        if save_interval is not None and save_interval_config is not None:
            logging.warning(
                "'save_interval' {}: {}".format(warning_text, save_interval)
            )
        self.save_interval = save_interval or save_interval_config
        assert (
            self.save_interval is not None
        ), "save_interval has to be set explicitly or via fairseq_hydra_config"
        max_epoch_config = fairseq_hydra_config.config_dict.get("optimization", {}).get(
            "max_epoch", None
        )
        if max_epoch is not None and max_epoch_config is not None:
            logging.warning("'max_epoch' {}: {}".format(warning_text, max_epoch))
        self.max_epoch = max_epoch or max_epoch_config
        assert (
            self.max_epoch is not None
        ), "max_epoch has to be set explicitly or via fairseq_hydra_config"
        stored_epochs = list(
            range(self.save_interval, self.max_epoch, self.save_interval)
        ) + [self.max_epoch]
        if keep_epochs is None:
            self.keep_epochs = set(stored_epochs)
        else:
            self.keep_epochs = set(keep_epochs)
        self.fairseq_python_exe = (
            fairseq_python_exe
            if fairseq_python_exe is not None
            else getattr(gs, "FAIRSEQ_PYTHON_EXE", None)
        )
        self.fairseq_hydra_exe = fairseq_hydra_exe
        self.fairseq_root = fairseq_root
        # We assume that only one of the two possible entry points is given as an input
        assert (self.fairseq_root is not None) ^ (self.fairseq_hydra_exe is not None)
        if self.fairseq_root is not None:
            assert self.fairseq_python_exe is not None
        self.use_cache_manager=use_cache_manager

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
        self.out_cached_audio_manifest = self.output_path("cached_audio_manifest", directory=True)

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
        if self.use_cache_manager:
            manifest_path = self.fairseq_hydra_config.config_dict["task"]["data"].get_path()
            for name in ["train.tsv", "valid.tsv"]:
                with open(f"{manifest_path}/{name}", "r") as manifest_file:
                    manifest_lines = manifest_file.read().splitlines()
                audio_path = manifest_lines[0]
                bundle_lines = map(lambda line: audio_path + "/" + line.split("\t")[0], manifest_lines[1:])
                with open(f"{name}.bundle", 'w') as bundle_file:
                    bundle_file.write("\n".join(bundle_lines))
                try:
                    cached_audio_fn = sp.check_output(["cf", f"{name}.bundle"]).strip().decode("utf8")
                except sp.CalledProcessError:
                    print(f"Cache manager: Error occurred for files in {name}")
                    raise

                with open(cached_audio_fn) as local_bundle:
                    bundle_lines = list(map(os.path.dirname, local_bundle.readlines()))
                    assert bundle_lines.count(bundle_lines[0]) == len(bundle_lines), f"not all {name} files in same directory"
                    manifest_lines[0] = bundle_lines[0]
                with open(f"{self.out_cached_audio_manifest.get_path()}/{name}", "w") as cached_audio_manifest_file:
                    cached_audio_manifest_file.write('\n'.join(manifest_lines))

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
        if self.save_interval != self.fairseq_hydra_config.config_dict.get(
            "checkpoint", {}
        ).get("save_interval", None):
            run_cmd += ["checkpoint.save_interval=" + str(self.save_interval)]
        if self.max_epoch != self.fairseq_hydra_config.config_dict.get(
            "optimization", {}
        ).get("max_epoch", None):
            run_cmd += ["optimization.max_epoch=" + str(self.max_epoch)]

        if self.use_cache_manager:
            run_cmd += ["task.data=" + self.out_cached_audio_manifest.get_path()]

        if self.fairseq_root is not None:
            sys.path.insert(0, self.fairseq_root)
            hydra_train_entry = self.fairseq_root + "fairseq_cli/hydra_train.py"
            run_cmd.insert(0, tk.uncached_path(hydra_train_entry))
        else:
            run_cmd.insert(0, tk.uncached_path(self.fairseq_hydra_exe))
        if self.fairseq_python_exe is not None:
            run_cmd.insert(0, tk.uncached_path(self.fairseq_python_exe))
        return run_cmd

    @classmethod
    def hash(cls, kwargs):
        d = copy.copy(kwargs)
        d.pop("use_cache_manager", None)
        d.pop("time_rqmt", None)
        d.pop("mem_rqmt", None)
        d.pop("cpu_rqmt", None)
        d.pop("gpu_rqmt", None)
        return super().hash(d)

