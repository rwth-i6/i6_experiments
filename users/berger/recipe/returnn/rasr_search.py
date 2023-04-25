__all__ = ["ReturnnRasrSearchJob"]

from sisyphus import *

Path = setup_path(__package__)

import copy
import os
import subprocess as sp

import i6_core.rasr as rasr
import i6_core.mm as mm
import i6_core.util as util

from i6_core.returnn.config import ReturnnConfig


class ReturnnRasrSearchJob(Job):
    """
    Given a model checkpoint, run search task with RETURNN that uses ExternSpringDataset, and needs
    to write RASR config and flow files.
    """

    def __init__(
        self,
        crp,
        feature_flow,
        model_checkpoint,
        returnn_config,
        *,
        output_mode="py",
        log_verbosity=3,
        use_gpu=False,
        rtf=1.0,
        mem=4,
        cpu_rqmt=2,
        returnn_python_exe=None,
        returnn_root=None,
        buffer_size=200 * 1024,
        extra_rasr_config=None,
        extra_rasr_post_config=None,
        additional_rasr_config_files=None,
        additional_rasr_post_config_files=None,
    ):
        """
        :param rasr.CommonRasrParameters crp:
        :param rasr.FlowNetwork feature_flow: RASR flow file for feature extraction or feature cache
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param str output_mode: "txt" or "py"
        :param int log_verbosity: RETURNN log verbosity
        :param bool use_gpu:
        :param float|int time_rqmt: job time requirement in hours
        :param float|int mem: job memory requirement in GB
        :param float|int cpu_rqmt: job cpu requirement in GB
        :param tk.Path|str|None returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param tk.Path|str|None returnn_root: path to the RETURNN src folder
        :param buffer_size:
        :param extra_rasr_config:
        :param extra_rasr_post_config:
        :param additional_rasr_config_files:
        :param additional_rasr_post_config_files:
        """
        assert isinstance(returnn_config, ReturnnConfig)
        kwargs = locals()
        del kwargs["self"]

        self.model_checkpoint = model_checkpoint

        self.returnn_python_exe = returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE

        self.returnn_root = returnn_root if returnn_root is not None else gs.RETURNN_ROOT

        self.rasr_exe = rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer")

        (
            self.rasr_config,
            self.rasr_post_config,
        ) = ReturnnRasrSearchJob.create_config(**kwargs)

        self.additional_rasr_config_files = additional_rasr_config_files or {}
        self.additional_rasr_post_config_files = additional_rasr_post_config_files or {}

        self.feature_flow = ReturnnRasrSearchJob.create_flow(feature_flow)

        self.out_returnn_config_file = self.output_path("returnn.config")

        self.out_search_file = self.output_path("search_out")

        self.returnn_config = ReturnnRasrSearchJob.create_returnn_config(**kwargs)
        self.returnn_config.post_config["search_output_file"] = self.out_search_file

        self.rqmt = {
            "gpu": 1 if use_gpu else 0,
            "cpu": cpu_rqmt,
            "mem": mem,
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 0.5),
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        config = self.returnn_config
        config.write(self.out_returnn_config_file.get_path())

        cmd = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]

        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        assert os.path.exists(
            tk.uncached_path(self.model_checkpoint.index_path)
        ), "Provided model does not exists: %s" % str(self.model_checkpoint)

        rasr.RasrCommand.write_config(
            self.rasr_config,
            self.rasr_post_config,
            "rasr.eval.config",
        )

        additional_files = set(self.additional_rasr_config_files.keys())
        additional_files.update(set(self.additional_rasr_post_config_files.keys()))
        for f in additional_files:
            rasr.RasrCommand.write_config(
                self.additional_rasr_config_files.get(f, {}),
                self.additional_rasr_post_config_files.get(f),
                f + ".config",
            )

        self.feature_flow.write_to_file("feature.flow")

    def run(self):
        call = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        sp.check_call(call)

    @classmethod
    def create_returnn_config(
        cls,
        crp,
        model_checkpoint,
        returnn_config,
        output_mode,
        log_verbosity,
        use_gpu,
        **kwargs,
    ):
        """
        Creates search RETURNN config
        :param rasr.CommonRasrParameters crp:
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param str output_mode: "txt" or "py"
        :param int log_verbosity: RETURNN log verbosity
        :param bool use_gpu:
        :rtype: ReturnnConfig
        """
        original_config = returnn_config.config
        assert "network" in original_config
        assert output_mode in ["py", "txt"]

        config = {
            "load": model_checkpoint.ckpt_path,
            "search_output_file_format": output_mode,
            "need_data": False,
            "search_do_eval": 0,
        }

        config.update(copy.deepcopy(original_config))  # update with the original config

        # override always
        config["task"] = "search"
        config["max_seq_length"] = 0

        search_data = {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
            "sprintConfigStr": "--config=rasr.eval.config --*.LOGFILE=nn-trainer.eval.log --*.TASK=1",
            "partitionEpoch": 1,
        }

        if "search_data" in original_config:
            config["search_data"] = {
                **original_config["search_data"].copy(),
                **search_data,
            }
        else:
            config["search_data"] = search_data

        post_config = {
            "device": "gpu" if use_gpu else "cpu",
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
    def create_flow(cls, feature_flow):
        flow = copy.deepcopy(feature_flow)
        flow.flags["cache_mode"] = "bundle"
        return flow

    @classmethod
    def create_config(cls, crp, buffer_size, extra_rasr_config, extra_rasr_post_config, **kwargs):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "acoustic_model": "neural-network-trainer.model-combination.acoustic-model",
                "corpus": "neural-network-trainer.corpus",
                "lexicon": "neural-network-trainer.model-combination.lexicon",
            },
            parallelize=(crp.concurrent == 1),
        )

        config.neural_network_trainer.action = "python-control"
        config.neural_network_trainer.feature_extraction.file = "feature.flow"
        config.neural_network_trainer.python_control_enabled = True
        config.neural_network_trainer.python_control_loop_type = "iterate-corpus"

        config.neural_network_trainer.buffer_type = "utterance"
        config.neural_network_trainer.buffer_size = buffer_size
        config.neural_network_trainer.shuffle = False
        config.neural_network_trainer.window_size = 1
        config.neural_network_trainer.window_size_derivatives = 0
        config.neural_network_trainer.regression_window_size = 5

        config._update(extra_rasr_config)
        post_config._update(extra_rasr_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        d = {
            "config": config,
            "feature_flow": cls.create_flow(kwargs["feature_flow"]),
            "returnn_config": cls.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
            "rasr_exe": kwargs["crp"].nn_trainer_exe,
        }
        if kwargs["additional_rasr_config_files"] is not None:
            d["additional_rasr_config_files"] = kwargs["additional_rasr_config_files"]

        return super().hash(d)
