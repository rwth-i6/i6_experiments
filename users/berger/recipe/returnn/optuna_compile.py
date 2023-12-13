__all__ = ["OptunaCompileTFGraphJob"]

import inspect
from sisyphus import *

Path = setup_path(__package__)

import copy
import os
import subprocess as sp

import i6_core.util as util


class OptunaCompileTFGraphJob(Job):
    """
    This Job is a wrapper around the RETURNN tool comptile_tf_graph.py

    """

    __sis_hash_exclude__ = {"device": None, "epoch": None}

    def __init__(
        self,
        optuna_returnn_config,
        trial,
        train=0,
        eval=0,
        search=0,
        epoch=None,
        verbosity=4,
        device=None,
        summaries_tensor_name=None,
        output_format="meta",
        returnn_python_exe=None,
        returnn_root=None,
        rec_step_by_step=None,
        rec_json_info=None,
    ):
        """

        :param ReturnnConfig|Path|str returnn_config: Path to a RETURNN config file
        :param int train:
        :param int eval:
        :param int search:
        :param int|None epoch: compile a specific epoch for networks that might change with every epoch
        :param int log_verbosity: RETURNN log verbosity from 1 (least verbose) to 5 (most verbose)
        :param str|None device: optimize graph for cpu or gpu. If `None`, defaults to cpu for current RETURNN.
            For any RETURNN version before `cd4bc382`, the behavior will depend on the `device` entry in the
            `returnn_conig`, or on the availability of a GPU on the execution host if not defined at all.
        :param summaries_tensor_name:
        :param str output_format: graph output format, one of ["pb", "pbtxt", "meta", "metatxt"]
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        :param str|None rec_step_by_step: name of rec layer for step-by-step graph
        :param bool|None rec_json_info: whether to enable rec json info for step-by-step graph compilation
        """
        self.returnn_config = None
        self.optuna_returnn_config = optuna_returnn_config
        self.trial = trial
        self.train = train
        self.eval = eval
        self.search = search
        self.epoch = epoch
        self.verbosity = verbosity
        self.device = device
        self.summaries_tensor_name = summaries_tensor_name
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root

        self.rec_step_by_step = rec_step_by_step
        self.rec_json_info = rec_json_info

        self.out_graph = self.output_path("graph.%s" % output_format)
        self.out_model_params = self.output_var("model_params.pickle", pickle=True)
        self.out_state_vars = self.output_var("state_vars.pickle", pickle=True)
        self.out_returnn_config = self.output_path("returnn.config")

        self.rqmt = {"gpu": 1, "mem": 4.0}  # None

    def tasks(self):
        yield Task("create_files", mini_task=True)
        if self.rqmt:
            yield Task("run", resume="run", rqmt=self.rqmt)
        else:
            yield Task("run", resume="run", mini_task=True)

    def create_files(self):
        self.returnn_config = self.optuna_returnn_config.generate_config(self.trial.get())
        self.returnn_config.write(self.out_returnn_config.get_path())

    def run(self):
        args = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "tools/compile_tf_graph.py"),
            self.out_returnn_config.get_path(),
            f"--train={self.train}",
            f"--eval={self.eval}",
            f"--search={self.search}",
            f"--verbosity={self.verbosity}",
            f"--output_file={self.out_graph.get_path()}",
            "--output_file_model_params_list=model_params",
            "--output_file_state_vars_list=state_vars",
        ]
        if self.device is not None:
            args.append(f"--device={self.device}")
        if self.epoch is not None:
            if isinstance(self.epoch, tk.Variable):
                epoch_get = self.epoch.get()
            else:
                epoch_get = self.epoch

            args.append(f"--epoch={epoch_get}")
        if self.summaries_tensor_name is not None:
            args.append(f"--summaries_tensor_name={self.summaries_tensor_name}")
        if self.rec_step_by_step is not None:
            args.append(f"--rec_step_by_step={self.rec_step_by_step}")
            if self.rec_json_info:
                args.append("--rec_step_by_step_output_file=rec.info")

        util.create_executable("run.sh", args)

        sp.check_call(args)

        with open("model_params", "rt") as input:
            lines = [l.strip() for l in input if len(l.strip()) > 0]
            self.out_model_params.set(lines)
        with open("state_vars", "rt") as input:
            lines = [l.strip() for l in input if len(l.strip()) > 0]
            self.out_state_vars.set(lines)

    @classmethod
    def hash(cls, kwargs):
        c = copy.copy(kwargs)
        del c["optuna_returnn_config"]
        c.update(
            {
                "returnn_config_generator": inspect.getsource(kwargs["optuna_returnn_config"].config_generator),
                "returnn_config_generator_kwargs": list(sorted(kwargs["optuna_returnn_config"].config_kwargs)),
            }
        )
        return super().hash(c)
