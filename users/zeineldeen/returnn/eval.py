__all__ = ['ReturnnForwardJob']

from sisyphus import *

import copy
import glob
import os
import shutil
import subprocess as sp

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint
import i6_core.util as util

Path = setup_path(__package__)


class ReturnnEvalJob(Job):
    """
    Run a RETURNN "eval" task, can optionally store HDFs
    """
    def __init__(self, model_checkpoint, returnn_config, hdf_outputs=None,
                 *,  # args below are keyword only
                 log_verbosity=3, device='gpu',
                 time_rqmt=4, mem_rqmt=4, cpu_rqmt=4,
                 returnn_python_exe=None, returnn_root=None):
        """

        :param Checkpoint model_checkpoint: Checkpoint object pointing to a stored RETURNN tensorflow model
        :param ReturnnConfig returnn_config: RETURNN config dict
        :param dict returnn_post_config: RETURNN config dict (no hashing)
        :param list[str] hdf_outputs: list of additional hdf output layer file names that the network generates (e.g. attention.hdf);
          The hdf outputs have to be a valid subset or be equal to the hdf_dump_layers in the config.
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :param int time_rqmt: job time requirement
        :param int mem_rqmt: job memory requirement
        :param int cpu_rqmt: job cpu requirement
        :param Path|str returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param Path|str returnn_root: path to the RETURNN src folder
        """
        self.returnn_python_exe   = returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE
        self.returnn_root         = returnn_root       if returnn_root       is not None else gs.RETURNN_ROOT

        self._model_checkpoint = model_checkpoint
        self._returnn_config = returnn_config
        self._log_verbosity = log_verbosity
        self._device = device

        self.out_returnn_config_file = self.output_path('returnn.config')

        self.out_hdf_files = {}
        hdf_outputs = hdf_outputs if hdf_outputs else []
        for output in hdf_outputs:
            self.out_hdf_files[output] = self.output_path(output)

        self.rqmt = {'gpu': 1 if device == "gpu" else 0,
                     'cpu': cpu_rqmt, 'mem': mem_rqmt, 'time': time_rqmt}

    def tasks(self):
        yield Task('create_files', mini_task=True)
        yield Task('run', resume='run', rqmt=self.rqmt)

    def create_files(self):
        config = self.create_returnn_config(
            model_checkpoint=self._model_checkpoint,
            returnn_config=self._returnn_config,
            log_verbosity=self._log_verbosity,
            device=self._device
        )
        config.write(self.out_returnn_config_file.get_path())

        cmd = [tk.uncached_path(self.returnn_python_exe),
               os.path.join(tk.uncached_path(self.returnn_root), 'rnn.py'),
               self.out_returnn_config_file.get_path()]
        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        assert os.path.exists(self._model_checkpoint.index_path.get_path()), \
            "Provided model does not exists: %s" % str(self._model_checkpoint)

    def run(self):
        call = [tk.uncached_path(self.returnn_python_exe),
                os.path.join(tk.uncached_path(self.returnn_root), 'rnn.py'),
                self.out_returnn_config_file.get_path()]
        sp.check_call(call)

        # move hdf outputs to output folder
        for k, v in self.out_hdf_files.items():
            shutil.move(k, v.get_path())

        # delete dumped file and hdf files that were not marked as output, if remaining
        for file in glob.glob("dump*"):
            os.unlink(file)
        for file in glob.glob("*.hdf"):
            os.unlink(file)

    @classmethod
    def create_returnn_config(cls, model_checkpoint, returnn_config, log_verbosity, device, **kwargs):
        """

        :param Checkpoint model_checkpoint:
        :param ReturnnConfig returnn_config:
        :param int log_verbosity:
        :param str device:
        :param kwargs:
        :return:
        """
        assert device in ['gpu', 'cpu']
        assert 'network' in returnn_config.config

        res = copy.deepcopy(returnn_config)

        config = {"load": model_checkpoint}

        post_config = { 'device'          : device,
                        'log'             : ['./returnn.log'],
                        'log_verbosity'   : log_verbosity,
                        'task'            : 'eval',
                      }

        config.update(returnn_config.config)
        post_config.update(returnn_config.post_config)

        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = { 'returnn_config'     : ReturnnEvalJob.create_returnn_config(**kwargs),
              'returnn_python_exe' : kwargs['returnn_python_exe'],
              'returnn_root'       : kwargs['returnn_root'],}

        return super().hash(d)