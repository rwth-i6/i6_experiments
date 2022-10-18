from sisyphus import *
import os
import subprocess
from i6_core.util import create_executable


class CompileTFGraphJob(Job):
  def __init__(self, returnn_config, rec_step_by_step_name, time_rqtm=1,
               mem_rqmt=2, returnn_python_exe=None, returnn_root=None):
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.returnn_config = returnn_config
    self.rec_step_by_step_name = rec_step_by_step_name

    self.mem_rqmt = mem_rqmt
    self.time_rqmt = time_rqtm

    self.out_graph = self.output_path("out-graph.meta")
    self.out_rec_info = self.output_path("out-rec.info")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

  def run(self):
    self.returnn_config.write("returnn.config")
    command = [
      self.returnn_python_exe,
      os.path.join(self.returnn_root, "tools/compile_tf_graph.py"),
      "returnn.config", "--rec_step_by_step", self.rec_step_by_step_name,
      "--rec_step_by_step_output_file", self.out_rec_info.get_path(),
      "--output_file", self.out_graph.get_path()
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])