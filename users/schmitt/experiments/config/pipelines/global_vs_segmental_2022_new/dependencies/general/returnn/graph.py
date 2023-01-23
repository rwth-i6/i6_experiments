from i6_private.users.schmitt.returnn.tools import CompileTFGraphJob

from i6_core.returnn.config import ReturnnConfig

from sisyphus import *


class ReturnnGraph:
  def __init__(self, returnn_config: ReturnnConfig):
    self.meta_graph_path = CompileTFGraphJob(returnn_config, "output").out_graph
