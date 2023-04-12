import json

from i6_core.returnn.config import CodeWrapper
from recipe.i6_experiments.users.schmitt.experiments.swb.transducer import conformer
from sisyphus.delayed_ops import DelayedFunction
# from returnn.tf.util.data import Dim

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.network_builder.base import NetworkBuilderBase

from abc import ABC


class GlobalNetworkBuilder(NetworkBuilderBase):
  @property
  def chunk_size_dim(self) -> CodeWrapper:
    return CodeWrapper('SpatialDim("chunk-size")')

  @property
  def chunked_time_dim(self) -> CodeWrapper:
    return CodeWrapper('SpatialDim("chunked-time", 1)')

  @property
  def rec_layer_name(self) -> str:
    return "output"

  def add_attention_windows(self):
    self._net_dict[self.rec_layer_name]["unit"].update({
      "att_values": {
        "class": "split_dims",
        "from": "base:encoder",
        "axis": "T",
        "dims": (self.chunked_time_dim, self.chunk_size_dim),
      "att_ctx": {
        "class": "split_dims",
        "from": "base:enc_ctx",
        "axis": "T",
        "dims": (self.chunked_time_dim, self.chunk_size_dim),
      }
    }})
