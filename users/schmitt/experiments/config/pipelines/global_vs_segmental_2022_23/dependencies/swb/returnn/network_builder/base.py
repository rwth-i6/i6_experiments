import json

from i6_core.returnn.config import CodeWrapper
from recipe.i6_experiments.users.schmitt.experiments.swb.transducer import conformer
from sisyphus.delayed_ops import DelayedFunction
# from returnn.tf.util.data import Dim

from abc import ABC, abstractmethod

from typing import Dict


class NetworkBuilderBase(ABC):
  def __init__(self):
    self._net_dict = {}

    self.dec_att_num_heads_dim = CodeWrapper('SpatialDim("dec-att-num-heads", 1)')
    self.att_value_dim = CodeWrapper('FeatureDim("val", 512)')

  @property
  @abstractmethod
  def chunk_size_dim(self) -> CodeWrapper:
    pass

  @property
  @abstractmethod
  def chunked_time_dim(self) -> CodeWrapper:
    pass

  @property
  @abstractmethod
  def rec_layer_name(self) -> str:
    pass

  def add_bi_lstm_encoder(self):
    self._net_dict.update({
      # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
      "conv0": {
        "class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
        "with_bias": True, "auto_use_channel_first": False},  # (T,40,32)
      "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0",
                 "use_channel_first": False},  # (T,20,32)
      "conv1": {
        "class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
        "with_bias": True, "auto_use_channel_first": False},  # (T,20,32)
      "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1",
                 "use_channel_first": False},  # (T,10,32)
      "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)
    })
    # Add encoder BLSTM stack.
    num_lstm_layers = 6
    src_layer = "conv_merged"
    time_reduction = [3, 2]
    if num_lstm_layers >= 1:
      self._net_dict.update({
        "lstm0_fw": {
          "class": "rec", "unit": "nativelstm2", "n_out": 1024, "L2": 0.0001, "direction": 1,
          "from": src_layer, "trainable": True}, "lstm0_bw": {
          "class": "rec", "unit": "nativelstm2", "n_out": 1024, "L2": 0.0001, "direction": -1,
          "from": src_layer, "trainable": True}})
      src_layer = ["lstm0_fw", "lstm0_bw"]
    for i in range(1, num_lstm_layers):
      red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
      self._net_dict.update({
        "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,),
                                  "from": src_layer}})
      src_layer = "lstm%i_pool" % (i - 1)
      self._net_dict.update({
        "lstm%i_fw" % i: {
          "class": "rec", "unit": "nativelstm2", "n_out": 1024, "L2": 0.0001, "direction": 1,
          "from": src_layer, "dropout": 0.3, "trainable": True}, "lstm%i_bw" % i: {
          "class": "rec", "unit": "nativelstm2", "n_out": 1024, "L2": 0.0001, "direction": -1,
          "from": src_layer, "dropout": 0.3, "trainable": True}})
      src_layer = ["lstm%i_fw" % i, "lstm%i_bw" % i]
    self._net_dict["encoder"] = {"class": "copy", "from": src_layer}

  def add_conformer_encoder(self):
    pass

  @abstractmethod
  def add_attention_windows(self):
    """
    This function needs to add the windows over which the attention is calculated, e.g. for segmental attention, the ctx
    and values need to be split into the corresponding segments.
    :return:
    """
    pass

  def add_attention_ctx(
          self
  ):
    self._net_dict["enc_ctx"] = {
      "L2": 0.0001,
      "activation": None,
      "class": "linear",
      "from": "encoder",
      "n_out": 1024,
      "with_bias": True}

  def add_decoder_base(self):
    self._net_dict[self.rec_layer_name] = {
      "unit": {
        "att": {"axes": "static", "class": "merge_dims", "from": "att0"},
        "att0": {
          "class": "dot",
          "from": ["att_weights", "att_values"],
          "reduce": self.chunk_size_dim,
          "var1": "auto",
          "var2": "auto",
        },
        "att_weights": {
          "axis": self.chunk_size_dim,
          "class": "softmax_over_spatial",
          "from": "energy",
        },
        "att_query": {
          "activation": None,
          "class": "linear",
          "from": "lm",
          "is_output_layer": False,
          "n_out": 1024,
          "with_bias": False,
        },
        "energy": {
          "L2": 0.0001,
          "activation": None,
          "class": "linear",
          "from": "energy_tanh",
          "n_out": 1,
          "out_dim": self.dec_att_num_heads_dim,
          "with_bias": False,
        },
        "energy_in": {
          "class": "combine",
          "from": ["att_ctx", "att_query"],
          "kind": "add",
          "n_out": 1024,
        },
        "energy_tanh": {
          "activation": "tanh",
          "class": "activation",
          "from": "energy_in",
        },
        "input_embed": {
          "axes": "except_time",
          "class": "merge_dims",
          "from": "input_embed0",
          "name_scope": "lm_masked/input_embed",
        },
        "input_embed0": {
          "class": "window",
          "from": "prev_non_blank_embed",
          "name_scope": "lm_masked/input_embed0",
          "window_left": 0,
          "window_right": 0,
          "window_size": 1,
        },
        "label_log_prob": {
          "class": "combine",
          "from": ["label_log_prob0"],
          "kind": "add",
        },
        "label_log_prob0": {
          "activation": "log_softmax",
          "class": "linear",
          "dropout": 0.3,
          "from": "readout",
          "n_out": 1030,
        },
        "label_prob": {
          "activation": "exp",
          "class": "activation",
          "from": "label_log_prob",
          "is_output_layer": True,
          "loss": "ce",
          "loss_opts": {"focal_loss_factor": 0.0, "label_smoothing": 0.1},
          "target": "label_ground_truth",
        },
        "lm": {
          "class": "rec",
          "from": ["prev:att", "input_embed"],
          "n_out": 1024,
          "name_scope": "lm_masked/lm",
          "unit": "nativelstm2",
        },
        "output": {
          "beam_size": 4,
          "cheating": "exclusive",
          "class": "choice",
          "from": "data",
          "initial_output": 0,
          "target": "label_ground_truth",
        },
        "prev_non_blank_embed": {
          "activation": None,
          "class": "linear",
          "from": "prev_out_non_blank",
          "n_out": 621,
          "with_bias": False,
        },
        "prev_out_non_blank": {
          "class": "reinterpret_data",
          "from": "prev:output",
          "set_sparse": True,
          "set_sparse_dim": 1031,
        },
        "readout": {
          "class": "reduce_out",
          "from": "readout_in",
          "mode": "max",
          "num_pieces": 2,
        },
        "readout_in": {
          "activation": None,
          "class": "linear",
          "from": ["lm", "att"],
          "n_out": 1000,
        },
      }
    }

  def get_net_dict(self) -> Dict:
    self.add_bi_lstm_encoder()
    self.add_attention_ctx()
    self.add_decoder_base()
    self.add_attention_windows()

    return self._net_dict