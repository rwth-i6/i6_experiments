import json

from i6_core.returnn.config import CodeWrapper
from recipe.i6_experiments.users.schmitt.experiments.swb.transducer import conformer
from sisyphus.delayed_ops import DelayedFunction
# from returnn.tf.util.data import Dim


class SegmentalNetworkBuilder:
  def __init__(self):
    self.net_dict = {}

  def _add_encoder_base(self):
    pass

  def add_bi_lstm_encoder(self):
    self.net_dict.update({
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

      "encoder": {"class": "copy", "from": "encoder0"},
    })
    # Add encoder BLSTM stack.
    num_lstm_layers = 6
    src_layer = "conv_merged"
    time_reduction = [3, 2]
    if num_lstm_layers >= 1:
      self.net_dict.update({
        "lstm0_fw": {
          "class": "rec", "unit": "nativelstm2", "n_out": 1024, "L2": 0.0001, "direction": 1,
          "from": src_layer, "trainable": True}, "lstm0_bw": {
          "class": "rec", "unit": "nativelstm2", "n_out": 1024, "L2": 0.0001, "direction": -1,
          "from": src_layer, "trainable": True}})
      src_layer = ["lstm0_fw", "lstm0_bw"]
    for i in range(1, num_lstm_layers):
      red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
      self.net_dict.update({
        "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,),
                                  "from": src_layer}})
      src_layer = "lstm%i_pool" % (i - 1)
      self.net_dict.update({
        "lstm%i_fw" % i: {
          "class": "rec", "unit": "nativelstm2", "n_out": 1024, "L2": 0.0001, "direction": 1,
          "from": src_layer, "dropout": 0.3, "trainable": True}, "lstm%i_bw" % i: {
          "class": "rec", "unit": "nativelstm2", "n_out": 1024, "L2": 0.0001, "direction": -1,
          "from": src_layer, "dropout": 0.3, "trainable": True}})
      src_layer = ["lstm%i_fw" % i, "lstm%i_bw" % i]
    self.net_dict["encoder0"] = {"class": "copy", "from": src_layer}

  def add_conformer_encoder(self):
    pass

  def _add_label_model(self):
    self.net_dict["label_model"] = {
      "back_prop": True,
      "class": "rec",
      "from": "data:label_ground_truth",
      "include_eos": True,
      "is_output_layer": True,
      "name_scope": "output/rec"
    }
    self.net_dict["output"]["unit"] = {
      "att": {"axes": "except_time", "class": "merge_dims", "from": "att0"},
      "att0": {
        "add_var2_if_empty": False,
        "class": "dot",
        "from": ["att_val_split", "att_weights"],
        "reduce": "stag:att_t",
        "var1": "f",
        "var2": None,
      },
      "att_ctx": {
        "L2": None,
        "activation": None,
        "class": "linear",
        "dropout": 0.0,
        "from": "segments",
        "n_out": 1024,
        "with_bias": False,
      },
      "att_energy": {
        "class": "reinterpret_data",
        "from": "att_energy0",
        "is_output_layer": False,
        "set_dim_tags": {
          "f": CodeWrapper('Dim(kind=Dim.Types.Spatial, description="att_heads", dimension=1)')
        },
      },
      "att_energy0": {
        "activation": None,
        "class": "linear",
        "from": ["energy_tanh"],
        "n_out": 1,
        "with_bias": False,
      },
      "att_energy_in": {
        "class": "combine",
        "from": ["att_ctx", "att_query"],
        "kind": "add",
        "n_out": 1024,
      },
      "att_query": {
        "activation": None,
        "class": "linear",
        "from": "lm",
        "is_output_layer": False,
        "n_out": 1024,
        "with_bias": False,
      },
      "att_val": {"class": "copy", "from": "segments"},
      "att_val_split": {
        "class": "reinterpret_data",
        "from": "att_val_split0",
        "set_dim_tags": {
          "dim:1": CodeWrapper('Dim(kind=Dim.Types.Spatial, description="att_heads", dimension=1)')
        },
      },
      "att_val_split0": {
        "axis": "f",
        "class": "split_dims",
        "dims": (1, -1),
        "from": "att_val",
      },
      "att_weights": {
        "class": "dropout",
        "dropout": 0.1,
        "dropout_noise_shape": {"*": None},
        "from": "att_weights0",
        "is_output_layer": False,
      },
      "att_weights0": {
        "axis": "stag:att_t",
        "class": "softmax_over_spatial",
        "energy_factor": 0.03125,
        "from": "att_energy",
      },
      "energy_tanh": {
        "activation": "tanh",
        "class": "activation",
        "from": ["att_energy_in"],
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
      "segment_lens": {
        "axis": "t",
        "class": "gather",
        "from": "base:data:segment_lens_masked",
        "position": ":i",
      },
      "segment_starts": {
        "axis": "t",
        "class": "gather",
        "from": "base:data:segment_starts_masked",
        "position": ":i",
      },
      "segments": {
        "class": "reinterpret_data",
        "from": "segments0",
        "set_dim_tags": {
          "stag:sliced-time:segments": CodeWrapper('Dim(kind=Dim.Types.Spatial, description="att_t")')
        },
      },
      "segments0": {
        "class": "slice_nd",
        "from": "base:encoder",
        "size": "segment_lens",
        "start": "segment_starts",
      },
    }

  def _add_length_model_base(self):
    self.net_dict["output"] = {
      "back_prop": True,
      "class": "rec",
      "from": "encoder",
      "include_eos": True,
      "size_target": "targetb",
      "target": "targetb"
    }
    self.net_dict["output"]["unit"] = {
      "am": {"class": "copy", "from": "data:source"},
      "blank_log_prob": {
        "class": "eval",
        "eval": "tf.math.log_sigmoid(-source(0))",
        "from": "emit_prob0",
      },
      "const1": {"class": "constant", "value": 1},
      "emit_blank_log_prob": {
        "class": "copy",
        "from": ["blank_log_prob", "emit_log_prob"],
      },
      "emit_blank_prob": {
        "activation": "exp",
        "class": "activation",
        "from": "emit_blank_log_prob",
        "loss": "ce",
        "loss_opts": {"focal_loss_factor": 0.0},
        "target": "emit_ground_truth",
      },
      "emit_log_prob": {
        "activation": "log_sigmoid",
        "class": "activation",
        "from": "emit_prob0",
      },
      "emit_prob0": {
        "activation": None,
        "class": "linear",
        "from": "s",
        "is_output_layer": True,
        "n_out": 1,
      },
      "output": {
        "beam_size": 4,
        "cheating": "exclusive",
        "class": "choice",
        "from": "data",
        "initial_output": 0,
        "input_type": "log_prob",
        "target": "targetb",
      },
      "output_emit": {
        "class": "compare",
        "from": "output",
        "initial_output": True,
        "kind": "not_equal",
        "value": 1030,
      },
      "prev_out_embed": {
        "activation": None,
        "class": "linear",
        "from": "prev:output",
        "n_out": 128,
      },
      "s": {
        "L2": 0.0001,
        "class": "rec",
        "dropout": 0.3,
        "from": ["am", "prev_out_embed"],
        "n_out": 128,
        "unit": "nativelstm2",
        "unit_opts": {"rec_weight_dropout": 0.3},
      },
    }

  def add_segmental_decoder(self):
    self.net_dict["output"]["unit"].update({
      "segment_lens": {
        "class": "combine",
        "from": ["segment_lens0", "const1"],
        "is_output_layer": True,
        "kind": "add",
      },
      "segment_lens0": {
        "class": "combine",
        "from": [":i", "segment_starts"],
        "kind": "sub",
      },
      "segment_starts": {
        "class": "switch",
        "condition": "prev:output_emit",
        "false_from": "prev:segment_starts",
        "initial_output": 0,
        "is_output_layer": True,
        "true_from": ":i"}})

  def add_center_window_decoder(self):
    self.net_dict["output"]["unit"].update({
      "segment_ends": {
        "class": "switch",
        "condition": "seq_end_too_far",
        "false_from": "segment_ends1",
        "true_from": "seq_lens",
      },
      "segment_ends1": {
        "class": "eval",
        "eval": "source(0) + source(1) + 4",
        "from": ["segment_starts1", "segment_lens1"],
      },
      "segment_lens": {
        "class": "eval",
        "eval": "source(0) - source(1)",
        "from": ["segment_ends", "segment_starts"],
        "is_output_layer": True,
      },
      "segment_lens0": {
        "class": "combine",
        "from": [":i", "segment_starts1"],
        "kind": "sub",
      },
      "segment_lens1": {
        "class": "combine",
        "from": ["segment_lens0", "const1"],
        "is_output_layer": True,
        "kind": "add",
      },
      "segment_starts": {
        "class": "switch",
        "condition": "seq_start_too_far",
        "false_from": "segment_starts2",
        "true_from": 0,
      },
      "segment_starts1": {
        "class": "switch",
        "condition": "prev:output_emit",
        "false_from": "prev:segment_starts1",
        "initial_output": 0,
        "is_output_layer": True,
        "true_from": ":i",
      },
      "segment_starts2": {
        "class": "eval",
        "eval": "source(0) + source(1) - 4",
        "from": ["segment_starts1", "segment_lens1"],
        "is_output_layer": True,
      },
      "seq_end_too_far": {
        "class": "compare",
        "from": ["segment_ends1", "seq_lens"],
        "kind": "greater",
      },
      "seq_lens": {"class": "length", "from": "base:encoder"},
      "seq_start_too_far": {
        "class": "compare",
        "from": ["segment_starts2"],
        "kind": "less",
        "value": 0}})
