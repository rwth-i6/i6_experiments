import json

from recipe.i6_core.returnn.config import CodeWrapper
from sisyphus.delayed_ops import DelayedFunction
# from returnn.tf.util.data import Dim

def get_alignment_net_dict(pretrain_idx):
  """
  :param int|None pretrain_idx: starts at 0. note that this has a default repetition factor of 6
  :return: net_dict or None if pretrain should stop
  :rtype: dict[str,dict[str]|int]|None
  """
  import numpy
  # Note: epoch0 is 0-based here! I.e. in contrast to elsewhere, where it is 1-based.
  # Also, we never use #repetition here, such that this is correct.
  # This is important because of sub-epochs and storing the HDF files,
  # to know exactly which HDF files cover the dataset completely.
  epoch0 = pretrain_idx
  net_dict = {}

  # network
  # (also defined by num_inputs & num_outputs)
  # EncKeyTotalDim = 200
  # AttNumHeads = 1  # must be 1 for hard-att
  # AttentionDropout = 0.1
  # l2 = 0.0001
  # EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
  # EncValueTotalDim = 2048
  # EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
  # LstmDim = EncValueTotalDim // 2

  if pretrain_idx is not None:
    net_dict["#config"] = {}

    # Do this in the very beginning.
    # lr_warmup = [0.0] * eval("epoch_split")  # first collect alignments with existing model, no training
    lr_warmup = list(numpy.linspace(eval("learning_rate") * 0.1, eval("learning_rate"), num=10))
    if pretrain_idx < len(lr_warmup):
      net_dict["#config"]["learning_rate"] = lr_warmup[pretrain_idx]

  # We import the model, thus no growing.
  start_num_lstm_layers = 2
  final_num_lstm_layers = 6
  num_lstm_layers = final_num_lstm_layers
  if pretrain_idx is not None:
    pretrain_idx = max(pretrain_idx, 0) // 5  # Repeat a bit.
    num_lstm_layers = pretrain_idx + start_num_lstm_layers
    pretrain_idx = num_lstm_layers - final_num_lstm_layers
    num_lstm_layers = min(num_lstm_layers, final_num_lstm_layers)

  if final_num_lstm_layers > start_num_lstm_layers:
    start_dim_factor = 0.5
    grow_frac = 1.0 - float(final_num_lstm_layers - num_lstm_layers) / (final_num_lstm_layers - start_num_lstm_layers)
    dim_frac = start_dim_factor + (1.0 - start_dim_factor) * grow_frac
  else:
    dim_frac = 1.

  time_reduction = [3, 2] if num_lstm_layers >= 3 else [6]

  if pretrain_idx is not None and pretrain_idx <= 1 and "learning_rate" not in net_dict["#config"]:
    # Fixed learning rate for the beginning.
    net_dict["#config"]["learning_rate"] = eval("learning_rate")

  net_dict["#info"] = {
    "epoch0": epoch0,  # Set this here such that a new construction for every pretrain idx is enforced in all cases.
    "num_lstm_layers": num_lstm_layers, "dim_frac": dim_frac, }

  # We use this pretrain construction during the whole training time (epoch0 > num_epochs).
  if pretrain_idx is not None and epoch0 % eval("epoch_split") == 0 and epoch0 > eval("num_epochs"):
    # Stop pretraining now.
    return None

  net_dict.update({
    "source": {
      "class": "eval",
      "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {
      "class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True, "trainable": True},  # (T,40,32)
    "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0"},  # (T,20,32)
    "conv1": {
      "class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True, "trainable": True},  # (T,20,32)
    "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1"},  # (T,10,32)
    "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)

    # Encoder LSTMs added below, resulting in "encoder0".

    "encoder": {"class": "copy", "from": "encoder0"},
    "enc_ctx0": {
      "class": "linear", "from": "encoder", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim, "L2": l2,
      "dropout": 0.2},
    "enc_ctx_win": {"class": "window", "from": "enc_ctx0", "window_size": 5},  # [B,T,W,D]
    "enc_val": {"class": "copy", "from": "encoder"},
    "enc_val_win": {"class": "window", "from": "enc_val", "window_size": 5},  # [B,T,W,D]

    "enc_ctx": {
      "class": "linear", "from": "encoder", "activation": "tanh", "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2},

    "enc_seq_len": {"class": "length", "from": "encoder", "sparse": True},

    # for task "search" / search_output_layer
    "output_wo_b0": {
      "class": "masked_computation", "unit": {"class": "copy"}, "from": "output", "mask": "output/output_emit"},
    "output_wo_b": {"class": "reinterpret_data", "from": "output_wo_b0", "set_sparse_dim": eval("target_num_labels")},
    "decision": {
      "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": eval("target"), 'only_on_search':
      True}, })

  # Add encoder BLSTM stack.
  src = "conv_merged"
  if num_lstm_layers >= 1:
    net_dict.update({
      "lstm0_fw": {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1,
        "from": src, "trainable": True},
      "lstm0_bw": {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1,
        "from": src, "trainable": True}})
    src = ["lstm0_fw", "lstm0_bw"]
  for i in range(1, num_lstm_layers):
    red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
    net_dict.update({
      "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
    src = "lstm%i_pool" % (i - 1)
    net_dict.update({
      "lstm%i_fw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1,
        "from": src, "dropout": 0.3 * dim_frac, "trainable": True}, "lstm%i_bw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1,
        "from": src, "dropout": 0.3 * dim_frac, "trainable": True}})
    src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
  net_dict["encoder0"] = {"class": "copy", "from": src}  # dim: EncValueTotalDim

  # This is used for training.
  net_dict["lm_input0"] = {"class": "copy", "from": "data:%s" % eval("target")}
  net_dict["lm_input1"] = {"class": "prefix_in_time", "from": "lm_input0", "prefix": eval("targetb_blank_idx")}
  net_dict["lm_input"] = {"class": "copy", "from": "lm_input1"}

  def get_output_dict(train, search, targetb, beam_size=eval("beam_size")):
    return {
      "class": "rec", "from": "encoder",  # time-sync
      "include_eos": True, "back_prop": (eval("task") == "train") and train, "unit": {
        "am": {"class": "copy", "from": "data:source"},

        "enc_ctx_win": {"class": "gather_nd", "from": "base:enc_ctx_win", "position": ":i"},  # [B,W,D]
        "enc_val_win": {"class": "gather_nd", "from": "base:enc_val_win", "position": ":i"},  # [B,W,D]
        "att_query": {
          "class": "linear", "from": "am", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim},
        'att_energy': {
          "class": "dot", "red1": "f", "red2": "f", "var1": "static:0", "var2": None,
          "from": ['enc_ctx_win', 'att_query']},  # (B, W)
        'att_weights0': {
          "class": "softmax_over_spatial", "axis": "static:0", "from": 'att_energy',
          "energy_factor": EncKeyPerHeadDim ** -0.5},  # (B, W)
        'att_weights1': {
          "class": "dropout", "dropout_noise_shape": {"*": None}, "from": 'att_weights0',
          "dropout": AttentionDropout},
        "att_weights": {"class": "merge_dims", "from": "att_weights1", "axes": "except_time"},
        'att': {
          "class": "dot", "from": ['att_weights', 'enc_val_win'], "red1": "static:0", "red2": "static:0",
          "var1": None, "var2": "f"},  # (B, V)

        "prev_out_non_blank": {
          "class": "reinterpret_data", "from": "prev:output", "set_sparse_dim": eval("target_num_labels"),
          "set_sparse": True},
        "lm_masked": {
          "class": "masked_computation", "mask": "prev:output_emit", "from": "prev_out_non_blank",  # in decoding
          "masked_from": "base:lm_input" if eval("task") == "train" else None,  # enables optimization if used

          "unit": {
            "class": "subnetwork", "from": "data", "trainable": True, "subnetwork": {
              "input_embed": {
                "class": "linear", "n_out": 256, "activation": "identity", "trainable": True, "L2": l2,
                "from": "data"}, "lstm0": {
                "class": "rec", "unit": "nativelstm2", "dropout": 0.2, "n_out": 1024, "L2": l2, "from": "input_embed",
                "trainable": True}, "output": {"class": "copy", "from": "lstm0"}
            }}},
        "lm": {"class": "copy", "from": "lm_embed_unmask"},  # [B,L]

        # joint network: (W_enc h_{enc,t} + W_pred * h_{pred,u} + b)
        # train : (T-enc, B, F|2048) ; (U+1, B, F|256)
        # search: (B, F|2048) ; (B, F|256)
        "readout_in": {
          "class": "linear", "from": ["am", "att", "lm_masked"], "activation": None, "n_out": 1000, "L2": l2,
          "dropout": 0.2, "out_type": {
            "batch_dim_axis": 2 if eval("task") == "train" else 0,
            "shape": (None, None, 1000) if eval("task") == "train" else (1000,),
            "time_dim_axis": 0 if eval("task") == "train" else None}},  # (T, U+1, B, 1000)

        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},

        "label_log_prob": {
          "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": 0.3,
          "n_out": eval("target_num_labels")},  # (B, T, U+1, 1030)
        "emit_prob0": {"class": "linear", "from": "readout", "activation": None, "n_out": 1, "is_output_layer": True},
        # (B, T, U+1, 1)
        "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
        "blank_log_prob": {"class": "eval", "from": "emit_prob0", "eval": "tf.math.log_sigmoid(-source(0))"},
        # (B, T, U+1, 1)
        "label_emit_log_prob": {"class": "combine", "kind": "add", "from": ["label_log_prob", "emit_log_prob"]},
        # (B, T, U+1, 1), scaling factor in log-space
        "output_log_prob": {"class": "copy", "from": ["blank_log_prob", "label_emit_log_prob"]},  # (B, T, U+1, 1031)

        "output_prob": {
          "class": "eval", "from": ["output_log_prob", "base:data:" + eval("target"), "base:encoder"],
          "eval": eval("rna_loss"), "out_type": eval("rna_loss_out"), "loss": "as_is", },

        # this only works when the loop has been optimized, i.e. log-probs are (B, T, U, V)
        "rna_alignment": {
          "class": "eval", "from": ["output_log_prob", "base:data:" + eval("target"), "base:encoder"],
          "eval": eval("rna_alignment"),
          "out_type": eval("rna_alignment_out"), "is_output_layer": True} if eval("task") == "train"  # (B, T)
        else {"class": "copy", "from": "output_log_prob"},

        # During training   : targetb = "target"  (RNA-loss)
        # During recognition: targetb = "targetb"
        'output': {
          'class': 'choice', 'target': targetb, 'beam_size': beam_size, 'from': "output_log_prob",
          "input_type": "log_prob", "initial_output": 0, "cheating": "exclusive" if eval("task") == "train" else None,
          "explicit_search_sources": ["prev:out_str", "prev:output"] if eval("task") == "search" else None,
          "custom_score_combine": eval("targetb_recomb_recog") if eval("task") == "search" else None},

        "out_str": {
          "class": "eval", "from": ["prev:out_str", "output_emit", "output"], "initial_output": None,
          "out_type": {"shape": (), "dtype": "string"}, "eval": eval("out_str")},

        "output_is_not_blank": {
          "class": "compare", "from": "output", "value": eval("targetb_blank_idx"), "kind": "not_equal",
          "initial_output": True},

        # initial state=True so that we are consistent to the training and the initial state is correctly set.
        "output_emit": {
          "class": "copy", "from": "output_is_not_blank", "initial_output": True, "is_output_layer": True},

        "const0": {"class": "constant", "value": 0, "collocate_with": ["du", "dt"]},
        "const1": {"class": "constant", "value": 1, "collocate_with": ["du", "dt"]},

      },
    }

  if eval("task") == "train":
    net_dict["output"] = eval("get_output_dict(train=True, search=False, targetb=target)")
  else:
    net_dict["output"] = eval("get_output_dict(train=True, search=True, targetb='targetb')")

  if eval("task") in ("train", "forward"):
    net_dict["rna_alignment"] = {"class": "copy", "from": ["output/rna_alignment"]}  # (B, T)

  return net_dict

import numpy as np
def get_extended_net_dict(
  pretrain_idx, learning_rate, num_epochs, enc_val_dec_factor,
  target_num_labels, targetb_num_labels, targetb_blank_idx, target, task, scheduled_sampling, lstm_dim,
  l2, beam_size, length_model_inputs, prev_att_in_state, use_att, prev_target_in_readout,
  label_smoothing, emit_loss_scale, efficient_loss, emit_extra_loss, time_reduction, ctx_size="inf",
  fast_rec=False, fast_rec_full=False, sep_sil_model=None, sil_idx=None, sos_idx=0, direct_softmax=False,
  label_dep_length_model=False, search_use_recomb=True, feature_stddev=None, dump_align=False,
  label_dep_means=None, max_seg_len=None, hybrid_hmm_like_label_model=False, length_model_focal_loss=2.0,
  label_model_focal_loss=2.0):

  assert ctx_size == "inf" or type(ctx_size) == int
  assert not sep_sil_model or sil_idx == 0  # assume in order to construct output_prob vector (concat sil_prob, label_prob, blank_prob)
  assert not label_dep_length_model or (label_dep_means is not None and max_seg_len is not None)
  assert task != "train" or not label_dep_length_model

  net_dict = {"#info": {"lstm_dim": lstm_dim, "l2": l2, "learning_rate": learning_rate, "time_red": time_reduction}}
  net_dict.update({
    "source": {
      "class": "eval",
      "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

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

  if hybrid_hmm_like_label_model:
    net_dict.update({
      "label_log_prob_framewise": {
        "class": "linear", "n_out": target_num_labels if not sep_sil_model else target_num_labels - 1, "dropout": 0.3,
        "activation": "log_softmax", "from": "encoder"},
    })

  if task == "train":
    net_dict.update({
      # "cur_label": {"class": "gather", "from": "data:targetb", "axis": "t", "position": ":i"},
      "existing_alignment": {
        "class": "reinterpret_data", "from": "data:alignment", "set_sparse": True,  # not sure what the HDF gives us
        "set_sparse_dim": targetb_num_labels, "size_base": "encoder",  # for RNA...
      },
      "is_label": {"class": "compare", "from": "existing_alignment", "value": targetb_blank_idx, "kind": "not_equal"},
      "segment_starts_masked": {
        "class": "masked_computation", "from": "output/segment_starts", "mask": "is_label",
        "register_as_extern_data": "segment_starts_masked",
        "out_spatial_dim": CodeWrapper('Dim(kind=Dim.Types.Spatial, description="label-axis")'), "unit": {
          "class": "copy", "from": "data"}},
      "segment_lens_masked": {
        "class": "masked_computation", "from": "output/segment_lens", "mask": "is_label",
        "register_as_extern_data": "segment_lens_masked",
        "out_spatial_dim": CodeWrapper('Dim(kind=Dim.Types.Spatial, description="label-axis")'),
        "unit": {
          "class": "copy", "from": "data"}},
      "label_ground_truth_masked0": {
        "class": "masked_computation", "from": "existing_alignment", "mask": "is_label",
        "unit": {
          "class": "copy", "from": "data"
        }
      },
      "label_ground_truth_masked": {
        "class": "reinterpret_data", "from": "label_ground_truth_masked0", "enforce_batch_major": True,
        "register_as_extern_data": "label_ground_truth", "set_sparse_dim": target_num_labels,},
      "const0": {"class": "constant", "value": 0, "with_batch_dim": True},
      "const1": {"class": "constant", "value": 1, "with_batch_dim": True},
      "emit_ground_truth0": {
        "class": "switch", "condition": "is_label", "true_from": "const1", "false_from": "const0"
      },
      "emit_ground_truth": {
        "class": "reinterpret_data", "from": "emit_ground_truth0", "set_sparse": True, "set_sparse_dim": 2,
        "register_as_extern_data": "emit_ground_truth", "is_output_layer": True},

      "labels_with_blank_ground_truth": {
        "class": "copy", "from": "existing_alignment",
        "register_as_extern_data": "targetb" if task == "train" else None},

      "ctc_out": {"class": "softmax", "from": "encoder", "with_bias": False, "n_out": targetb_num_labels},
      "ctc_out_scores": {
        "class": "eval", "from": ["ctc_out"], "eval": "safe_log(source(0))", },

      "ctc": {
        "class": "copy", "from": "ctc_out_scores", "loss": "ctc" if task == "train" else None,
        "target": "label_ground_truth" if task == "train" else None, "loss_opts": {
          "beam_width": 1, "use_native": True, "output_in_log_space": True,
          "ctc_opts": {"logits_normalize": False}} if task == "train" else None}
    })
  elif task == "search":
    net_dict.update({
      "output_non_sil": {
        "class": "compare", "from": "output", "value": sil_idx if sil_idx is not None else -1,
        "kind": "not_equal"},
      "output_non_blank": {"class": "compare", "from": "output", "value": targetb_blank_idx, "kind": "not_equal"},
      "output_non_sil_non_blank": {
        "class": "combine", "kind": "logical_and", "from": ["output_non_sil", "output_non_blank"],
        "is_output_layer": True},
      "output_wo_b0": {
        "class": "masked_computation", "unit": {"class": "copy"}, "from": "output",
        "mask": "output_non_sil_non_blank"},
      "output_wo_b": {
        "class": "reinterpret_data", "from": "output_wo_b0",
        "set_sparse_dim": target_num_labels},
      "decision": {
        "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": target, 'only_on_search': True}
    })

  # Add encoder BLSTM stack.
  src = "conv_merged"
  num_lstm_layers = 6
  # time_reduction = [3, 2]
  if num_lstm_layers >= 1:
    net_dict.update({
      "lstm0_fw": {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": 1,
        "from": src, "trainable": True}, "lstm0_bw": {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": -1,
        "from": src, "trainable": True}})
    src = ["lstm0_fw", "lstm0_bw"]
  for i in range(1, num_lstm_layers):
    red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
    net_dict.update({
      "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
    src = "lstm%i_pool" % (i - 1)
    net_dict.update({
      "lstm%i_fw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": 1,
        "from": src, "dropout": 0.3, "trainable": True}, "lstm%i_bw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": -1,
        "from": src, "dropout": 0.3, "trainable": True}})
    src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
  net_dict["encoder0"] = {"class": "copy", "from": src}  # dim: EncValueTotalDim

  def get_fast_network_dict(rec, rec_full):
    if not rec:
      return {
        "class": "linear", "activation": "tanh", "n_out": 128, "dropout": 0.3, "L2": l2, "from": length_model_inputs}
    else:
      if not rec_full:
        """
        Use an LSTM which is reset, if the last frame was a segment boundary, i.e. it only uses recurrency WITHIN
        the segments
        """
        # assert type(ctx_size) == int and rec
        return {
          "class": "subnetwork", "from": length_model_inputs, "subnetwork": {
            # the input is dependent on the previous hidden state (output), if the last frame was not a segment boundary
            # otherwise, the previous hidden state is set to 0, i.e. has no impact on the following calculations
            "prev_out_rec": {"class": "copy", "from": "prev:output"},
            "prev_out_reset": {"class": "eval", "from": "prev:output", "eval": "source(0) * 0"},
            "input_reset": {"class": "copy", "from": ["prev_out_reset", "data:source"]},
            "input_rec": {"class": "copy", "from": ["prev_out_rec", "data:source"]},
            "input": {
              "class": "switch", "condition": "base:prev:output_emit", "true_from": "input_reset",
              "false_from": "input_rec"},

            # the different gates of the LSTM all perform calculations on 'input'
            "input_gate": {"class": "linear", "from": "input", "activation": "sigmoid", "n_out": 128},
            "forget_gate": {"class": "linear", "from": "input", "activation": "sigmoid", "n_out": 128},
            "output_gate": {"class": "linear", "from": "input", "activation": "sigmoid", "n_out": 128},
            "cell_in": {"class": "linear", "from": "input", "activation": "tanh", "n_out": 128},

            # the cell state is dependent on the previous cell state and the input,
            # if the last frame was not a segment boundary. otherwise, it is only dependent on the input
            "c_rec": {
              "class": "eval", "from": ["input_gate", "cell_in", "forget_gate", "prev:c"],
              "eval": "source(0) * source(1) + source(2) * source(3)"},
            "c_reset": {
              "class": "eval", "from": ["input_gate", "cell_in"],
              "eval": "source(0) * source(1)"},
            "c": {
              "class": "switch", "condition": "base:prev:output_emit", "true_from": "c_reset",
              "false_from": "c_rec"},

            # the output is dependent on the input and the cell state, both of which are only recurrent if the
            # last frame was a segment boundary
            "output": {
              "class": "eval", "from": ["output_gate", "c"], "eval": "source(0) * source(1)"},
          }
        }
      else:
        return {
          "class": "rec", "unit": "nativelstm2", "from": length_model_inputs, "n_out": 128, "L2": l2,
          "dropout": 0.3, "unit_opts": {"rec_weight_dropout": 0.3}}

  def get_label_model(net_dict):
    """Readout (per segment): linear layer which is used as input for the label model"""
    # during training, we mask the readout layer to save time and memory

    rec_unit_dict = {}

    # this model uses no label context but only information from the encoder
    if hybrid_hmm_like_label_model:
      seg_time_tag = CodeWrapper('Dim(kind=Dim.Types.Spatial, description="att_t")')
      rec_unit_dict.update({
        "label_log_prob_framewise_segments": {  # [B,t_sliced,D]
          "class": "slice_nd", "from": "base:label_log_prob_framewise", "start": "segment_starts",
          "size": "segment_lens", "out_spatial_dim": seg_time_tag},
        "label_log_prob0": {
          "class": "reduce", "from": "label_log_prob_framewise_segments", "mode": "sum",
          "axis": seg_time_tag},
        "label_log_prob": {
          "class": "activation", "activation": "log_softmax", "from": ["label_log_prob0"]
        }

      })
    else:
      rec_unit_dict.update({
        "prev_out_non_blank": {
          "class": "reinterpret_data", "from": "prev:output", "set_sparse_dim": targetb_num_labels, "set_sparse": True},
        "prev_non_blank_embed": {
          "class": "linear", "activation": None, "with_bias": False, "from": "prev_out_non_blank", "n_out": 621},
        "readout_in": {
          "class": "linear", "from": ["lm", "att"] if use_att else ["lm", "am"], "activation": None, "n_out": 1000},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},
        # the following two get updated later if there is a separate silence model
        "label_log_prob0": {
          "class": "linear", "from": "readout",
          "activation": "log_softmax" if not direct_softmax else None,
          "dropout": 0.3,
          "n_out": target_num_labels},
        "label_log_prob": {
          "class": "combine", "kind": "add",
          "from": ["label_log_prob0"]}})

      if prev_target_in_readout:
        rec_unit_dict["readout_in"]["from"].append("prev_non_blank_embed")

      lm_dict = {
        "input_embed0": {
          "class": "window", "window_size": 1, "window_left": 0,
          "window_right": 0} if ctx_size == "inf" else {
          "class": "window", "window_size": ctx_size, "window_right": 0, "window_left": ctx_size - 1},
        "input_embed": {
          "class": "merge_dims", "from": "input_embed0", "axes": "except_time"},
        "lm": {
          "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim,
          "from": ["prev:att", "input_embed"] if prev_att_in_state else "input_embed"} if ctx_size == "inf" else {
          "class": "linear", "activation": "tanh", "n_out": lstm_dim,
          "from": ["prev:att", "input_embed"] if prev_att_in_state else "input_embed"},
      }

      if sep_sil_model is not None:
        raise NotImplementedError
        assert sep_sil_model in ["mean", "min"]
        rec_unit_dict.update({
          "pool_segments": {
            "class": "copy", "from": "segments"},
          "pooled_segment": {
            "class": "reduce", "mode": sep_sil_model, "axes": ["stag:att_t"], "from": "pool_segments"},
          "sil_model": {
            "class": "linear", "activation": "tanh", "n_out": 128, "dropout": 0.3, "L2": l2, "from": ["pooled_segment"]},
          "sil_log_prob0": {"class": "linear", "from": "sil_model", "activation": None, "n_out": 1},
          "sil_log_prob1": {"class": "activation", "from": "sil_log_prob0", "activation": "log_sigmoid"},
          "non_sil_log_prob": {"class": "eval", "from": "sil_log_prob0", "eval": "tf.math.log_sigmoid(-source(0))"},
          # prob of sil/non-sil is combined with emit prob
          # if we have a label dep length model, we first concat the sil log prob with the label log prob and then
          # add the emit log prob because there is a emit prob for every label
          "sil_log_prob": {
            "class": "combine", "kind": "add",
            "from": ["sil_log_prob1"] if task == "train" or label_dep_length_model else ["sil_log_prob1", "emit_log_prob"]}})
        # update original log prob by adding the separate silence model
        rec_unit_dict["label_log_prob0"]["n_out"] = target_num_labels - 1
        rec_unit_dict["label_log_prob"]["from"].append("non_sil_log_prob")

      if task == "train":
        # rec_unit_dict.update({
        #   "output": {
        #     'class': 'choice', 'target': "label_ground_truth", 'beam_size': beam_size, 'from': "data",
        #     "initial_output": sos_idx, "cheating": "exclusive" if task == "train" else None}, })
        lm_dict["input_embed0"]["from"] = "prev_non_blank_embed"
        rec_unit_dict.update(lm_dict)

        rec_unit_dict["input_embed0"]["name_scope"] = "lm_masked/input_embed0"
        rec_unit_dict["input_embed"]["name_scope"] = "lm_masked/input_embed"
        rec_unit_dict["lm"]["name_scope"] = "lm_masked/lm"

        net_dict.update({
          "label_model": {
            "class": "rec", "from": "data:label_ground_truth", "include_eos": True, "back_prop": True,
            "name_scope": "output/rec", "is_output_layer": True, "unit": {}}})
        net_dict["label_model"]["unit"].update(rec_unit_dict)
      else:
        lm_dict["input_embed0"]["from"] = "data"
        lm_dict["lm"]["name_scope"] = "lm"
        if prev_att_in_state:
          lm_dict["lm"]["from"][0] = "base:prev:att"
        # in case of search, emit log prob needs to be added to the label log prob
        net_dict["output"]["unit"].update({
          "lm_masked": {
            "class": "masked_computation", "from": "prev_non_blank_embed", "mask": "prev:output_emit", "unit": {
              "class": "subnetwork", "from": "data", "subnetwork": {
                **lm_dict, "output": {"class": "copy", "from": "lm"}}}},
          "lm": {"class": "unmask", "from": "lm_masked", "mask": "prev:output_emit"}})

    # here, we add general components which are needed for train or search
    if task == "train":
      rec_unit_dict.update({
        "segment_lens": {
          "class": "gather", "from": "base:data:segment_lens_masked", "position": ":i", "axis": "t"},
        "segment_starts": {
          "class": "gather", "from": "base:data:segment_starts_masked", "position": ":i", "axis": "t"},
        "label_prob": {
          "class": "activation",
          "from": "label_log_prob" if not sep_sil_model else ["sil_log_prob", "label_log_prob"],
          "is_output_layer": True, "activation": "exp" if not direct_softmax else "softmax",
          "target": "label_ground_truth",
          "loss": "ce" if task == "train" and not efficient_loss else None,
          "loss_opts": {"focal_loss_factor": label_model_focal_loss, "label_smoothing": 0.1}},
        "output": {
          'class': 'choice', 'target': "label_ground_truth", 'beam_size': beam_size, 'from': "data",
          "initial_output": sos_idx, "cheating": "exclusive" if task == "train" else None}
      })
      net_dict.update({
        "label_model": {
          "class": "rec", "from": "data:label_ground_truth", "include_eos": True, "back_prop": True,
          "name_scope": "output/rec", "is_output_layer": True, "unit": {}}})
      net_dict["label_model"]["unit"].update(rec_unit_dict)
    else:
      # in case of search, emit log prob needs to be added to the label log prob
      # if there is a label dep length model, we first concat the silence log prob with the label log prob
      # and then add the emit log prob
      if sep_sil_model and label_dep_length_model:
        rec_unit_dict.update({
          "label_log_prob1": {
            "class": "combine", "from": ["label_log_prob0", "non_sil_log_prob"], "kind": "add", },
          "label_log_prob2": {
            "class": "copy", "from": ["sil_log_prob", "label_log_prob1"], },
          "label_log_prob": {
            "class": "combine", "from": ["label_log_prob2", "emit_log_prob"], "kind": "add", }
        })
      else:
        rec_unit_dict["label_log_prob"]["from"].append("emit_log_prob")
      net_dict["output"]["unit"].update(rec_unit_dict)

    return net_dict

  def get_length_model(net_dict):
    if "am" in length_model_inputs:
      net_dict["output"]["unit"].update({
        "am": {"class": "copy", "from": "data:source"}
      })
    if "prev_out_embed" in length_model_inputs:
      net_dict["output"]["unit"].update({
        "prev_out_embed": {"class": "linear", "from": "prev:output", "activation": None, "n_out": 128}
      })
    if "prev_out_is_non_blank" in length_model_inputs:
      net_dict["output"]["unit"].update({
        "const0.0_0": {"class": "constant", "value": 0.0, "with_batch_dim": True},
        "const0.0": {"class": "expand_dims", "axis": "F", "from": "const0.0_0"},
        "const1.0_0": {"class": "constant", "value": 1.0, "with_batch_dim": True},
        "const1.0": {"class": "expand_dims", "axis": "F", "from": "const1.0_0"},
        "2d_emb0": {"class": "copy", "from": ["const1.0", "const0.0"]},
        "2d_emb1": {"class": "copy", "from": ["const0.0", "const1.0"]},
        # 1, if prev frame was segment boundary; 0, otherwise
        # basically represents Kronecker delta
        "prev_out_is_non_blank": {
          "class": "switch", "condition": "prev:output_emit", "true_from": "2d_emb1", "false_from": "2d_emb0"}
      })
    if label_dep_length_model:
      assert label_dep_means is not None
      assert max_seg_len is not None

      net_dict.update({
        "max_seg_len_range0": {"class": "range", "limit": max_seg_len + 1, "start": 1, "delta": 1},
        "max_seg_len_range": {"class": "cast", "from": "max_seg_len_range0", "dtype": "float32"},
        "mean_seg_lens0": {
          "class": "constant", "value": CodeWrapper("np.array(%s)" % label_dep_means), "with_batch_dim": True},
        "mean_seg_lens": {"class": "cast", "dtype": "float32", "from": "mean_seg_lens0"},
        "length_model_norm0": {
          "class": "eval", "from": ["mean_seg_lens", "max_seg_len_range"],
          "eval": "tf.math.exp(-tf.math.abs(source(0) - source(1)))"},
        "length_model_norm": {
          "class": "reduce", "from": ["length_model_norm0"], "mode": "sum", "axes": ["stag:max_seg_len_range0:range"]},
      })
      net_dict["output"]["unit"].update({
        "const0.0_0": {"class": "constant", "value": 0.0, "with_batch_dim": True},
        "const0.0": {"class": "expand_dims", "axis": "F", "from": "const0.0_0"},
        "seg_lens_float": {"class": "cast", "dtype": "float32", "from": "segment_lens"},
        "length_model0": {
          "class": "eval", "from": ["seg_lens_float", "base:mean_seg_lens"],
          "eval": "tf.math.exp(-tf.math.abs(source(1) - source(0)))"},
        "length_model": {
          "class": "combine", "from": ["length_model0", "base:length_model_norm"], "kind": "truediv"},
        "emit_log_prob": {
          "class": "eval", "from": "length_model", "eval": "tf.math.log(source(0))"
        },
        "max_seg_len": {
          "class": "constant", "value": max_seg_len-1},
        "is_segment_longer_than_max": {
          "class": "compare", "from": ["segment_lens", "max_seg_len"], "kind": "greater"},
        'const_neg_inf': {'axis': 'F', 'class': 'expand_dims', 'from': 'const_neg_inf_0'},
        'const_neg_inf_0': {'class': 'constant', 'value': CodeWrapper("float('-inf')"), 'with_batch_dim': True},
        "blank_log_prob": {
          "class": "switch", "condition": "is_segment_longer_than_max", "true_from": "const_neg_inf",
          "false_from": "const0.0"
        }
      })
    else:
      net_dict["output"]["unit"].update({
        "s": get_fast_network_dict(rec=fast_rec, rec_full=fast_rec_full),
        "emit_prob0": {"class": "linear", "from": "s", "activation": None, "n_out": 1, "is_output_layer": True},
        "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},
        "blank_log_prob": {"class": "eval", "from": "emit_prob0", "eval": "tf.math.log_sigmoid(-source(0))"}})

    if task == "train":
      net_dict["output"]["unit"].update({
        "emit_blank_log_prob": {"class": "copy", "from": ["blank_log_prob", "emit_log_prob"]},
        "emit_blank_prob": {
          "class": "activation", "from": "emit_blank_log_prob",
          "activation": "exp", "target": "emit_ground_truth",
          "loss": "ce", "loss_opts": {"focal_loss_factor": length_model_focal_loss}}
      })

    return net_dict

  def get_output_dict():
    output_dict = {
      "class": "rec", "from": "encoder", "include_eos": True, "back_prop": (task == "train"),
      "target": "targetb", "size_target": "targetb" if task == "train" else None, "unit": {
        'output': {
          'class': 'choice', 'target': "targetb", 'beam_size': beam_size,
          'from': "data" if task == "train" else "output_log_prob",
          "input_type": "log_prob",
          "initial_output": sos_idx, "cheating": "exclusive" if task == "train" else None},
        "output_emit": {
          "class": "compare", "from": "output", "kind": "not_equal", "value": targetb_blank_idx, "initial_output": True}
      }
    }

    if task != "train":
      output_dict["unit"].update({
        # in case of a label dep length model, the silence log prob is earlier concatenated with the label log prob
        "output_log_prob": {
          "class": "copy",
          "from": ["label_log_prob", "blank_log_prob"] if not sep_sil_model or label_dep_length_model else [
            "sil_log_prob", "label_log_prob", "blank_log_prob"
          ]}})

    if task == "search" and search_use_recomb:
      output_dict["unit"]["output"]["custom_score_combine"] = CodeWrapper("targetb_recomb_recog")
      output_dict["unit"]["output"]["explicit_search_sources"] = ["prev:out_str", "prev:output"]
      output_dict["unit"].update({
        "out_str": {
          "class": "eval", "eval": "self.network.get_config().typed_value('out_str')(source, network=self.network)",
          "from": ["prev:out_str", "output_emit", "output"], "initial_output": None,
          "out_type": {"dtype": "string", "shape": ()}, },
      })

    return output_dict

  net_dict["output"] = get_output_dict()

  net_dict = get_label_model(net_dict=net_dict)
  net_dict = get_length_model(net_dict)
  # net_dict["output"]["unit"].update(get_ce_loss(targetb="targetb"))
  # if with_silence:
  #   net_dict["output"]["unit"].update({
  #     "output_is_sil": {
  #       "class": "compare", "from": "output_", "value": sil_idx, "kind": "equal", "initial_output": False},
  #     "output_is_not_sil": {
  #       "class": "compare", "from": "output_", "value": sil_idx, "kind": "not_equal", "initial_output": True},
  #     "output_is_non_sil_label": {
  #       "class": "combine", "from": ["output_is_not_blank", "output_is_not_sil"], "kind": "logical_and",
  #       "is_output_layer": True if task == "train" else False},
  #   })
    # if task == "search":
    #   net_dict.update({
    #     "output_is_not_sil": {
    #       "class": "compare", "from": "output_wo_b", "value": sil_idx, "kind": "not_equal"},
    #     "output_wo_sil": {
    #       "class": "masked_computation", "unit": {"class": "copy"}, "from": "output_wo_b", "mask": "output_is_not_sil"},
    #   })
    #   net_dict["decision"]["from"] = "output_wo_sil"

  if feature_stddev is not None:
    assert type(feature_stddev) == float
    net_dict["source_stddev"] = {
      "class": "eval", "from": "data", "eval": "source(0) / " + str(feature_stddev)
    }
    net_dict["source"]["from"] = "source_stddev"

  if dump_align:
    net_dict.update({
      "decision_w_b": {
        "class": "decide", "from": "output", "loss": "edit_distance", "target": "targetb", 'only_on_search': True},
      "dump_output": {
        "class": "hdf_dump", "from": "decision_w_b", "filename": "search_out_seqs", "is_output_layer": True}})

  if task == "search":
    net_dict["output"]["unit"]["output"]["length_normalization"] = False

  # TODO: what to do during "retrain" when pretraining is disabled by default
  if scheduled_sampling and task == "train":
    if pretrain_idx is None:
      net_dict["output"]["unit"]["output"]["scheduled_sampling"] = {"gold_mixin_prob": scheduled_sampling}
    elif pretrain_idx < 36:
      net_dict["output"]["unit"]["output"]["scheduled_sampling"] = False
    else:
      net_dict["output"]["unit"]["output"]["scheduled_sampling"] = {
        "gold_mixin_prob": max(scheduled_sampling, 0.9**pretrain_idx)
      }

  if emit_extra_loss:
    net_dict.update({
      "4_target_masked": {
        "class": "copy", "from": "_target_masked", "register_as_extern_data": "targetb_masked_1031"},
      "output_prob_non_blank": {
        "class": "masked_computation", "mask": "output/output_emit", "from": "output/output_prob",
        "unit": {"class": "copy", "from": ["data"]}},
      "output_loss_non_blank": {
        "class": "copy", "from": "output_prob_non_blank",
        "target": "targetb_masked_1031" if task == "train" else None,
        "loss": "ce" if task == "train" else None,
        "loss_opts": {"focal_loss_factor": 2.0, "scale": emit_extra_loss}},
    })

  return net_dict


def custom_construction_algo(idx, net_dict):
  if idx > 30:
    return None
  net_dict["#config"] = {}
  if idx is not None:
    # learning rate warm up
    lr_warmup = list(
      np.linspace(net_dict["#info"]["learning_rate"] * 0.1, net_dict["#info"]["learning_rate"], num=10))
    if idx < len(lr_warmup):
      net_dict["#config"]["learning_rate"] = lr_warmup[idx]

  # encoder construction
  start_num_lstm_layers = 2
  final_num_lstm_layers = 6
  num_lstm_layers = final_num_lstm_layers
  if idx is not None:
    idx = max(idx, 0) // 6  # each index is used 6 times
    num_lstm_layers = idx + start_num_lstm_layers  # 2, 3, 4, 5, 6 (each for 6 epochs)
    idx = num_lstm_layers - final_num_lstm_layers
    num_lstm_layers = min(num_lstm_layers, final_num_lstm_layers)

  if final_num_lstm_layers > start_num_lstm_layers:
    start_dim_factor = 0.5
    # grow_frac values: 0, 1/4, 1/2, 3/4, 1
    grow_frac = 1.0 - float(final_num_lstm_layers - num_lstm_layers) / (final_num_lstm_layers - start_num_lstm_layers)
    # dim_frac values: 0.5, 5/8, 3/4, 7/8, 1
    dim_frac = start_dim_factor + (1.0 - start_dim_factor) * grow_frac
  else:
    dim_frac = 1.
  net_dict["#info"].update({
    "dim_frac": dim_frac, "num_lstm_layers": num_lstm_layers, "pretrain_idx": idx
  })

  time_reduction = net_dict["#info"]["time_red"] if num_lstm_layers >= 3 else [int(np.prod(net_dict["#info"]["time_red"]))]

  # Add encoder BLSTM stack
  src = "conv_merged"
  lstm_dim = net_dict["#info"]["lstm_dim"]
  l2 = net_dict["#info"]["l2"]
  if num_lstm_layers >= 1:
    net_dict.update({
      "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": int(lstm_dim * dim_frac), "L2": l2, "direction": 1,
        "from": src, "trainable": True},
      "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": int(lstm_dim * dim_frac), "L2": l2, "direction": -1,
        "from": src, "trainable": True}})
    src = ["lstm0_fw", "lstm0_bw"]
  for i in range(1, num_lstm_layers):
    red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
    net_dict.update(
      {"lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
    src = "lstm%i_pool" % (i - 1)
    net_dict.update({
      "lstm%i_fw" % i: {"class": "rec", "unit": "nativelstm2", "n_out": int(lstm_dim * dim_frac), "L2": l2,
        "direction": 1, "from": src, "dropout": 0.3 * dim_frac, "trainable": True},
      "lstm%i_bw" % i: {"class": "rec", "unit": "nativelstm2", "n_out": int(lstm_dim * dim_frac), "L2": l2,
        "direction": -1, "from": src, "dropout": 0.3 * dim_frac, "trainable": True}})
    src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
  net_dict["encoder0"] = {"class": "copy", "from": src}  # dim: EncValueTotalDim

  # if necessary, include dim_frac in the attention values
  # TODO check if this is working
  try:
    if net_dict["output"]["unit"]["att_val"]["class"] == "linear":
      net_dict["output"]["unit"]["att_val"]["n_out"] = 2 * int(lstm_dim * dim_frac)
  except KeyError:
    pass

  net_dict["label_model"]["unit"]["label_prob"]["loss_opts"]["label_smoothing"] = 0

  return net_dict

def custom_construction_algo_alignment(idx, net_dict):
  # For debugging, use: python3 ./crnn/Pretrain.py config...
  net_dict = get_net_dict(pretrain_idx=idx)
  return net_dict


def new_custom_construction_algo(idx, net_dict):
  # For debugging, use: python3 ./crnn/Pretrain.py config... Maybe set repetitions=1 below.
  StartNumLayers = 2
  InitialDimFactor = 0.5
  orig_num_lstm_layers = 0
  while "lstm%i_fw" % orig_num_lstm_layers in net_dict:
    orig_num_lstm_layers += 1
  assert orig_num_lstm_layers >= 2
  orig_red_factor = 1
  for i in range(orig_num_lstm_layers - 1):
    orig_red_factor *= net_dict["lstm%i_pool" % i]["pool_size"][0]
  net_dict["#config"] = {}
  if idx < 4:
    # net_dict["#config"]["batch_size"] = 15000
    net_dict["#config"]["accum_grad_multiple_step"] = 4
  idx = max(idx - 4, 0)  # repeat first
  num_lstm_layers = idx + StartNumLayers  # idx starts at 0. start with N layers
  if num_lstm_layers > orig_num_lstm_layers:
    # Finish. This will also use label-smoothing then.
    return None
  if num_lstm_layers == 2:
    net_dict["lstm0_pool"]["pool_size"] = (orig_red_factor,)
  # Skip to num layers.
  net_dict["encoder"]["from"] = ["lstm%i_fw" % (num_lstm_layers - 1), "lstm%i_bw" % (num_lstm_layers - 1)]
  # Delete non-used lstm layers. This is not explicitly necessary but maybe nicer.
  for i in range(num_lstm_layers, orig_num_lstm_layers):
    del net_dict["lstm%i_fw" % i]
    del net_dict["lstm%i_bw" % i]
    del net_dict["lstm%i_pool" % (i - 1)]
  # Thus we have layers 0 .. (num_lstm_layers - 1).
  layer_idxs = list(range(0, num_lstm_layers))
  layers = ["lstm%i_fw" % i for i in layer_idxs] + ["lstm%i_bw" % i for i in layer_idxs]
  grow_frac = 1.0 - float(orig_num_lstm_layers - num_lstm_layers) / (orig_num_lstm_layers - StartNumLayers)
  dim_frac = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac
  for layer in layers:
    net_dict[layer]["n_out"] = int(net_dict[layer]["n_out"] * dim_frac)
    if "dropout" in net_dict[layer]:
      net_dict[layer]["dropout"] *= dim_frac
  # Use label smoothing only at the very end.
  net_dict["label_model"]["unit"]["label_prob"]["loss_opts"]["label_smoothing"] = 0
  return net_dict
