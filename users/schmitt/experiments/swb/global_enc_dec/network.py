def get_net_dict(
  lstm_dim, att_num_heads, att_key_dim, beam_size, sos_idx, l2, learning_rate, time_red, feature_stddev, target, task):
  att_key_per_head_dim = att_key_dim // att_num_heads
  net_dict = {"#info": {"lstm_dim": lstm_dim, "l2": l2, "learning_rate": learning_rate, "time_red": time_red}}
  net_dict.update({
    "source": {
      "class": "eval",
      "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {
      "class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True, "auto_use_channel_first": False},  # (T,40,32)
    "conv0p": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0",
      "use_channel_first": False},  # (T,20,32)
    "conv1": {
      "class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True, "auto_use_channel_first": False},  # (T,20,32)
    "conv1p": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1",
      "use_channel_first": False},  # (T,10,32)
    "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"}})

  # Add encoder BLSTM stack.
  src = "conv_merged"
  num_lstm_layers = 6
  # time_reduction = [3, 2]
  if num_lstm_layers >= 1:
    net_dict.update({
      "lstm0_fw": {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": 1, "from": src,
        "trainable": True}, "lstm0_bw": {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": -1, "from": src,
        "trainable": True}})
    src = ["lstm0_fw", "lstm0_bw"]
  for i in range(1, num_lstm_layers):
    red = time_red[i - 1] if (i - 1) < len(time_red) else 1
    net_dict.update({
      "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
    src = "lstm%i_pool" % (i - 1)
    net_dict.update({
      "lstm%i_fw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": 1, "from": src, "dropout": 0.3,
        "trainable": True}, "lstm%i_bw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": -1, "from": src,
        "dropout": 0.3, "trainable": True}})
    src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
  net_dict["encoder"] = {"class": "copy", "from": src}  # dim: EncValueTotalDim

  net_dict.update({
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"], "n_out": att_key_dim},
    # preprocessed_attended in Blocks
    "inv_fertility": {
      "class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": att_num_heads},
    "enc_value": {"class": "split_dims", "axis": "F", "dims": (att_num_heads, -1), "from": ["encoder"]},
    # (B, enc-T, H, D'/H)

    "output": {
      "class": "rec", "from": [], "unit": {
        'output': {
          'class': 'choice', 'target': target, 'beam_size': beam_size, 'from': ["output_prob"],
          "initial_output": sos_idx},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {
          'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 621,
          "initial_output": sos_idx},  # feedback_input
        "weight_feedback": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
          "n_out": att_key_dim},
        "s_transformed": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["s"], "n_out": att_key_dim},
        "energy_in": {
          "class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "s_transformed"],
          "n_out": att_key_dim}, "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": att_num_heads},
        # (B, enc-T, H)
        "att_weights": {
          "class": "softmax_over_spatial", "from": ["energy"], "energy_factor": att_key_per_head_dim ** -0.5},
        "accum_att_weights": {
          "class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
          "eval": "source(0) + source(1) * source(2) * 0.5",
          "out_type": {"dim": att_num_heads, "shape": (None, att_num_heads)}},
        "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
        "att": {"class": "merge_dims", "axes": "except_batch", "from": ["att0"]},  # (B, H*V)
        "s": {"class": "rec", "unit": "nativelstm2", "from": ["prev:target_embed", "prev:att"], "n_out": lstm_dim},
        # transform
        "readout_in": {"class": "linear", "from": ["s", "att"], "activation": None, "n_out": 1000},
        # merge + post_merge bias
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {
          "class": "softmax", "from": ["readout"], "dropout": 0.3, "target": target, "loss": "ce",
          "loss_opts": {"focal_loss_factor": 2.0, "label_smoothing": 0.1}}}, "target": target},

    "decision": {
      "class": "decide", "from": ["output"], "loss": "edit_distance", "target": target, "loss_opts": {
      }
    }
  })

  if task != "eval":
    net_dict["output"]["max_seq_len"] = "max_len_from('base:encoder')"

  return net_dict


def custom_construction_algo(idx, net_dict):
  import numpy as np

  if idx > 30:
    return None

  net_dict["#config"] = {}
  if idx is not None:
    # learning rate warm up
    lr_warmup = list(np.linspace(net_dict["#info"]["learning_rate"] * 0.1, net_dict["#info"]["learning_rate"], num=10))
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
      "dim_frac": dim_frac, "num_lstm_layers": num_lstm_layers, "pretrain_idx": idx})

    time_reduction = net_dict["#info"]["time_red"] if num_lstm_layers >= 3 else [int(np.prod(net_dict["#info"]["time_red"]))]

    # Add encoder BLSTM stack
    src = "conv_merged"
    lstm_dim = net_dict["#info"]["lstm_dim"]
    l2 = net_dict["#info"]["l2"]
    if num_lstm_layers >= 1:
      net_dict.update({
        "lstm0_fw": {
          "class": "rec", "unit": "nativelstm2", "n_out": int(lstm_dim * dim_frac), "L2": l2, "direction": 1,
          "from": src, "trainable": True},
        "lstm0_bw": {
          "class": "rec", "unit": "nativelstm2", "n_out": int(lstm_dim * dim_frac), "L2": l2, "direction": -1,
          "from": src, "trainable": True}})
      src = ["lstm0_fw", "lstm0_bw"]
    for i in range(1, num_lstm_layers):
      red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
      net_dict.update({
        "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
      src = "lstm%i_pool" % (i - 1)
      net_dict.update({
        "lstm%i_fw" % i: {
          "class": "rec", "unit": "nativelstm2", "n_out": int(lstm_dim * dim_frac), "L2": l2, "direction": 1,
          "from": src, "dropout": 0.3 * dim_frac, "trainable": True}, "lstm%i_bw" % i: {
          "class": "rec", "unit": "nativelstm2", "n_out": int(lstm_dim * dim_frac), "L2": l2, "direction": -1,
          "from": src, "dropout": 0.3 * dim_frac, "trainable": True}})
      src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
    net_dict["encoder"] = {"class": "copy", "from": src}  # dim: EncValueTotalDim



  # Use label smoothing only at the very end.
  net_dict["output"]["unit"]["output_prob"]["loss_opts"]["label_smoothing"] = 0


  return net_dict


def get_new_net_dict(
  lstm_dim, att_num_heads, att_key_dim, beam_size, sos_idx, l2, learning_rate, time_red, feature_stddev, target, task):
  net_dict = {"#info": {"att_num_heads": att_num_heads, "enc_val_per_head": (lstm_dim * 2) // att_num_heads}}
  net_dict.update({
    "source": {
      "class": "eval",
      "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {
      "class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True},  # (T,40,32)
    "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0"},  # (T,20,32)
    "conv1": {
      "class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True},  # (T,20,32)
    "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1"},  # (T,10,32)
    "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)

    "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": "conv_merged"},
    "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": "conv_merged"},
    "lstm0_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (3,), "from": ["lstm0_fw", "lstm0_bw"],
      "trainable": False},

    "lstm1_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm0_pool"], "dropout": 0.3},
    "lstm1_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm0_pool"], "dropout": 0.3},
    "lstm1_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (2,), "from": ["lstm1_fw", "lstm1_bw"],
      "trainable": False},

    "lstm2_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm1_pool"], "dropout": 0.3},
    "lstm2_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm1_pool"], "dropout": 0.3},
    "lstm2_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1,), "from": ["lstm2_fw", "lstm2_bw"],
      "trainable": False},

    "lstm3_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm2_pool"], "dropout": 0.3},
    "lstm3_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm2_pool"], "dropout": 0.3},
    "lstm3_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1,), "from": ["lstm3_fw", "lstm3_bw"],
      "trainable": False},

    "lstm4_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm3_pool"], "dropout": 0.3},
    "lstm4_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm3_pool"], "dropout": 0.3},
    "lstm4_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1,), "from": ["lstm4_fw", "lstm4_bw"],
      "trainable": False},

    "lstm5_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm4_pool"], "dropout": 0.3},
    "lstm5_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm4_pool"], "dropout": 0.3},

    "encoder": {"class": "copy", "from": ["lstm5_fw", "lstm5_bw"]},  # dim: EncValueTotalDim
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": att_key_dim},
    # preprocessed_attended in Blocks
    "inv_fertility": {
      "class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": att_num_heads},
    "enc_value": {"class": "split_dims", "axis": "F", "dims": (att_num_heads, -1), "from": ["encoder"]},
    # (B, enc-T, H, D'/H)

    "output": {
      "class": "rec", "from": [], "unit": {
        'output': {
          'class': 'choice', 'target': target, 'beam_size': beam_size, 'from': ["output_prob"], "initial_output": sos_idx},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {
          'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 621,
          "initial_output": 0},  # feedback_input
        "weight_feedback": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
          "n_out": att_key_dim}, "s_transformed": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["s"], "n_out": att_key_dim},
        "energy_in": {
          "class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "s_transformed"],
          "n_out": att_key_dim}, "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": att_num_heads},
        # (B, enc-T, H)
        "exp_energy": {"class": "activation", "activation": "exp", "from": ["energy"]},
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
        "accum_att_weights": {
          "class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
          "eval": "source(0) + source(1) * source(2) * 0.5",
          "out_type": {"dim": att_num_heads, "shape": (None, att_num_heads)}},
        "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
        "att": {"class": "merge_dims", "axes": "except_batch", "from": ["att0"]},  # (B, H*V)
        "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:target_embed", "prev:att"], "n_out": 1000},
        # transform
        "readout_in": {"class": "linear", "from": ["s", "prev:target_embed", "att"], "activation": None, "n_out": 1000},
        # merge + post_merge bias
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]}, "output_prob": {
          "class": "softmax", "from": ["readout"], "dropout": 0.3, "target": target, "loss": "ce",
          "loss_opts": {"label_smoothing": 0.1}}}, "target": target},

    "decision": {
      "class": "decide", "from": ["output"], "loss": "edit_distance", "target": target, "loss_opts": {
        # "debug_print": True
      }}})

  if task != "eval":
    net_dict["output"]["max_seq_len"] = "max_len_from('base:encoder')"

  if feature_stddev is not None:
    assert type(feature_stddev) == float
    net_dict["source_stddev"] = {
      "class": "eval", "from": "data", "eval": "source(0) / " + str(feature_stddev)
    }
    net_dict["source"]["from"] = "source_stddev"

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
        net_dict["#config"]["batch_size"] = 15000
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
    net_dict["enc_value"]["dims"] = (net_dict["#info"]["att_num_heads"], int(net_dict["#info"]["enc_val_per_head"] * dim_frac * 0.5) * 2)
    # Use label smoothing only at the very end.
    net_dict["output"]["unit"]["output_prob"]["loss_opts"]["label_smoothing"] = 0
    return net_dict


def get_net_dict_like_seg_model(
  lstm_dim, att_num_heads, att_key_dim, beam_size, sos_idx, l2, learning_rate, time_red, feature_stddev, target, task):
  net_dict = {"#info": {"lstm_dim": lstm_dim, "l2": l2, "learning_rate": learning_rate, "time_red": time_red}}
  net_dict.update({
    "source": {
      "class": "eval",
      "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {
      "class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True, "auto_use_channel_first": False},  # (T,40,32)
    "conv0p": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0",
      "use_channel_first": False},  # (T,20,32)
    "conv1": {
      "class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True, "auto_use_channel_first": False},  # (T,20,32)
    "conv1p": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1",
      "use_channel_first": False},  # (T,10,32)
    "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)

    "lstm0_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": "conv_merged", "L2": 0.0001},
    "lstm0_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": "conv_merged", "L2": 0.0001},
    "lstm0_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (3,), "from": ["lstm0_fw", "lstm0_bw"],
      "trainable": False},

    "lstm1_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm0_pool"],
      "dropout": 0.3, "L2": 0.0001},
    "lstm1_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm0_pool"],
      "dropout": 0.3, "L2": 0.0001},
    "lstm1_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (2,), "from": ["lstm1_fw", "lstm1_bw"],
      "trainable": False},

    "lstm2_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm1_pool"],
      "dropout": 0.3, "L2": 0.0001},
    "lstm2_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm1_pool"],
      "dropout": 0.3, "L2": 0.0001},
    "lstm2_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1,), "from": ["lstm2_fw", "lstm2_bw"],
      "trainable": False},

    "lstm3_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm2_pool"],
      "dropout": 0.3, "L2": 0.0001},
    "lstm3_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm2_pool"],
      "dropout": 0.3, "L2": 0.0001},
    "lstm3_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1,), "from": ["lstm3_fw", "lstm3_bw"],
      "trainable": False},

    "lstm4_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm3_pool"],
      "dropout": 0.3, "L2": 0.0001},
    "lstm4_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm3_pool"],
      "dropout": 0.3, "L2": 0.0001},
    "lstm4_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1,), "from": ["lstm4_fw", "lstm4_bw"],
      "trainable": False},

    "lstm5_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm4_pool"],
      "dropout": 0.3, "L2": 0.0001},
    "lstm5_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm4_pool"],
      "dropout": 0.3, "L2": 0.0001},

    "encoder": {"class": "copy", "from": ["lstm5_fw", "lstm5_bw"]},  # dim: EncValueTotalDim
    "enc_ctx": {
      "class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
      "n_out": att_key_dim, "L2": 0.0001, "dropout": 0.2},
    # preprocessed_attended in Blocks
    "inv_fertility": {
      "class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": att_num_heads},
    "enc_value": {"class": "split_dims", "axis": "F", "dims": (att_num_heads, -1), "from": ["encoder"]},
    # (B, enc-T, H, D'/H)

    "output": {
      "class": "rec", "from": [], "unit": {
        'output': {
          'class': 'choice', 'target': target, 'beam_size': beam_size, 'from': ["output_prob"], "initial_output": sos_idx},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {
          'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 621,
          "initial_output": 0},  # feedback_input
        "weight_feedback": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
          "n_out": att_key_dim},
        "att_query": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["lm"], "n_out": att_key_dim},
        "energy_in": {
          "class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "att_query"],
          "n_out": att_key_dim},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": att_num_heads},
        # (B, enc-T, H)
        "att_weights0": {
          "class": "softmax_over_spatial", "from": ["energy"],
          "energy_factor": (att_key_dim // att_num_heads) ** -0.5},  # (B, enc-T, H)
        "att_weights": {
          "class": "dropout", "dropout": 0.1, "dropout_noise_shape": {"*": None}, "from": "att_weights0"},
        "accum_att_weights": {
          "class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
          "eval": "source(0) + source(1) * source(2) * 0.5",
          "out_type": {"dim": att_num_heads, "shape": (None, att_num_heads)}},
        "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
        "att": {"class": "merge_dims", "axes": "except_batch", "from": ["att0"]},  # (B, H*V)
        "lm": {"class": "rec", "unit": "nativelstm2", "from": ["prev:target_embed", "prev:att"], "n_out": lstm_dim},
        # transform
        "readout_in": {"class": "linear", "from": ["lm", "att"], "activation": None, "n_out": 1000},
        # merge + post_merge bias
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {
          "class": "softmax", "from": ["readout"], "dropout": 0.3, "target": target, "loss": "ce",
          "loss_opts": {"label_smoothing": 0.1}}}, "target": target},

    "decision": {
      "class": "decide", "from": ["output"], "loss": "edit_distance", "target": target, "loss_opts": {
        # "debug_print": True
      }}})

  if task != "eval":
    net_dict["output"]["max_seq_len"] = "max_len_from('base:encoder')"

  if feature_stddev is not None:
    assert type(feature_stddev) == float
    net_dict["source_stddev"] = {
      "class": "eval", "from": "data", "eval": "source(0) / " + str(feature_stddev)
    }
    net_dict["source"]["from"] = "source_stddev"

  return net_dict


def get_best_net_dict(
        lstm_dim, att_num_heads, att_key_dim, beam_size, sos_idx, feature_stddev, target, task, targetb_num_labels,
        weight_dropout, with_state_vector, with_weight_feedback, prev_target_in_readout, dump_output):
  net_dict = {"#info": {"att_num_heads": att_num_heads, "enc_val_per_head": (lstm_dim * 2) // att_num_heads}}
  net_dict.update({
    "source": {
      "class": "eval",
      "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {
      "class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True},  # (T,40,32)
    "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0"},  # (T,20,32)
    "conv1": {
      "class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True},  # (T,20,32)
    "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1"},  # (T,10,32)
    "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)

    "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": "conv_merged", "L2": 0.0001},
    "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": "conv_merged", "L2": 0.0001},
    "lstm0_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (3,), "from": ["lstm0_fw", "lstm0_bw"],
      "trainable": False},

    "lstm1_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm0_pool"], "dropout": 0.3,
      "L2": 0.0001}, "lstm1_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm0_pool"], "dropout": 0.3,
      "L2": 0.0001}, "lstm1_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (2,), "from": ["lstm1_fw", "lstm1_bw"],
      "trainable": False},

    "lstm2_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm1_pool"], "dropout": 0.3,
      "L2": 0.0001}, "lstm2_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm1_pool"], "dropout": 0.3,
      "L2": 0.0001}, "lstm2_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1,), "from": ["lstm2_fw", "lstm2_bw"],
      "trainable": False},

    "lstm3_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm2_pool"], "dropout": 0.3,
      "L2": 0.0001}, "lstm3_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm2_pool"], "dropout": 0.3,
      "L2": 0.0001}, "lstm3_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1,), "from": ["lstm3_fw", "lstm3_bw"],
      "trainable": False},

    "lstm4_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm3_pool"], "dropout": 0.3,
      "L2": 0.0001}, "lstm4_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm3_pool"], "dropout": 0.3,
      "L2": 0.0001}, "lstm4_pool": {
      "class": "pool", "mode": "max", "padding": "same", "pool_size": (1,), "from": ["lstm4_fw", "lstm4_bw"],
      "trainable": False},

    "lstm5_fw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": ["lstm4_pool"], "dropout": 0.3,
      "L2": 0.0001}, "lstm5_bw": {
      "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": ["lstm4_pool"], "dropout": 0.3,
      "L2": 0.0001},

    "encoder": {"class": "copy", "from": ["lstm5_fw", "lstm5_bw"]},  # dim: EncValueTotalDim
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"], "n_out": att_key_dim},
    # preprocessed_attended in Blocks
    "inv_fertility": {
      "class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": att_num_heads},
    "enc_value": {"class": "split_dims", "axis": "F", "dims": (att_num_heads, -1), "from": ["encoder"]},
    # (B, enc-T, H, D'/H)

    "ctc_out": {"class": "softmax", "from": "encoder", "with_bias": False, "n_out": targetb_num_labels},
    "ctc_out_scores": {
      "class": "eval", "from": ["ctc_out"], "eval": "safe_log(source(0))", }, # "_bpe0": {
    #   "class": "slice", "from": "data:%s" % target, "axis": "t", "slice_end": -1,  # cut of the EOS symbol
    #   "register_as_extern_data": "bpe0" if task == "train" else None},
    "ctc": {
      "class": "copy", "from": "ctc_out_scores", "loss": "ctc", "target": target if task == "train" else "bpe",
      "loss_opts": {
        "beam_width": 1, "use_native": True, "output_in_log_space": True, "ctc_opts": {"logits_normalize": False}}},

    "target_with_eos": {
      "class": "postfix_in_time", "postfix": sos_idx, "repeat": 1, "from": "data:%s" % target,
      "register_as_extern_data": "target_w_eos" if task == "train" else None},

    "output": {
      "class": "rec", "from": [], "unit": {
        'output': {
          'class': 'choice', 'target': "target_w_eos" if task == "train" else target,
          'beam_size': beam_size, 'from': ["label_prob"],
          "initial_output": sos_idx}, "end": {"class": "compare", "from": ["output"], "value": 0}, 'target_embed': {
          'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 621,
          "initial_output": 0},  # feedback_input
        "weight_feedback": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
          "n_out": att_key_dim}, "att_query": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["lm"], "n_out": att_key_dim},
        "att_energy_in": {
          "class": "combine", "kind": "add",
          "from": ["base:enc_ctx", "weight_feedback", "att_query"] if with_weight_feedback else ["base:enc_ctx",
                                                                                                 "att_query"],
          "n_out": att_key_dim},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["att_energy_in"]}, "energy": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": att_num_heads},
        # (B, enc-T, H)
        "att_weights0": {
          "class": "softmax_over_spatial", "from": ["energy"], "energy_factor": (att_key_dim // att_num_heads) ** -0.5},
        # (B, enc-T, H)
        "att_weights": {
          "class": "dropout", "dropout_noise_shape": {"*": None}, "from": 'att_weights0', "dropout": weight_dropout},
        "accum_att_weights": {
          "class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
          "eval": "source(0) + source(1) * source(2) * 0.5",
          "out_type": {"dim": att_num_heads, "shape": (None, att_num_heads)}},
        "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
        "att": {
          "class": "merge_dims", "axes": ["dim:%s" % att_num_heads, "dim:%s" % (lstm_dim * 2)], "from": ["att0"]},
        # (B, H*V)
        "lm": {
          "class": "rec", "unit": "nativelstm2",
          "from": ["prev:target_embed", "prev:att"] if with_state_vector else ["prev:target_embed"], "n_out": lstm_dim},
        # transform
        "readout_in": {
          "class": "linear", "from": ["lm", "prev:target_embed", "att"] if prev_target_in_readout else ["lm", "att"],
          "activation": None, "n_out": 1000}, # merge + post_merge bias
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "label_prob": {
          "class": "softmax", "from": ["readout"], "dropout": 0.3,
          "target": "target_w_eos" if task == "train" else target, "loss": "ce",
          "loss_opts": {"label_smoothing": 0.1}}}, "target": "target_w_eos" if task == "train" else target},

    "decision": {
      "class": "decide", "from": ["output"], "loss": "edit_distance", "target": target, "loss_opts": {
        # "debug_print": True
      }}})

  if dump_output:
    net_dict.update({
      "decision_no_mask": {
        "class": "decide", "from": "output", "loss": "edit_distance", "target": target, 'only_on_search': True},
      "dump_output": {
        "class": "hdf_dump", "from": "decision_no_mask", "filename": "search_out_seqs", "is_output_layer": True}})

  if task != "eval":
    net_dict["output"]["max_seq_len"] = "max_len_from('base:encoder')"

  if feature_stddev is not None:
    assert type(feature_stddev) == float
    net_dict["source_stddev"] = {
      "class": "eval", "from": "data", "eval": "source(0) / " + str(feature_stddev)}
    net_dict["source"]["from"] = "source_stddev"

  return net_dict


def best_custom_construction_algo(idx, net_dict):
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
  net_dict["enc_value"]["dims"] = (net_dict["#info"]["att_num_heads"], int(net_dict["#info"]["enc_val_per_head"] * dim_frac * 0.5) * 2)
  # Use label smoothing only at the very end.
  net_dict["output"]["unit"]["label_prob"]["loss_opts"]["label_smoothing"] = 0
  net_dict["output"]["unit"]["att"]["axes"][0] = ["dim:%s" % net_dict["#info"]["att_num_heads"]]
  net_dict["output"]["unit"]["att"]["axes"][1] = [
    "dim:%s" % (int(net_dict["#info"]["enc_val_per_head"] * dim_frac * 0.5) * 2)]

  return net_dict

