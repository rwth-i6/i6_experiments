def custom_construction_algo_segmental_att(idx, net_dict):
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

  if "task" not in net_dict["#info"]:
    net_dict["label_model"]["unit"]["label_prob"]["loss_opts"]["label_smoothing"] = 0

  return net_dict


def custom_construction_algo_global_att(idx, net_dict):
  import numpy as np
  learning_rate = 0.001
  time_red = [3, 2]

  if idx > 30:
    return None

  net_dict["#config"] = {}
  if idx is not None:
    # learning rate warm up
    lr_warmup = list(np.linspace(learning_rate * 0.1, learning_rate, num=10))
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

    time_reduction = time_red if num_lstm_layers >= 3 else [int(np.prod(time_red))]

    # Add encoder BLSTM stack
    src = "conv_merged"
    lstm_dim = 1024
    l2 = 0.0001
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
  net_dict["output"]["unit"]["label_prob"]["loss_opts"]["label_smoothing"] = 0
  # net_dict["output"]["unit"]["att"]["axes"][0] = ["dim:%s" % net_dict["#info"]["att_num_heads"]]
  # net_dict["output"]["unit"]["att"]["axes"][1] = [
  #   "dim:%s" % (int(net_dict["#info"]["enc_val_per_head"] * dim_frac * 0.5) * 2)]


  return net_dict
