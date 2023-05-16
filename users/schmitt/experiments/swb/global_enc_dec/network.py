

def custom_construction_algo(idx, net_dict):
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
  net_dict["output"]["unit"]["att"]["axes"][0] = ["dim:%s" % net_dict["#info"]["att_num_heads"]]
  net_dict["output"]["unit"]["att"]["axes"][1] = [
    "dim:%s" % (int(net_dict["#info"]["enc_val_per_head"] * dim_frac * 0.5) * 2)]


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

