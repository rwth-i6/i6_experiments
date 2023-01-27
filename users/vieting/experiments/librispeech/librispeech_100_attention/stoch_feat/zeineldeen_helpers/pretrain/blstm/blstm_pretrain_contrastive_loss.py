def custom_construction_algo(idx, net_dict):
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
    net_dict["#config"]["batch_size"] = 20000
  idx = max(idx - 4, 0)  # repeat first
  num_lstm_layers = idx + StartNumLayers  # idx starts at 0. start with N layers
  if num_lstm_layers > orig_num_lstm_layers:
    # Finish. This will also use label-smoothing then.
    return None
  if num_lstm_layers == 2:
    del net_dict['contrastive_loss']
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
    if "L2" in net_dict[layer]:
      net_dict[layer]['L2'] *= dim_frac
  net_dict["enc_value"]["dims"] = (AttNumHeads, int(EncValuePerHeadDim * dim_frac * 0.5) * 2)
  # Use label smoothing only at the very end.
  if 'loss_opts' in net_dict["output"]["unit"]["output_prob"]:
    net_dict["output"]["unit"]["output_prob"]["loss_opts"]["label_smoothing"] = 0
  return net_dict


def get_funcs():
  funcs = []
  for k, v in list(globals().items()):
    if callable(v):
      if k == 'get_funcs':
        continue
      funcs.append(v)
  return funcs
