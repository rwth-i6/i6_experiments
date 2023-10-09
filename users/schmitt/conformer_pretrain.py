def custom_construction_algo_mohammad_conf(idx, net_dict):
  final_num_blocks = 12
  initial_num_blocks = 2
  initial_dim_factor = 0.5
  initial_batch_size = 20000
  att_num_heads = 8
  encoder_key_total_dim = 512


  # different batch size in the beginning
  if idx < 3:
    net_dict["#config"] = {"batch_size": initial_batch_size}

  idx = max(idx-1, 0)  # repeat first

  idx += 1
  num_blocks = 2 ** idx  # 2, 2, 4, 8, 12

  if num_blocks > 12:
    return None

  grow_frac_enc = 1.0 - float(final_num_blocks - num_blocks) / (final_num_blocks - initial_num_blocks)
  dim_frac_enc = initial_dim_factor + (1.0 - initial_dim_factor) * grow_frac_enc

  encoder_pos_enc_keys = ["conformer_%d_%s" % (i, layer_name) for i in range(1, num_blocks + 1) for layer_name in
    ["mhsa_mod_relpos_encoding"]]

  encoder_keys = [
    "conformer_%d_%s" % (i, layer_name) for i in range(1, num_blocks + 1) for layer_name in [
      "conv_mod_bn", "conv_mod_depthwise_conv", "conv_mod_dropout", "conv_mod_glu", "conv_mod_ln",
      "conv_mod_pointwise_conv_1", "conv_mod_pointwise_conv_2", "conv_mod_res_add", "conv_mod_swish",
      "ffmod_1_dropout", "ffmod_1_dropout_linear", "ffmod_1_half_res_add", "ffmod_1_linear_swish",
      "ffmod_1_ln", "ffmod_2_dropout", "ffmod_2_dropout_linear", "ffmod_2_half_res_add",
      "ffmod_2_linear_swish", "ffmod_2_ln", "mhsa_mod_att_linear", "mhsa_mod_dropout", "mhsa_mod_ln",
      "mhsa_mod_res_add", "mhsa_mod_self_attention"
    ]
  ]

  encoder_keys += [
    "input_linear", # "conv_3", "conv_2", "conv_1"
  ]

  # modify the net dict

  # reduce number of conformer blocks
  net_dict["encoder"]["from"] = "conformer_%d_output" % num_blocks
  # remove auxiliary losses
  net_dict["enc_output"].update({"loss": None, "loss_opts": None})
  net_dict["enc_output_loss"].update({"loss": None, "loss_opts": None})

  # modify layer dimensions and regularization
  for k in encoder_keys:
    if "L2" in net_dict[k]:
      net_dict[k]["L2"] = 0.0 if idx <= 1 else net_dict[k]["L2"] * dim_frac_enc
    if "dropout" in net_dict[k]:
      net_dict[k]["dropout"] = 0.0 if idx <= 1 else net_dict[k]["dropout"] * dim_frac_enc
    if "n_out" in net_dict[k]:
      net_dict[k]["n_out"] = int(net_dict[k]["n_out"] * dim_frac_enc / float(att_num_heads)) * att_num_heads
    if "groups" in net_dict[k]:
      net_dict[k]["groups"] = int(net_dict[k]["groups"] * dim_frac_enc / float(att_num_heads)) * att_num_heads
    if "total_key_dim" in net_dict[k]:
      net_dict[k]["total_key_dim"] = int(net_dict[k]["total_key_dim"] * dim_frac_enc / float(att_num_heads)) * att_num_heads

  for k in encoder_pos_enc_keys:
    net_dict[k]["n_out"] = int(encoder_key_total_dim * dim_frac_enc / float(att_num_heads))

  return net_dict


def get_network(epoch, **kwargs):
  # from networks import networks_dict
  for epoch_ in sorted(networks_dict.keys(), reverse=True):
    if epoch_ <= epoch:
      return networks_dict[epoch_]
  assert False, "Error, no networks found"
