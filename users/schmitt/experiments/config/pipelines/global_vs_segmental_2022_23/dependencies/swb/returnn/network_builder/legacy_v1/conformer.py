def get_conformer_encoder_tim(num_blocks=16, dropout=0.03, l2=None):
  network = {
    "aux_12_ff1": {
      "activation": "relu",
      "class": "linear",
      "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
      "from": ["aux_12_length_masked"],
      "n_out": 256,
      "with_bias": True,
    },
    "aux_12_ff2": {
      "activation": None,
      "class": "linear",
      "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
      "from": ["aux_12_ff1"],
      "n_out": 256,
      "with_bias": True,
    },
    "aux_12_length_masked": {
      "class": "reinterpret_data",
      "from": ["enc_012"],
    },
    "aux_12_output_prob": {
      "class": "softmax",
      "dropout": 0.0,
      "from": ["aux_12_ff2"],
      "loss": "ce",
      "loss_opts": {
        "focal_loss_factor": 0.0,
        "label_smoothing": 0.0,
        "use_normalized_loss": False,
      },
      "loss_scale": 0.5,
      "target": "targetb",
    },
    "aux_4_ff1": {
      "activation": "relu",
      "class": "linear",
      "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
      "from": ["aux_4_length_masked"],
      "n_out": 256,
      "with_bias": True,
    },
    "aux_4_ff2": {
      "activation": None,
      "class": "linear",
      "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
      "from": ["aux_4_ff1"],
      "n_out": 256,
      "with_bias": True,
    },
    "aux_4_length_masked": {
      "class": "reinterpret_data",
      "from": ["enc_004"],
      "size_base": "data:targetb",
    },
    "aux_4_output_prob": {
      "class": "softmax",
      "dropout": 0.0,
      "from": ["aux_4_ff2"],
      "loss": "ce",
      "loss_opts": {
        "focal_loss_factor": 0.0,
        "label_smoothing": 0.0,
        "use_normalized_loss": False,
      },
      "loss_scale": 0.5,
      "target": "targetb",
    },
    "aux_8_ff1": {
      "activation": "relu",
      "class": "linear",
      "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
      "from": ["aux_8_length_masked"],
      "n_out": 256,
      "with_bias": True,
    },
    "aux_8_ff2": {
      "activation": None,
      "class": "linear",
      "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
      "from": ["aux_8_ff1"],
      "n_out": 256,
      "with_bias": True,
    },
    "aux_8_length_masked": {
      "class": "reinterpret_data",
      "from": ["enc_008"],
      "size_base": "data:targetb",
    },
    "aux_8_output_prob": {
      "class": "softmax",
      "dropout": 0.0,
      "from": ["aux_8_ff2"],
      "loss": "ce",
      "loss_opts": {
        "focal_loss_factor": 0.0,
        "label_smoothing": 0.0,
        "use_normalized_loss": False,
      },
      "loss_scale": 0.5,
      "target": "targetb",
    },
    "conv0_0": {
      "activation": None,
      "class": "conv",
      "filter_size": (3, 3),
      "from": "source0",
      "in_spatial_dims": ["T", "dim:40"],
      "n_out": 32,
      "padding": "same",
      "with_bias": True,
    },
    "conv0_1": {
      "activation": "relu",
      "class": "conv",
      "filter_size": (3, 3),
      "from": "conv0_0",
      "in_spatial_dims": ["T", "dim:40"],
      "n_out": 32,
      "padding": "same",
      "with_bias": True,
    },
    "conv0p": {
      "class": "pool",
      "from": "conv0_1",
      "in_spatial_dims": ["T", "dim:40"],
      "mode": "max",
      "padding": "same",
      "pool_size": (1, 2),
      "strides": (1, 2),
    },
    "conv1_0": {
      "activation": None,
      "class": "conv",
      "filter_size": (3, 3),
      "from": "conv0p",
      "in_spatial_dims": ["T", "dim:20"],
      "n_out": 64,
      "padding": "same",
      "with_bias": True,
    },
    "conv1_1": {
      "activation": "relu",
      "class": "conv",
      "filter_size": (3, 3),
      "from": "conv1_0",
      "in_spatial_dims": ["T", "dim:20"],
      "n_out": 64,
      "padding": "same",
      "with_bias": True,
    },
    "conv1p": {
      "class": "pool",
      "from": "conv1_1",
      "in_spatial_dims": ["T", "dim:20"],
      "mode": "max",
      "padding": "same",
      "pool_size": (1, 1),
      "strides": (1, 1),
    },
    "conv_merged": {
      "axes": ["dim:20", "dim:64"],
      "class": "merge_dims",
      "from": "conv1p",
    },
    "embedding": {
      "activation": None,
      "class": "linear",
      "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
      "from": ["feature_stacking_merged"],
      "n_out": 512,
      "with_bias": True,
    },
    "embedding_dropout": {"class": "dropout", "dropout": 0.0, "from": ["embedding"]},
    "encoder": {"class": "layer_norm", "from": ["enc_016"]},
    "feature_stacking_merged": {
      "axes": ["dim:6", "F"],
      "class": "merge_dims",
      "from": ["feature_stacking_window"],
    },
    "feature_stacking_window": {
      "class": "window",
      "from": ["conv_merged"],
      "stride": 6,
      "window_left": 5,
      "window_right": 0,
      "window_size": 6,
    },
    "source0": {
      "axis": "F",
      "class": "split_dims",
      "dims": (-1, 1),
      "from": ["source"],
    },
  }

  for i in range(1, num_blocks+1):
    current_idx = "%03d" % i
    prev_idx = "%03d" % (i-1)
    network.update({
      "enc_%s" % current_idx: {"class": "copy", "from": "enc_%s_ff2_out" % current_idx},
      "enc_%s_conv_GLU" % current_idx: {
        "activation": "identity",
        "class": "gating",
        "from": ["enc_%s_conv_pointwise1" % current_idx],
      },
      "enc_%s_conv_SE_act1" % current_idx: {
        "activation": "swish",
        "class": "activation",
        "from": "enc_%s_conv_SE_linear1" % current_idx,
      },
      "enc_%s_conv_SE_act2" % current_idx: {
        "activation": "swish",
        "class": "activation",
        "from": "enc_%s_conv_SE_linear2" % current_idx,
      },
      "enc_%s_conv_SE_elm_mul" % current_idx: {
        "class": "eval",
        "eval": "source(0) * source(1)",
        "from": ["enc_%s_conv_SE_act2" % current_idx, "enc_%s_conv_depthwise"  % current_idx],
      },
      "enc_%s_conv_SE_linear1" % current_idx: {
        "class": "linear",
        "from": "enc_%s_conv_SE_reduce" % current_idx,
        "n_out": 32,
      },
      "enc_%s_conv_SE_linear2" % current_idx: {
        "class": "linear",
        "from": "enc_%s_conv_SE_act1" % current_idx,
        "n_out": 512,
      },
      "enc_%s_conv_SE_reduce" % current_idx: {
        "axes": "T",
        "class": "reduce",
        "from": "enc_%s_conv_depthwise" % current_idx,
        "mode": "mean",
      },
      "enc_%s_conv_act" % current_idx: {
        "activation": "swish",
        "class": "activation",
        "from": ["enc_%s_conv_batchnorm" % current_idx],
      },
      "enc_%s_conv_batchnorm" % current_idx: {
        "class": "layer_norm",
        "from": ["enc_%s_conv_SE_elm_mul" % current_idx],
      },
      "enc_%s_conv_depthwise" % current_idx: {
        "activation": None,
        "class": "conv",
        "filter_size": (32,),
        "from": ["enc_%s_conv_GLU" % current_idx],
        "groups": 512,
        "n_out": 512,
        "padding": "same",
        "with_bias": True,
      },
      "enc_%s_conv_dropout" % current_idx: {
        "class": "dropout",
        "dropout": dropout,
        "from": ["enc_%s_conv_pointwise2" % current_idx],
      },
      "enc_%s_conv_laynorm" % current_idx: {"class": "layer_norm", "from": ["enc_%s_self_att_out" % current_idx]},
      "enc_%s_conv_output" % current_idx: {
        "class": "combine",
        "from": ["enc_%s_self_att_out" % current_idx, "enc_%s_conv_dropout" % current_idx],
        "kind": "add",
        "n_out": 512,
      },
      "enc_%s_conv_pointwise1" % current_idx: {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_%s_conv_laynorm" % current_idx],
        "n_out": 1024,
        "with_bias": False,
      },
      "enc_%s_conv_pointwise2" % current_idx: {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_%s_conv_act" % current_idx],
        "n_out": 512,
        "with_bias": False,
      },
      "enc_%s_ff1_conv1" % current_idx: {
        "activation": "swish",
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_%s_ff1_laynorm" % current_idx],
        "n_out": 2048,
        "with_bias": True,
      },
      "enc_%s_ff1_conv2" % current_idx: {
        "activation": None,
        "class": "linear",
        "dropout": dropout,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_%s_ff1_conv1" % current_idx],
        "n_out": 512,
        "with_bias": True,
      },
      "enc_%s_ff1_drop" % current_idx: {
        "class": "dropout",
        "dropout": dropout,
        "from": ["enc_%s_ff1_conv2" % current_idx],
      },
      "enc_%s_ff1_drop_half" % current_idx: {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": ["enc_%s_ff1_drop" % current_idx],
      },
      "enc_%s_ff1_ff1_SE_act1" % current_idx: {
        "activation": "swish",
        "class": "activation",
        "from": "enc_%s_ff1_ff1_SE_linear1" % current_idx,
      },
      "enc_%s_ff1_ff1_SE_act2" % current_idx: {
        "activation": "swish",
        "class": "activation",
        "from": "enc_%s_ff1_ff1_SE_linear2" % current_idx,
      },
      "enc_%s_ff1_ff1_SE_elm_mul" % current_idx: {
        "class": "eval",
        "eval": "source(0) * source(1)",
        "from": ["enc_%s_ff1_ff1_SE_act2" % current_idx, "enc_%s_ff1_conv1" % current_idx],
      },
      "enc_%s_ff1_ff1_SE_linear1" % current_idx: {
        "class": "linear",
        "from": "enc_%s_ff1_ff1_SE_reduce" % current_idx,
        "n_out": 32,
      },
      "enc_%s_ff1_ff1_SE_linear2" % current_idx: {
        "class": "linear",
        "from": "enc_%s_ff1_ff1_SE_act1" % current_idx,
        "n_out": 512,
      },
      "enc_%s_ff1_ff1_SE_reduce" % current_idx: {
        "axes": "T",
        "class": "reduce",
        "from": "enc_%s_ff1_conv1" % current_idx,
        "mode": "mean",
      },
      "enc_%s_ff1_laynorm" % current_idx: {
        "class": "layer_norm", "from": "embedding_dropout" if i == 1 else "enc_%s" % prev_idx},
      "enc_%s_ff1_out" % current_idx: {
        "class": "combine",
        "from": ["embedding_dropout" if i == 1 else "enc_%s" % prev_idx, "enc_%s_ff1_drop_half" % current_idx],
        "kind": "add",
      },
      "enc_%s_ff2_conv1" % current_idx: {
        "activation": "swish",
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_%s_ff2_laynorm" % current_idx],
        "n_out": 2048,
        "with_bias": True,
      },
      "enc_%s_ff2_conv2" % current_idx: {
        "activation": None,
        "class": "linear",
        "dropout": dropout,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_%s_ff2_conv1" % current_idx],
        "n_out": 512,
        "with_bias": True,
      },
      "enc_%s_ff2_drop" % current_idx: {
        "class": "dropout",
        "dropout": dropout,
        "from": ["enc_%s_ff2_conv2" % current_idx],
      },
      "enc_%s_ff2_drop_half" % current_idx: {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": ["enc_%s_ff2_drop" % current_idx],
      },
      "enc_%s_ff2_ff2_SE_act1" % current_idx: {
        "activation": "swish",
        "class": "activation",
        "from": "enc_%s_ff2_ff2_SE_linear1" % current_idx,
      },
      "enc_%s_ff2_ff2_SE_act2" % current_idx: {
        "activation": "swish",
        "class": "activation",
        "from": "enc_%s_ff2_ff2_SE_linear2" % current_idx,
      },
      "enc_%s_ff2_ff2_SE_elm_mul" % current_idx: {
        "class": "eval",
        "eval": "source(0) * source(1)",
        "from": ["enc_%s_ff2_ff2_SE_act2" % current_idx, "enc_%s_ff2_conv1" % current_idx],
      },
      "enc_%s_ff2_ff2_SE_linear1" % current_idx: {
        "class": "linear",
        "from": "enc_%s_ff2_ff2_SE_reduce" % current_idx,
        "n_out": 32,
      },
      "enc_%s_ff2_ff2_SE_linear2" % current_idx: {
        "class": "linear",
        "from": "enc_%s_ff2_ff2_SE_act1" % current_idx,
        "n_out": 512,
      },
      "enc_%s_ff2_ff2_SE_reduce" % current_idx: {
        "axes": "T",
        "class": "reduce",
        "from": "enc_%s_ff2_conv1" % current_idx,
        "mode": "mean",
      },
      "enc_%s_ff2_laynorm" % current_idx: {"class": "layer_norm", "from": "enc_%s_conv_output" % current_idx},
      "enc_%s_ff2_out" % current_idx: {
        "class": "combine",
        "from": ["enc_%s_conv_output" % current_idx, "enc_%s_ff2_drop_half" % current_idx],
        "kind": "add",
      },
      "enc_%s_rel_pos" % current_idx: {
        "class": "relative_positional_encoding",
        "clipping": 400,
        "fixed": False,
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_%s_self_att_laynorm" % current_idx],
        "n_out": 64,
      },
      "enc_%s_self_att_att" % current_idx: {
        "attention_dropout": 0.03,
        "attention_left_only": False,
        "class": "self_attention",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_%s_self_att_laynorm" % current_idx],
        "key_shift": "enc_%s_rel_pos" % current_idx,
        "n_out": 512,
        "num_heads": 8,
        "total_key_dim": 512,
      },
      "enc_%s_self_att_drop" % current_idx: {
        "class": "dropout",
        "dropout": dropout,
        "from": ["enc_%s_self_att_lin" % current_idx],
      },
      "enc_%s_self_att_laynorm" % current_idx: {"class": "layer_norm", "from": ["enc_%s_ff1_out" % current_idx]},
      "enc_%s_self_att_lin" % current_idx: {
        "activation": None,
        "class": "linear",
        "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
        "from": ["enc_%s_self_att_att" % current_idx],
        "n_out": 512,
        "with_bias": False,
      },
      "enc_%s_self_att_out" % current_idx: {
        "class": "combine",
        "from": ["enc_%s_ff1_out" % current_idx, "enc_%s_self_att_drop" % current_idx],
        "kind": "add",
        "n_out": 512,
      },
    })

  return network


def get_conformer_encoder_wei(use_blstm=False, batch_norm=True, num_blocks=12):
  network = {
    'input_dropout': {'class': 'copy', 'dropout': 0.1, 'from': 'input_linear'},
    'input_linear': {'L2': 5e-06, 'activation': None, 'class': 'linear', 'from': 'conv_merged', 'n_out': 512,
                     'with_bias': False},
    'encoder': {'class': 'reinterpret_data', 'from': 'conformer_%d_output' % num_blocks},
    'enc_output': {
      'class': 'softmax', 'from': 'encoder', 'loss': 'ce', 'loss_opts': {'focal_loss_factor': 1.0}, 'target': 'targetb'},
    'enc_output_loss': {
      'class': 'softmax', 'from': 'conformer_6_output', 'loss': 'ce', 'loss_opts': {'focal_loss_factor': 1.0},
      'loss_scale': 0.3, 'target': 'targetb'},
  }

  for i in range(1, num_blocks+1):
    network.update({
      'conformer_%d_conv_mod_bn' % i: {
        'class': 'batch_norm', 'delay_sample_update': True, 'epsilon': 1e-05,
        'from': 'conformer_%d_conv_mod_depthwise_conv' % i, 'momentum': 0.1,
        'update_sample_only_in_training': True} if batch_norm else {
        'class': 'layer_norm', 'from': 'conformer_%d_conv_mod_depthwise_conv' % i},
      'conformer_%d_conv_mod_depthwise_conv' % i: {
        'L2': 5e-06, 'activation': None, 'class': 'conv', 'filter_size': (32,), 'from': 'conformer_%d_conv_mod_glu' % i,
        'groups': 512, 'n_out': 512, 'padding': 'same', 'with_bias': True},
      'conformer_%d_conv_mod_dropout' % i: {
        'class': 'copy', 'dropout': 0.1, 'from': 'conformer_%d_conv_mod_pointwise_conv_2' % i},
      'conformer_%d_conv_mod_glu' % i: {
        'activation': None, 'class': 'gating', 'from': 'conformer_%d_conv_mod_pointwise_conv_1' % i,
        'gate_activation': 'sigmoid'},
      'conformer_%d_conv_mod_ln' % i: {'class': 'layer_norm', 'from': 'conformer_%d_ffmod_1_half_res_add' % i},
      'conformer_%d_conv_mod_pointwise_conv_1' % i: {
        'L2': 5e-06, 'activation': None, 'class': 'linear', 'from': 'conformer_%d_conv_mod_ln' % i, 'n_out': 1024},
      'conformer_%d_conv_mod_pointwise_conv_2' % i: {
        'L2': 5e-06, 'activation': None, 'class': 'linear', 'from': 'conformer_%d_conv_mod_swish' % i, 'n_out': 512},
      'conformer_%d_conv_mod_res_add' % i: {
        'class': 'combine', 'from': ['conformer_%d_conv_mod_dropout' % i, 'conformer_%d_ffmod_1_half_res_add' % i],
        'kind': 'add'},
      'conformer_%d_conv_mod_swish' % i: {
        'activation': 'swish', 'class': 'activation', 'from': 'conformer_%d_conv_mod_bn' % i},
      'conformer_%d_ffmod_1_dropout' % i: {
        'class': 'copy', 'dropout': 0.1, 'from': 'conformer_%d_ffmod_1_dropout_linear' % i},
      'conformer_%d_ffmod_1_dropout_linear' % i: {
        'L2': 5e-06, 'activation': None, 'class': 'linear', 'dropout': 0.1,
        'from': 'conformer_%d_ffmod_1_linear_swish' % i,
        'n_out': 512},
      'conformer_%d_ffmod_1_half_res_add' % i: {
        'class': 'eval', 'eval': '0.5 * source(0) + source(1)',
        'from': ['conformer_%d_ffmod_1_dropout' % i, ('conformer_%d_output' % (i-1)) if i > 1 else "input_dropout"]},
      'conformer_%d_ffmod_1_linear_swish' % i: {
        'L2': 5e-06, 'activation': 'swish', 'class': 'linear', 'from': 'conformer_%d_ffmod_1_ln' % i, 'n_out': 2048},
      'conformer_%d_ffmod_1_ln' % i: {
        'class': 'layer_norm', 'from': ('conformer_%d_output' % (i-1)) if i > 1 else "input_dropout"},
      'conformer_%d_ffmod_2_dropout' % i: {
        'class': 'copy', 'dropout': 0.1, 'from': 'conformer_%d_ffmod_2_dropout_linear' % i},
      'conformer_%d_ffmod_2_dropout_linear' % i: {
        'L2': 5e-06, 'activation': None, 'class': 'linear', 'dropout': 0.1,
        'from': 'conformer_%d_ffmod_2_linear_swish' % i,
        'n_out': 512},
      'conformer_%d_ffmod_2_half_res_add' % i: {
        'class': 'eval', 'eval': '0.5 * source(0) + source(1)',
        'from': ['conformer_%d_ffmod_2_dropout' % i, 'conformer_%d_mhsa_mod_res_add' % i]},
      'conformer_%d_ffmod_2_linear_swish' % i: {
        'L2': 5e-06, 'activation': 'swish', 'class': 'linear',
        'from': 'conformer_%d_ffmod_2_ln' % i, 'n_out': 2048},
      'conformer_%d_ffmod_2_ln' % i: {'class': 'layer_norm', 'from': 'conformer_%d_mhsa_mod_res_add' % i},
      'conformer_%d_mhsa_mod_att_linear' % i: {
        'L2': 5e-06, 'activation': None, 'class': 'linear', 'from': 'conformer_%d_mhsa_mod_self_attention' % i,
        'n_out': 512, 'with_bias': False},
      'conformer_%d_mhsa_mod_dropout' % i: {
        'class': 'copy', 'dropout': 0.1, 'from': 'conformer_%d_mhsa_mod_att_linear' % i},
      'conformer_%d_mhsa_mod_ln' % i: {'class': 'layer_norm', 'from': 'conformer_%d_conv_mod_res_add' % i},
      'conformer_%d_mhsa_mod_relpos_encoding' % i: {
        'class': 'relative_positional_encoding', 'clipping': 32, 'from': 'conformer_%d_mhsa_mod_ln' % i, 'n_out': 64},
      'conformer_%d_mhsa_mod_res_add' % i: {
        'class': 'combine',
        'from': ['conformer_%d_mhsa_mod_dropout' % i, 'conformer_%d_conv_mod_res_add' % i], 'kind': 'add'},
      'conformer_%d_mhsa_mod_self_attention' % i: {
        'attention_dropout': 0.1, 'class': 'self_attention', 'from': 'conformer_%d_mhsa_mod_ln' % i,
        'key_shift': 'conformer_%d_mhsa_mod_relpos_encoding' % i, 'n_out': 512, 'num_heads': 8, 'total_key_dim': 512},
      'conformer_%d_output' % i: {'class': 'layer_norm', 'from': 'conformer_%d_ffmod_2_half_res_add' % i},
    })

  if use_blstm:
    network.update({
      'lstm0_bw': {'L2': 0.0001, 'class': 'rec', 'direction': -1, 'dropout': 0.1, 'from': 'source', 'n_out': 512,
                   'unit': 'nativelstm2'},
      'lstm0_fw': {'L2': 0.0001, 'class': 'rec', 'direction': 1, 'dropout': 0.1, 'from': 'source', 'n_out': 512,
                   'unit': 'nativelstm2'},

      'lstm0_pool': {'class': 'pool', 'from': ['lstm0_fw', 'lstm0_bw'], 'mode': 'max', 'padding': 'same',
                     'pool_size': (3,), 'trainable': False},

      'lstm1_bw': {'L2': 0.0001, 'class': 'rec', 'direction': -1, 'dropout': 0.1, 'from': 'lstm0_pool', 'n_out': 512,
                   'unit': 'nativelstm2'},

      'lstm1_fw': {'L2': 0.0001, 'class': 'rec', 'direction': 1, 'dropout': 0.1, 'from': 'lstm0_pool', 'n_out': 512,
                   'unit': 'nativelstm2'},

      'lstm1_pool': {'class': 'pool', 'from': ['lstm1_fw', 'lstm1_bw'], 'mode': 'max', 'padding': 'same',
                     'pool_size': (2,), 'trainable': False},
    })
    network["input_linear"]["from"] = "lstm1_pool"
  else:
    network.update({
      'conv_source': {'axis': 'F', 'class': 'split_dims', 'dims': (-1, 1), 'from': 'source'},
      'conv_1': {'L2': 0.01,
                 'activation': 'swish',
                 'class': 'conv',
                 'filter_size': (3, 3),
                 'from': 'conv_source',
                 'n_out': 32,
                 'padding': 'same',
                 'with_bias': True},
      'conv_1_pool': {'class': 'pool', 'from': 'conv_1', 'mode': 'max', 'padding': 'same', 'pool_size': (1, 2),
                      'trainable': False},
      'conv_2': {'L2': 0.01,
                 'activation': 'swish',
                 'class': 'conv',
                 'filter_size': (3, 3),
                 'from': 'conv_1_pool',
                 'n_out': 64,
                 'padding': 'same',
                 'strides': (3, 1),
                 'with_bias': True},
      'conv_3': {'L2': 0.01,
                 'activation': 'swish',
                 'class': 'conv',
                 'filter_size': (3, 3),
                 'from': 'conv_2',
                 'n_out': 64,
                 'padding': 'same',
                 'strides': (2, 1),
                 'with_bias': True},
      'conv_merged': {'axes': 'static', 'class': 'merge_dims', 'from': 'conv_3'},
    })

  return network
