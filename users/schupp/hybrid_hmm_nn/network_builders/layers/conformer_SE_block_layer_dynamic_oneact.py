DEFAULT_INIT = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)"

# Patchining in some alternate conformer arcitectures

def add_SE_block(network, in_layer, name_prefix, se_act="swish"):
  # This adds and SE block anywhere
  # Returns the output layer name

  network[name_prefix + "_SE_reduce"] = {
    "class" : "reduce",
    "mode" : "mean",
    "from"  : in_layer,
    "axes" : "T"
  }

  network[name_prefix + "_SE_linear1"] = {
    "class" : "linear",
    "from" : name_prefix + "_SE_reduce",
    "n_out" : 32
  }

  network[name_prefix + "_SE_act1"] = {
    "class" : "activation",
    "activation" : se_act,
    "from" : name_prefix + "_SE_linear1"
  }

  network[name_prefix + "_SE_linear2"] = {
    "class" : "linear",
    "from" : name_prefix + "_SE_act1",
    "n_out" : 256
  }

  network[name_prefix + "_SE_elm_mul"] = {
    "class" : "eval",
    "eval" : "source(0) * source(1)",
    "from" : [name_prefix + "_SE_linear2", in_layer]
  }

  return name_prefix + "_SE_elm_mul"


def conformer_enc_layer_all_in_one_SE(
  network, name, num_heads, model_dim, key_dim, value_dim, ff_dim,
  kernel_size,
  sa_dropout, sa_post_dropout, ff_activation_dropout, ff_post_dropout,
  from_layers, conv_post_dropout,
  initialization=DEFAULT_INIT, ff_activation="swish",
  end_layernorm=False,
  normal_conv=False, output_channels=16,
  kernel_size_for_feature=3,
  attention_left_only=False, separated=False,
  windowing=False, window_size=None, gauss_window=False,
  relative_pe=False, fixed=False, clipping=100, untied_pe=False, relative_pe_transformer_xl=False,
  linear_mapping = True, linear_mapping_bias = False, switch = False,
  energy_factor = -0.5,
  half_ratio = 0.5,
  half_ratio_levels = None, 
  with_se = True,
  se_pos = None,
  se_act = "swish"
):
  if windowing or untied_pe or relative_pe_transformer_xl or energy_factor != -0.5:
    assert separated

  if with_se:
    assert not se_pos is None, "this version needs se_pos != None"

  if half_ratio_levels is not None:
    idx = int(name.split("_")[-1]) - 1 # Hack but does the trick
    half_ratio = half_ratio_levels[idx]

  if from_layers is None:
    from_layers = ["data"]
  elif isinstance(from_layers, str):
    from_layers = [from_layers]

  ## first ffn with residual connection
  network[f"{name}_ff1_laynorm"] = {'class': "layer_norm",
                                    'from': from_layers}
  network[f"{name}_ff1_conv1"] = {
    'class': "linear", 'activation': ff_activation, 'with_bias': True,
    'from': [f"{name}_ff1_laynorm"],
    'n_out': ff_dim, 'forward_weights_init': initialization
  }

  network[f"{name}_ff1_conv2"] = {
    'class': "linear", 'activation': None, 'with_bias': True,
    'from': [f"{name}_ff1_conv1"], 'dropout': ff_activation_dropout,
    'n_out': model_dim, 'forward_weights_init': initialization
  }
  network[f"{name}_ff1_drop"] = {'class': "dropout",
                                 'dropout': ff_post_dropout,
                                 'from': [f"{name}_ff1_conv2"]}

  network[f"{name}_ff1_drop_half"] = {
    'class': "eval",
    'eval': f"{half_ratio} * source(0)",
    'from': [f"{name}_ff1_drop"]
  }
  network[f"{name}_ff1_out"] = {
    'class': "combine", 'kind': "add",
    'from': from_layers + [f"{name}_ff1_drop_half"]
  }

  ## MHSA module
  network[f"{name}_self_att_laynorm"] = {'class': "layer_norm",
                                         'from': [f"{name}_ff1_out"]}

  if separated:
    key_per_head = int(key_dim / num_heads)
    value_per_head = int(value_dim / num_heads)

    network[f"{name}_att_query0"] = {
      'class': "linear", 'activation': None, 'with_bias': False,
      'from': [f"{name}_self_att_laynorm"], 'n_out': key_dim,
      'forward_weights_init': initialization
    }


    # query per head
    network[f"{name}_att_query"] = {
      'class': "split_dims", 'axis': "F",
      'dims': (num_heads, key_per_head),  # (B, T, H, D/H)
      'from': [f"{name}_att_query0"],
    }

    network[f"{name}_att_key0"] = {
      'class': "linear", 'activation': None, 'with_bias': False,
      'from': [f"{name}_self_att_laynorm"], 'n_out': key_dim,  # (B, enc-T, D)
      'forward_weights_init': initialization,
    }
    network[f"{name}_att_value0"] = {
      'class': "linear", 'activation': None, 'with_bias': False,
      'from': [f"{name}_self_att_laynorm"], 'n_out': value_dim,
      'forward_weights_init': initialization}

    ## split the key and value vectors for each head
    network[f"{name}_att_key"] = {
      'class': "split_dims", 'axis': "F", 'dims': (num_heads, key_per_head),
      'from': [f"{name}_att_key0"],  # (B, enc-T, H, D/H)
    }

    network[f"{name}_att_value"] = {
      'class': "split_dims", 'axis': "F", 'dims': (num_heads, value_per_head),
      'from': [f"{name}_att_value0"],  # (B, enc-T, H, D'/H)
    }

    ## encoder-decoder energy
    ## we have exactly enc-T energy values
    network[f"{name}_att_energy"] = {
      'class': "dot", 'red1': -1, 'red2': -1, 'var1': "T", 'var2': "T?",
      'from': [f"{name}_att_key", f"{name}_att_query"]}  # (B, H, key-T, query-T)

    ## normalize the attention weights (depends on key/query dim.)
    network[f"{name}_att_weights"] = {
      'class': "softmax_over_spatial", 'from': [f"{name}_att_energy"],
      'energy_factor': key_per_head ** energy_factor,  # (B, H, key-T, query-T), key-T is where softmax is performed
    }

    # relative_pe as in transformer xl
    if relative_pe_transformer_xl and not relative_pe and not untied_pe:

      shared_layers = False
      network[f"{name}_att_emb_emb"] = network[f"{name}_att_energy"]

      # (B, enc-T, d_pos)
      assert 'source' in network
      if 'pos' not in network:
        network["pos"] = {
          'class': "positional_encoding",
          'add_to_input': False,
          'from': ["source"],
          'n_out': model_dim
        }
      # network['pos_with_0'] = {
      #   "class": "eval", "from": ["pos"],
      #   "eval": f"tf.slice(tf.concat([tf.expand_dims(tf.tile(tf.reshape([0, 1] * ({model_dim}//2), " \
      #           f"(1, {model_dim})), [tf.shape(source(0))[0], 1]), 1), source(0)], 1), [0, 0, 0], [-1, tf.shape(source(0))[1], -1])"}

      if shared_layers:
        network["att_pos_key0"] = {
          'class': "linear", 'activation': None, 'with_bias': False,
          'from': ['pos'], 'n_out': key_dim,  # (B, enc-T, D) # pos_with_0
          'forward_weights_init': initialization,
        }
        network["att_pos_key"] = {
          'class': "split_dims", 'axis': "F",
          'dims': (num_heads, key_per_head),
          'from': ["att_pos_key0"],  # (B, enc-T, H, D/H)
        }
      else:
        network[f"{name}_att_pos_key0"] = {
          'class': "linear", 'activation': None, 'with_bias': False,
          'from': ['pos'], 'n_out': key_dim,  # (B, enc-T, D) # pos_with_0
          'forward_weights_init': initialization,
        }
        network[f"{name}_att_pos_key"] = {
          'class': "split_dims", 'axis': "F",
          'dims': (num_heads, key_per_head),
          'from': [f"{name}_att_pos_key0"],  # (B, enc-T, H, D/H)
        }

      # (B, enc-T, H, D/H), (B, dec-T, H, D/H) -> (B, H, enc-T, dec-T)
      network[f"{name}_att_emb_pos"] = {
        'class': "dot", 'red1': -1, 'red2': -1, 'var1': "T", 'var2': "T?",
        'from': [f"{name}_att_pos_key", f"{name}_att_query"]
      }

      if shared_layers:
        network[f"{name}_att_emb_pos"]['from'] = ["att_pos_key", f"{name}_att_query"]

      # (B, H, enc-T, dec-T)
      network[f"{name}_att_emb_pos_shifted"] = {
        'class': "eval",
        'eval': "self.network.get_config().typed_value('rel_shift')(source(0))",
        'from': [f"{name}_att_emb_pos"],
        'out_type': {'shape': (num_heads, None, None),
                     'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1}
      }

      # (B, 4, F)
      if shared_layers:
        network["pos_emb_bias"] = {
          'class': "variable",
          'shape': (num_heads, key_per_head),
          'add_time_axis': True,
          'init': DEFAULT_INIT
        }
      else:
        network[f"{name}_pos_emb_bias"] = {
          'class': "variable",
          'shape': (num_heads, key_per_head),
          'add_time_axis': True,
          'init': DEFAULT_INIT
        }
      # (B, enc-T, H, D / H), (B, 1, H, D / H) --> (B, H, enc-T, dec-T=1)
      network[f"{name}_att_pos_emb"] = {
        'class': "dot", 'red1': -1, 'red2': -1, 'var1': "T", 'var2': "T?",
        'from': [f"{name}_att_key", f"{name}_pos_emb_bias"],
        'out_type': {'shape': (num_heads, None, 1)}
                     #'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1, "dim": num_heads}
      }

      if shared_layers:
        network[f"{name}_att_pos_emb"]['from'] = [f"{name}_att_key", "pos_emb_bias"]

      network[f"{name}_att_pos_emb_tiled"] = {
        'class': "rel_shift",
        'rel_shift': False,
        'from': [f"{name}_att_pos_emb"],
        'out_type': {'shape': (num_heads, None, None),
                     'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1, 'dim': num_heads}

      }
      if shared_layers:
        network["pos_pos_bias"] = {
          'class': "variable",
          'shape': (num_heads, key_per_head),  # (B, d, 4)
          'add_time_axis': True,
          'init': DEFAULT_INIT
        }

        # (B, enc - T, H, D / H), (B, 1, H, D / H) --> (B, H, enc-T, dec-T = 1)
        network["att_pos_pos"] = {
          'class': "dot", 'red1': -1, 'red2': -1, 'var1': "T", 'var2': "T?",
          'from': ["att_pos_key", "pos_pos_bias"],
          'out_type': {'shape': (num_heads, None, 1)}
          # 'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1, "dim": num_heads}
        }

        # (B, H, T, T')
        network["att_pos_pos_shifted"] = {
          'class': "rel_shift",
          'from': ["att_pos_pos"],
          'out_type': {'shape': (num_heads, None, None),
                       'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1, 'dim': num_heads}

        }
      else:
        network[f"{name}_pos_pos_bias"] = {
          'class': "variable",
          'shape': (num_heads, key_per_head), #(B, d, 4)
          'add_time_axis': True,
          'init': DEFAULT_INIT
        }

        # (B, enc - T, H, D / H), (B, 1, H, D / H) --> (B, H, enc-T, dec-T = 1)
        network[f"{name}_att_pos_pos"] = {
          'class': "dot", 'red1': -1, 'red2': -1, 'var1': "T", 'var2': "T?",
          'from': [f"{name}_att_pos_key", f"{name}_pos_pos_bias"],
          'out_type': {'shape': (num_heads, None, 1)}
                       #'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1, "dim": num_heads}
        }

        # (B, H, T, T')
        network[f"{name}_att_pos_pos_shifted"] = {
          'class': "rel_shift",
          'from': [f"{name}_att_pos_pos"],
          'out_type': {'shape': (num_heads, None, None),
                       'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1, 'dim': num_heads}

        }

      network[f"{name}_att_energy"] = {
        'class': "combine",
        'kind': "add",
        'from': [f"{name}_att_emb_emb", f"{name}_att_pos_emb_tiled",
                 f"{name}_att_emb_pos_shifted", f"{name}_att_pos_pos_shifted"]
      }
      if shared_layers:
        network[f"{name}_att_energy"]['from'] = [f"{name}_att_emb_emb", f"{name}_att_pos_emb_tiled",
                 f"{name}_att_emb_pos_shifted", "att_pos_pos_shifted"]

    if untied_pe and not relative_pe:
      assert 'source' in network
      if 'pos' not in network:
        network["pos"] = {
          'class': "positional_encoding",
          'add_to_input': False,
          'from': ["source"],
          'n_out': model_dim
        }
      # shared
      if False:
        if 'att_pos_query0' not in network:
          network["att_pos_query0"] = {
            'class': "linear", 'activation': None, 'with_bias': False,
            'from': ["pos"], 'n_out': key_dim,
            'forward_weights_init': initialization}

          network["att_pos_query"] = {
            'class': "split_dims", 'axis': "F",
            'dims': (num_heads, key_per_head),  # (B, T, H, D/H)
            'from': ["att_pos_query0"],
          }

          network["att_pos_key0"] = {
            'class': "linear", 'activation': None, 'with_bias': False,
            'from': ["pos"], 'n_out': key_dim,  # (B, enc-T, D)
            'forward_weights_init': initialization,
          }
          network["att_pos_key"] = {
            'class': "split_dims", 'axis': "F",
            'dims': (num_heads, key_per_head),
            'from': ["att_pos_key0"],  # (B, enc-T, H, D/H)
          }

          network["att_pos_energy"] = {
            'class': "dot", 'red1': -1, 'red2': -1, 'var1': "T", 'var2': "T?",
            'from': ["att_pos_key", "att_pos_query"]}

        network[f"{name}_att_energy_with_pos_corr"] = {
          'class': "combine",
          'kind': "add",
          'from': [f"{name}_att_energy", "att_pos_energy"]
        }

      # per layer
      if False:
        network[f"{name}_att_pos_query0"] = {
          'class': "linear", 'activation': None, 'with_bias': False,
          'from': ["pos"], 'n_out': key_dim,
          'forward_weights_init': initialization}

        network[f"{name}_att_pos_query"] = {
          'class': "split_dims", 'axis': "F",
          'dims': (num_heads, key_per_head),  # (B, T, H, D/H)
          'from': [f"{name}_att_pos_query0"],
        }

        network[f"{name}_att_pos_key0"] = {
          'class': "linear", 'activation': None, 'with_bias': False,
          'from': ["pos"], 'n_out': key_dim,  # (B, enc-T, D)
          'forward_weights_init': initialization,
        }
        network[f"{name}_att_pos_key"] = {
          'class': "split_dims", 'axis': "F",
          'dims': (num_heads, key_per_head),
          'from': [f"{name}_att_pos_key0"],  # (B, enc-T, H, D/H)
        }

        network[f"{name}_att_pos_energy"] = {
          'class': "dot", 'red1': -1, 'red2': -1, 'var1': "T", 'var2': "T?",
          'from': [f"{name}_att_pos_key", f"{name}_att_pos_query"]}

        network[f"{name}_att_energy_with_pos_corr"] = {
          'class': "combine",
          'kind': "add",
          'from': [f"{name}_att_energy", f"{name}_att_pos_energy"]
        }

      # with corrected normalization factor
      if True:
        network[f"{name}_att_pos_query0"] = {
          'class': "linear", 'activation': None, 'with_bias': False,
          'from': ["pos"], 'n_out': key_dim,
          'forward_weights_init': initialization}

        network[f"{name}_att_pos_query"] = {
          'class': "split_dims", 'axis': "F",
          'dims': (num_heads, key_per_head),  # (B, T, H, D/H)
          'from': [f"{name}_att_pos_query0"],
        }

        network[f"{name}_att_pos_key0"] = {
          'class': "linear", 'activation': None, 'with_bias': False,
          'from': ["pos"], 'n_out': key_dim,  # (B, enc-T, D)
          'forward_weights_init': initialization,
        }
        network[f"{name}_att_pos_key"] = {
          'class': "split_dims", 'axis': "F",
          'dims': (num_heads, key_per_head),
          'from': [f"{name}_att_pos_key0"],  # (B, enc-T, H, D/H)
        }

        network[f"{name}_att_pos_energy"] = {
          'class': "dot", 'red1': -1, 'red2': -1, 'var1': "T", 'var2': "T?",
          'from': [f"{name}_att_pos_key", f"{name}_att_pos_query"]}

        network[f"{name}_att_energy_with_pos_corr"] = {
          'class': "combine",
          'kind': "add",
          'from': [f"{name}_att_energy", f"{name}_att_pos_energy"]
        }

        network[f"{name}_att_weights"]['energy_factor'] = (2 * key_per_head) ** energy_factor

      # scale per layer
      if False:
        if 'att_pos_query0' not in network:
          network["att_pos_query0"] = {
            'class': "linear", 'activation': None, 'with_bias': False,
            'from': ["pos"], 'n_out': key_dim,
            'forward_weights_init': initialization}

          network["att_pos_query"] = {
            'class': "split_dims", 'axis': "F",
            'dims': (num_heads, key_per_head),  # (B, T, H, D/H)
            'from': ["att_pos_query0"],
          }

          network["att_pos_key0"] = {
            'class': "linear", 'activation': None, 'with_bias': False,
            'from': ["pos"], 'n_out': key_dim,  # (B, enc-T, D)
            'forward_weights_init': initialization,
          }
          network["att_pos_key"] = {
            'class': "split_dims", 'axis': "F",
            'dims': (num_heads, key_per_head),
            'from': ["att_pos_key0"],  # (B, enc-T, H, D/H)
          }

          network["att_pos_energy"] = {
            'class': "dot", 'red1': -1, 'red2': -1, 'var1': "T", 'var2': "T?",
            'from': ["att_pos_key", "att_pos_query"]}

        network[f"{name}_att_pos_energy_scale"] = {
          'class': 'variable',
          'shape': (num_heads,),
          'init': 1.0,
          'add_batch_axis': False
        }
        network[f"{name}_att_energy_with_pos_corr"] = {
          'class': "eval",
          'eval': f"tf.add(source(0), tf.multiply(source(1), tf.reshape(source(2), (1, {num_heads}, 1, 1))))",
          'from': [f"{name}_att_energy", "att_pos_energy", f"{name}_att_pos_energy_scale"]
        }

      network[f"{name}_att_weights"]["from"] = [f"{name}_att_energy_with_pos_corr"]

    ## attention weights dropout
    network[f"{name}_att_weights_drop"] = {
      'class': "dropout", 'dropout_noise_shape': {'*': None},
      'dropout': sa_dropout, 'from': [f"{name}_att_weights"],
    }

    ## now we have an attention weight value for each encoder-side output
    ## we get per head one vector
    network[f"{name}_att0"] = {
      'class': "generic_attention", 'weights': f"{name}_att_weights_drop",
      'base': f"{name}_att_value",  # (B, T, H, V) #(B, H, V)
    }

    network[f"{name}_self_att_att"] = {
      'class': "merge_dims", 'axes': "static",  # "static"
      'from': [f"{name}_att0"]
    }

    ## not sure, if this works
    if windowing:
      #hard masking
      if not gauss_window:
        eval_win_size = f'tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(tf.constant({window_size}, dtype=tf.int32), axis = -1), axis = -1), '\
                        f'[1, tf.shape(source(0))[-2], tf.shape(source(0))[-1]]), 0)'
        eval_win_start = f'tf.expand_dims(tf.map_fn(fn = lambda t: tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-1]), 0), '\
                         f'[tf.shape(source(0))[2], 1]) - t, elems=tf.constant({window_size}, dtype=tf.int32)//2), 0)'


        # eval_encoderT_pos = 'tf.tile(tf.expand_dims(tf.expand_dims(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-2]), -1), '\
        #   '[1, tf.shape(source(0))[-1]]), 0), 0), [1, tf.shape(source(0))[1], 1, 1])'

        eval_encoderT_pos = 'tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-2]), -1), '\
                            '[tf.shape(source(0))[1], tf.shape(source(0))[-1]]), tf.shape(source(0))[1:]), 0)'

        # without batch dim.
        #eval_masking = 'tf.logical_and(tf.less_equal(source(0), source(1)), tf.greater_equal(source(0), source(2)))'
        eval_masking = 'tf.tile(tf.logical_and(tf.less_equal(source(0), source(1)), tf.greater_equal(source(0), source(2))), '\
                       '[tf.shape(source(3))[0], 1, 1, 1])'

        network[f"{name}_att_energy"]['out_type'] = {'time_dim_axis': 3}
        network[f"{name}_win_size"] = {
          'class': 'eval',
          'eval': eval_win_size,
          'from': [f"{name}_att_energy"],
          'out_type': {'dtype': 'int32'}
        }

        network[f"{name}_win_start"] = {
          'class': 'eval',
          'eval': eval_win_start,
          'from': [f"{name}_att_energy"],
          'out_type': {'dtype': 'int32'}
        }

        ## normalize the attention weights (depends on key/query dim.)
        # network[f"{name}_att_weights"]['window_start'] = f"{name}_win_start"
        # network[f"{name}_att_weights"]['window_size'] = f"{name}_win_size"

        network[f"{name}_win_end"] = {
          'class': 'combine',
          'from': [f"{name}_win_start", f"{name}_win_size"],
          'kind': 'add'
        }

        network[f"{name}_encoderT_pos"] = {
          'class': 'eval',
          'eval': eval_encoderT_pos,
          'from': [f"{name}_att_energy"],
          'out_type': {'dtype': 'int32'}
        }

        network[f"{name}_masking"] = {
          'class': 'eval',
          'eval': eval_masking,
          'from': [f"{name}_encoderT_pos", f"{name}_win_end", f"{name}_win_start", f"{name}_att_energy"],
          'out_type': {'dtype': 'bool'}
        }

        network[f"{name}_att_energy_masked"] = {
          'class': 'eval',
          'eval': f"tf.where(source(0), source(1), "\
          f"tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant(float('-inf')), 0), 0), 0), 0), tf.shape(source(1))))",
          'from': [f"{name}_masking", f"{name}_att_energy"],
          'out_type': {'dtype': 'float32'}
        }
      #soft masking: Gaussian window
      else:
        eval_key_pos = 'tf.cast(tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-2]), -1), ' \
                            '[tf.shape(source(0))[1], tf.shape(source(0))[-1]]), tf.shape(source(0))[1:]), 0), "float32")'
        eval_query_pos = f'tf.cast(tf.expand_dims(tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-1]), 0), '\
          f'[tf.shape(source(0))[-2], 1]), 0), [{num_heads}, 1, 1]), 0), "float32")'

        network[f"{name}_key_pos"] = {
          'class': 'eval',
          'eval': eval_key_pos,
          'from': [f"{name}_att_energy"],
          'out_type': {'dtype': 'float32'}
        }
        network[f"{name}_query_pos"] = {
          'class': 'eval',
          'eval': eval_query_pos,
          'from': [f"{name}_att_energy"],
          'out_type': {'dtype': 'float32'}
        }
        network[f"{name}_std_for_gaussian_window"] = {
          'class': 'variable',
          'init': window_size[0],
          'shape': (num_heads,)
        }

        network[f"{name}_masking"] = {
          'class': 'eval',
          'eval': f'{half_ratio} * tf.square(source(0) - source(1)) / tf.reshape(tf.square(source(2)), [tf.shape(source(3))[0], {num_heads}, 1, 1])',
          'from': [f"{name}_query_pos", f"{name}_key_pos", f"{name}_std_for_gaussian_window", f"{name}_att_energy"],
          'out_type': {'dtype': 'float32'}
        }

        network[f"{name}_att_energy_masked"] = {
          'class': 'combine',
          'kind': 'add',
          'from': [f"{name}_masking", f"{name}_att_energy"],
          'out_type': {'dtype': 'float32'}
        }

      network[f"{name}_att_weights"]['from'] = [f"{name}_att_energy_masked"]
      network[f"{name}_att_weights"]['use_time_mask'] = False


  else:
    network[f"{name}_self_att_att"] = {
      'class': "self_attention", 'num_heads': num_heads,
      'total_key_dim': key_dim, 'n_out': value_dim,
      'from': [f"{name}_self_att_laynorm"],
      'attention_left_only': attention_left_only,
      'attention_dropout': sa_dropout,
      'forward_weights_init': initialization,
    }

    if relative_pe:
      network[f"{name}_rel_pos"] = {
          "class": "relative_positional_encoding",
          "from": [f"{name}_self_att_laynorm"],
          "fixed": fixed,
          "clipping": clipping,
          "n_out": key_dim // num_heads,
          "forward_weights_init": initialization
      }
      network[f"{name}_self_att_att"]["key_shift"] = f"{name}_rel_pos"
  if linear_mapping:
    network[f"{name}_self_att_lin"] = {
      'class': "linear", 'activation': None, 'with_bias': linear_mapping_bias,
      'from': [f"{name}_self_att_att"], 'n_out': model_dim,
      'forward_weights_init': initialization
    }
    network[f"{name}_self_att_drop"] = {
      'class': "dropout", 'dropout': sa_post_dropout,
      'from': [f"{name}_self_att_lin"]
    }
  else:
    network[f"{name}_self_att_drop"] = {
      'class': "dropout", 'dropout': sa_post_dropout,
      'from': [f"{name}_self_att_att"]
    }

  network[f"{name}_self_att_out"] = {
    'class': "combine", 'kind': "add",
    'from': [f"{name}_ff1_out", f"{name}_self_att_drop"],
    'n_out': model_dim
  }
  ## convolution module
  network[f"{name}_conv_laynorm"] = {'class': "layer_norm",
                                     'from': [f"{name}_self_att_out"]}

  ## d --> 2d for GLU activation
  ## can linear as an alternative to pointwise conv.?
  network[f"{name}_conv_pointwise1"] = {
    'class': "linear", 'activation': None, 'with_bias': False,
    'from': [f"{name}_conv_laynorm"], 'n_out': 2 * model_dim,
    'forward_weights_init': initialization
  }



  ## (batch, time, feature)
  network[f"{name}_conv_GLU"] = {
    'class': "gating",
    'activation': "identity",
    'from': [f"{name}_conv_pointwise1"]
  }

  out_layer_name = f"{name}_conv_GLU"

  if se_pos == "after_first_conv":
    # TODO: implement
    inpl = f"{name}_conv_GLU"
    out_layer_name = add_SE_block(network, inpl, name, se_act)

  if normal_conv:
    network[f"{name}_conv_expanded"] = {
      "class": "split_dims", "axis": "F", "dims": (-1, 1),
      "from": [out_layer_name]
    }
    ## (T, F, 1)
    network[f"{name}_conv_normal"] = {
      "class": "conv",
      "from": [f"{name}_conv_expanded"], "padding": "same",
      "filter_size": (kernel_size, kernel_size_for_feature),
      "n_out": output_channels, "activation": None, "with_bias": True #model_dim//kernel_size
    }
    network[f"{name}_conv_normal_flattened"] = {
      "class": "merge_dims",
      "from": [f"{name}_conv_normal"],
      "axes": "static"
    }
    ## parameter intensiv
    network[f"{name}_conv_transformed"] = {
      'class': "linear",
      'activation': None,
      'with_bias': False,
      'forward_weights_init': initialization,
      'n_out': model_dim,
      "from": [f"{name}_conv_normal_flattened"]
    }

    network[f"{name}_conv_batchnorm"] = {
      'class': "batch_norm",
      'from': [f"{name}_conv_transformed"]
    }
  else:

    network[f"{name}_conv_depthwise"] = {
      'activation': None,
      'class': 'conv',
      'filter_size': (kernel_size,),
      'from': [out_layer_name],
      'groups': model_dim,
      'n_out': model_dim,
      'padding': 'same',
      'with_bias': True
    }

    out_layer_name = f"{name}_conv_depthwise"

    if se_pos == "after_depthwise_conv":
      # TODO: implement
      inpl = f"{name}_conv_depthwise"
      out_layer_name = add_SE_block(network, inpl, name, se_act)

    network[f"{name}_conv_batchnorm"] = {
      'class': "batch_norm",
      'from': [out_layer_name]
    }

  network[f"{name}_conv_act"] = {
    'class': "activation",
    'activation': "swish",
    'from': [f"{name}_conv_batchnorm"]
  }

  network[f"{name}_conv_pointwise2"] = {
    'class': "linear", 'activation': None, 'with_bias': False,
    'from': [f"{name}_conv_act"], 'n_out': model_dim,
    'forward_weights_init': initialization
  }

  out_layer_name = f"{name}_conv_pointwise2"

  if se_pos == "after_sec_conv":
    # TODO: implement
    inpl = f"{name}_conv_pointwise2"
    out_layer_name = add_SE_block(network, inpl, name, se_act)


  network[f"{name}_conv_dropout"] = {
    'class': "dropout", 'dropout': conv_post_dropout,
    'from': [out_layer_name],
  }
  network[f"{name}_conv_output"] = {
    'class': "combine", 'kind': "add",
    'from': [f"{name}_self_att_out", f"{name}_conv_dropout"], 'n_out': model_dim,

  }

  ## second ffn layer
  network[f"{name}_ff2_laynorm"] = {'class': "layer_norm",
                                    'from': [f"{name}_conv_output"]}
  network[f"{name}_ff2_conv1"] = {
    'class': "linear", 'activation': ff_activation, 'with_bias': True,
    'from': [f"{name}_ff2_laynorm"],
    'n_out': ff_dim, 'forward_weights_init': initialization
  }

  network[f"{name}_ff2_conv2"] = {
    'class': "linear", 'activation': None, 'with_bias': True,
    'from': [f"{name}_ff2_conv1"], 'dropout': ff_activation_dropout,
    'n_out': model_dim, 'forward_weights_init': initialization
  }
  network[f"{name}_ff2_drop"] = {'class': "dropout",
                                 'dropout': ff_post_dropout,
                                 'from': [f"{name}_ff2_conv2"]}

  network[f"{name}_ff2_drop_half"] = {
    'class': "eval",
    'eval': f"{half_ratio} * source(0)",
    'from': [f"{name}_ff2_drop"]
  }
  network[f"{name}_ff2_out"] = {
    'class': "combine", 'kind': "add",
    'from': [f"{name}_conv_output", f"{name}_ff2_drop_half"]
  }

  if switch:
    network[f"{name}_conv_output"]['from'] = [f"{name}_ff1_out", f"{name}_conv_dropout"]
    network[f"{name}_conv_laynorm"]['from'] = [f"{name}_ff1_out"]

    network[f"{name}_self_att_laynorm"]['from'] = [f"{name}_conv_output"]
    network[f"{name}_self_att_out"]['from'] = [f"{name}_conv_output", f"{name}_self_att_drop"]

    network[f"{name}_ff2_laynorm"]['from'] = [f"{name}_self_att_out"]
    network[f"{name}_ff2_out"]['from'] = [f"{name}_self_att_out", f"{name}_ff2_drop_half"]

  ## final layer norm
  if end_layernorm:
    network[f"{name}"] = {
      'class': "layer_norm",
      'from': [f"{name}_ff2_out"]
    }
  else:
    network[f"{name}"] = {
      'class': "copy",
      'from': [f"{name}_ff2_out"]
    }
